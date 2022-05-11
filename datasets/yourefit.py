# -*- coding: utf-8 -*-

"""
YouRefIt referring image PyTorch dataset.
"""
import pandas as pd
from PIL import Image
import os
import sys
import cv2
import json
import uuid
import tqdm
import math
import torch
import random
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from collections import OrderedDict

import util.dist

sys.path.append('.')
import operator
import argparse
import collections
import logging
import json
import pickle5 as pickle
import re
import util.dist as dist
from transformers import RobertaTokenizerFast

sys.path.append('./datasets')
from .yourefit_token import match_pos
import copy
from util.box_ops import generalized_box_iou
from magic_numbers import *
import pandas as pd

import torch
if SAVE_EVALUATION_PREDICTIONS and SAVE_CLIP_SCORES:
    import clip
from PIL import Image

import torchvision
import torchvision.transforms as T

transform = T.ToPILImage()

cv2.setNumThreads(0)

if USE_MDETR_PREDICTIONS_AS_GROUNDTRUTHS:
    mdetr_predictions = pd.read_csv(MDETR_PREDICTION_PATH)
else:
    mdetr_predictions = None
eye_to_fingertip_annotation_df_train = pd.read_csv(
    EYE_TO_FINGERTIP_ANNOTATION_TRAIN_PATH)
eye_to_fingertip_annotation_df_valid = pd.read_csv(
    EYE_TO_FINGERTIP_ANNOTATION_VALID_PATH)


def progressBar(i, max, text):
    """
    Print a progress bar during training.
    :param i: index of current iteration/epoch.
    :param max: max number of iterations/epochs.
    :param text: Text to print on the right of the progress bar.
    :return: None
    """
    bar_size = 60
    j = (i + 1) / max
    sys.stdout.write('\r')
    sys.stdout.write(
        f"[{'=' * int(bar_size * j):{bar_size}s}] {int(100 * j)}%  {text}")
    sys.stdout.flush()


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)

    result = positive_map / (positive_map.sum(-1)[:, None] + 1e-6)
    return result


class DatasetNotFoundError(Exception):
    pass


class ReferDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'yourefit': {'splits': ('train', 'val', 'test')}
    }

    def __init__(self, data_root, split_root='data', dataset='yourefit',
                 transform=None, augment=False, device=None, return_idx=False,
                 testmode=False,
                 split='train', max_query_len=128, lstm=False,
                 bert_model='bert-base-uncased', args=None):
        self.images = []
        self.image_files = {}
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.query_len = max_query_len
        self.lstm = lstm
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.device = device
        self.augment = augment
        self.return_idx = return_idx
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            args.text_encoder_type)
        if self.dataset == 'yourefit':
            self.dataset_root = osp.join(self.data_root, 'yourefit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.inpaint_dir = osp.join(self.dataset_root, INPAINT_DIR)
            self.arm_dir = osp.join(self.dataset_root, 'arms.json')
            with open(self.arm_dir, "r") as f:
                self.arm_data = json.load(f)
        else:  ## refcoco, etc.
            pass
        if not self.exists_dataset():
            print('Please download index cache to data folder')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]

        for split in splits:
            imgset_file = osp.join(self.dataset_root,
                                   '{0}_id.txt'.format(split))

            with open(imgset_file, 'r') as f:
                while True:
                    img_name = f.readline()
                    if img_name:
                        if USE_MDETR_PREDICTIONS_AS_GROUNDTRUTHS and REPLACE_ARM_WITH_EYE_TO_FINGERTIP:
                            raise NotImplementedError()
                        if USE_MDETR_PREDICTIONS_AS_GROUNDTRUTHS and self.split == 'train':
                            # Only for the training set:
                            # Ignore images without mdetr foreground predictions
                            # Check if current image has mdetr foreground prediction.
                            # If it does, append its name into image list.
                            has_mdetr_foreground_prediction = \
                                mdetr_predictions[
                                    mdetr_predictions['img_name'] == img_name[
                                                                     :-1]].shape[
                                    0] > 0
                            if has_mdetr_foreground_prediction:
                                self.images.append(img_name[:-1])
                        elif REPLACE_ARM_WITH_EYE_TO_FINGERTIP and self.split == 'train':
                            has_eye_to_fingertip_annotation = \
                                eye_to_fingertip_annotation_df_train[
                                    eye_to_fingertip_annotation_df_train[
                                        'name'] == img_name[:-1]].shape[0] > 0
                            if has_eye_to_fingertip_annotation:
                                self.images.append(img_name[:-1])
                        else:
                            self.images.append(img_name[:-1])
                    else:
                        break
            f.close()

        for split_idx in range(len(self.images)):
            self.image_files[self.images[split_idx]] = split_idx

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item_sentence(self, idx):
        img_name = self.images[idx]
        pickle_file = osp.join(osp.join(self.dataset_root, 'pickle'),
                               img_name + '.p')
        pick = pickle.load(open(pickle_file, "rb"))
        sentence = pick['anno_sentence']
        return sentence

    def pull_item_box(self, idx, return_img=False):

        # img_file, _, bbox, phrase, attri = self.images[idx]
        ## box format: to x1y1x2y2
        img_name = self.images[idx]
        pickle_file = osp.join(osp.join(self.dataset_root, 'pickle'),
                               img_name + '.p')
        pick = pickle.load(open(pickle_file, "rb"))
        bbox = pick['bbox']
        bbox = np.array(bbox, dtype=int)  # x1y1x2y2
        img = None
        if return_img:
            img_path = osp.join(self.im_dir, img_name + '.jpg')
            img = cv2.imread(img_path)
        return bbox, img_name, img

    def pull_item(self, idx):

        img_name = self.images[idx]
        ## box format: to x1y1x2y2
        pickle_file = osp.join(osp.join(self.dataset_root, 'pickle'),
                               img_name + '.p')
        pick = pickle.load(open(pickle_file, "rb"))
        bbox = pick['bbox']
        target_word = pick['anno_target']
        phrase = pick['anno_sentence']
        token_pos = match_pos(phrase, target_word)
        token_pos = [token_pos]
        bbox = np.array(bbox, dtype=int)  # x1y1x2y2
        if not REPLACE_IMAGES_WITH_INPAINT:
            img_path = osp.join(self.im_dir, img_name + '.jpg')
            img = Image.open(img_path).convert('RGB')
        else:
            try:
                img_path = osp.join(self.inpaint_dir, img_name + '.jpg')
                img = Image.open(img_path).convert('RGB')
            except:
                if util.dist.get_rank() == 0:
                    print()
                    print(
                        'Missing inpaint for ' + img_name + ', using original image instead')
                    print()
                img_path = osp.join(self.im_dir, img_name + '.jpg')
                img = Image.open(img_path).convert('RGB')
        # replace bbox with MDETR predictions
        if USE_MDETR_PREDICTIONS_AS_GROUNDTRUTHS and self.split == 'train':
            width = img.width
            height = img.height
            df_for_current_image = mdetr_predictions[
                mdetr_predictions['img_name'] == img_name]
            if df_for_current_image.shape[0] > 0:
                xmin, ymin, xmax, ymax = [df_for_current_image.xmin,
                                          df_for_current_image.ymin,
                                          df_for_current_image.xmax,
                                          df_for_current_image.ymax]
                xmin, ymin, xmax, ymax = [xmin * width, ymin * height,
                                          xmax * width, ymax * height]
                xmin, ymin, xmax, ymax = [int(xmin), int(ymin), int(xmax),
                                          int(ymax)]
                bbox = np.array([xmin, ymin, xmax, ymax])
            else:
                raise RuntimeError(
                    'Using MDETR predictions as groundtruths, but current image does not have any mdetr foreground prediction')

        htmapdir = self.im_dir.replace('images', 'paf')
        htmapfile = img_name + '_rendered.png'
        htmap_path = osp.join(htmapdir, htmapfile)
        htmap = cv2.imread(htmap_path)
        ht = np.asarray(htmap)
        ht = np.mean(ht, axis=2)

        ptdir = self.im_dir.replace('images', 'saliency')
        ptfile = img_name + '.jpeg'
        pt_path = osp.join(ptdir, ptfile)
        pt = cv2.imread(pt_path)
        pt = cv2.resize(pt, (256, 256))
        pt = np.reshape(pt, (3, 256, 256))
        arm = self.arm_data[img_name]

        # Replace arm coordinates with eye and fingertip coordinates
        if REPLACE_ARM_WITH_EYE_TO_FINGERTIP:
            if self.dataset == 'yourefit':
                # Determine the split of dataset
                if self.split == 'train':
                    df = eye_to_fingertip_annotation_df_train
                elif self.split == 'val':
                    df = eye_to_fingertip_annotation_df_valid
                else:
                    raise NotImplementedError(
                        'replace arm with eye to fingertip is only implemented for yourefit train and valid splits')
                # Find out the coordinates of eye and fingertip
                df_for_current_image = df[df['name'] == img_name]
                if df_for_current_image.shape[0] == 0:
                    if self.split == 'train':
                        # For the training set, raise error if an image does not have eye to fingertip annotations.
                        raise RuntimeError(
                            'Missing eye to fingertip annotation for image: ' + img_name)
                    else:
                        # For the valid set, set coordinates to negative values to signal missing annotations.
                        eye_x = -img.width
                        eye_y = -img.height
                        fingertip_x = -img.width
                        fingertip_y = -img.height
                else:
                    # If current image has eye to fingertip annotation, get the coordinates from annotations
                    eye_x = df_for_current_image.iloc[0].eye_x
                    eye_y = df_for_current_image.iloc[0].eye_y
                    fingertip_x = df_for_current_image.iloc[0].fingertip_x
                    fingertip_y = df_for_current_image.iloc[0].fingertip_y
                # Replace arm coordinates with eye and fingertip coordinates
                arm = [eye_x, eye_y, fingertip_x, fingertip_y]
            else:
                raise NotImplementedError(
                    'replace arm with eye to fingertip not implemented for current dataset')

        return img, pt, ht, phrase, bbox, token_pos, arm, img_name

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, pt, ht, phrase, bbox, token_pos, arm, img_name = self.pull_item(
            idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()
        ## seems a bug in torch transformation resize, so separate in advance
        w, h = img.size
        target = {}
        target['orig_size'] = torch.tensor([h, w])
        target['size'] = torch.tensor([h, w])
        target['boxes'] = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0)
        target['caption'] = phrase
        target['img_name'] = img_name
        target['image_id'] = torch.tensor([idx])
        target['tokens_positive'] = token_pos
        target['labels'] = torch.tensor([1])
        target['arm'] = torch.tensor(arm).flatten()
        assert target['arm'].shape[0] == 4

        assert len(target["boxes"]) == len(target["tokens_positive"])
        # TODO: check if 'tokenized' can be used as embedded text
        tokenized = self.tokenizer(phrase, return_tensors="pt")
        target["positive_map"] = create_positive_map(tokenized,
                                                     target["tokens_positive"])
        target['ht_map'] = torch.tensor(ht).unsqueeze(0)
        if self.transform is not None:
            img, target = self.transform(img, target)
        target['dataset_name'] = 'yourefit'
        return img, target


class YouRefItEvaluator_old(object):
    def __init__(self, ref_dataset, iou_types, k=(1, 2, 10), thresh_iou=0.25,
                 draw=False):
        assert isinstance(k, (list, tuple))
        ref_dataset = copy.deepcopy(ref_dataset)
        self.refexp_gt = ref_dataset
        self.iou_types = iou_types
        self.img_ids = self.refexp_gt.image_files.keys()
        self.predictions = {}
        self.k = k
        self.thresh_iou = thresh_iou
        self.draw = draw

    def accumulate(self):
        pass

    def update(self, predictions):
        self.predictions.update(predictions)

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        merged_predictions = {}
        for p in all_predictions:
            merged_predictions.update(p)
        self.predictions = merged_predictions

    def summarize(self):
        if dist.is_main_process():
            dataset2score = {
                "yourefit": {k: 0.0 for k in self.k},
            }

            dataset2count = {"yourefit": 0.0}

            for image_id in self.predictions.keys():
                print('Evaluating{0}...'.format(image_id))

                bbox, img_name, img = self.refexp_gt.pull_item_box(image_id,
                                                                   self.draw)

                prediction = self.predictions[image_id]
                assert prediction is not None
                sorted_scores_boxes = sorted(
                    zip(prediction["scores"].tolist(),
                        prediction["boxes"].tolist()), reverse=True
                )
                sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
                sorted_boxes = torch.cat(
                    [torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
                converted_bbox = bbox

                giou = generalized_box_iou(sorted_boxes,
                                           torch.as_tensor(converted_bbox).view(
                                               -1, 4))
                for k in self.k:
                    if max(giou[:k]) >= self.thresh_iou:
                        dataset2score["yourefit"][k] += 1.0
                dataset2count['yourefit'] += 1.0

            for key, value in dataset2score.items():
                for k in self.k:
                    try:
                        value[k] /= dataset2count[key]
                    except:
                        pass
            results = {}
            for key, value in dataset2score.items():
                results[key] = sorted([v for k, v in value.items()])
                print(
                    f" Dataset: {key} - Precision @ 1, 5, 10: {results[key]} \n")

            return results
        return None


def draw_box(img, box, img_name, arm=None, gt_box=None, box1=None):
    pt1 = (int(box[0]), int(box[1]))
    pt2 = (int(box[2]), int(box[3]))
    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 4)  # green
    if arm is not None:
        pt3 = (int(arm[0]), int(arm[1]))
        pt4 = (int(arm[2]), int(arm[3]))
        cv2.line(img, pt3, pt4, (0, 0, 255), 5)  # red
    if gt_box is not None:
        pt5 = (int(gt_box[0]), int(gt_box[1]))
        pt6 = (int(gt_box[2]), int(gt_box[3]))
        cv2.rectangle(img, pt5, pt6, (255, 0, 0), 4)  # blue
    if box1 is not None:  # second best box
        pt7 = (int(box1[0]), int(box1[1]))
        pt8 = (int(box1[2]), int(box1[3]))
        cv2.rectangle(img, pt7, pt8, (255, 255, 0), 4)  # light blue
    cv2.imwrite(img_name, img)


import torch.nn.functional as F


def aligned_score(arm, box):
    arm_tensor = torch.Tensor(arm[2:]) - torch.Tensor(arm[:2])
    box_center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
    box_tensor = torch.Tensor(box_center) - torch.Tensor(arm[:2])
    cos_sim = F.cosine_similarity(arm_tensor, box_tensor, dim=0)
    return cos_sim


def aligned_scores(arm, boxes):
    arm_tensor = torch.Tensor(arm[2:]) - torch.Tensor(arm[:2])
    box_center = [(boxes[:, 0] + boxes[:, 2]) / 2,
                  (boxes[:, 1] + boxes[:, 3]) / 2]
    box_tensor = torch.vstack(box_center).transpose(0, 1) - torch.Tensor(
        arm[:2])
    cos_sim = F.cosine_similarity(arm_tensor, box_tensor, dim=1)
    return cos_sim


class YouRefItEvaluator(object):
    def __init__(self, ref_dataset, iou_types, k=1,
                 thresh_iou=(0.25, 0.5, 0.75), draw=True):
        ref_dataset = copy.deepcopy(ref_dataset)
        self.refexp_gt = ref_dataset
        self.iou_types = iou_types
        self.img_ids = self.refexp_gt.image_files.keys()
        self.predictions = {}
        self.k = k
        self.thresh_iou = thresh_iou
        self.draw = draw

    def accumulate(self):
        pass

    def update(self, predictions):
        self.predictions.update(predictions)

    def synchronize_between_processes(self):
        all_predictions = dist.all_gather(self.predictions)
        merged_predictions = {}
        for p in all_predictions:
            merged_predictions.update(p)
        self.predictions = merged_predictions

    def summarize(self):
        if dist.is_main_process():
            dataset2score = {
                "yourefit": {thresh_iou: 0.0 for thresh_iou in self.thresh_iou},
            }

            dataset2count = {"yourefit": 0.0}

            arm_token_pair = {}

            image_name_list = []
            box_xmin_list = []
            box_ymin_list = []
            box_xmax_list = []
            box_ymax_list = []
            arm_xmin_list = []
            arm_ymin_list = []
            arm_xmax_list = []
            arm_ymax_list = []
            box_score_list = []
            giou_list = []
            align_cost_list = []
            clip_score_list = []
            sentence_list = []

            total_num_images = len(self.predictions.keys())
            current_image_index = 0
            print()
            print("Summarizing:")

            for image_id in self.predictions.keys():
                # print('Evaluating{0}...'.format(image_id))

                gt_bbox, img_name, img = self.refexp_gt.pull_item_box(image_id,
                                                                      self.draw)  # img.shape: (H, W, 3)
                H, W, _ = img.shape

                prediction = self.predictions[image_id]
                assert prediction is not None
                device = prediction['scores'].device

                sorted_scores_boxes = sorted(
                    zip(prediction["scores"].tolist(),
                        prediction["boxes"].tolist(),
                        prediction["align_cost"].tolist()), reverse=True
                )
                sorted_scores, sorted_boxes, sorted_align_cost = zip(*sorted_scores_boxes)
                sorted_align_cost = torch.tensor(sorted_align_cost)

                sorted_boxes = torch.cat(
                    [torch.as_tensor(x).view(1, 4) for x in sorted_boxes]).to(device)

                # Handle the situation when the assertion
                # (boxes1[:, 2:] >= boxes1[:, :2]).all() fails
                boxes1 = sorted_boxes
                boxes2 = torch.as_tensor(gt_bbox).view(-1, 4).to(device)
                if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
                    for i in range(boxes1.shape[0]):
                        current_row = boxes1[i]
                        # swap max and min if current row's max < min
                        if not current_row[2:] >= current_row[:2]:
                            xmin, ymin, xmax, ymax = current_row
                            if xmax < xmin:
                                temp = xmin
                                xmin = xmax
                                xmax = temp
                            if ymax < ymin:
                                temp = ymin
                                ymin = ymax
                                ymax = temp
                            boxes1[i][0] = xmin
                            boxes1[i][1] = ymin
                            boxes1[i][2] = xmax
                            boxes1[i][3] = ymax

                giou = generalized_box_iou(boxes1, boxes2)

                if ARGS_POSE:
                    tmp_idx = prediction["arms_scores"].argmax()
                    arm_token_pair[img_name] = tmp_idx
                    torch.save(arm_token_pair, 'arm_token_pair.pth')

                if 'arms' in prediction:
                    sorted_scores_arms = sorted(
                        zip(prediction["arms_scores"].tolist(),
                            prediction["arms"].tolist()), reverse=True
                    )
                    sorted_arm_scores, sorted_arms = zip(*sorted_scores_arms)
                    sorted_arms = torch.cat(
                        [torch.as_tensor(x).view(1, 4) for x in sorted_arms]).to(device)

                    if EVAL_EARLY_STOP:
                        fingertip_xyxy = sorted_arms[0, 2:4]
                        eye_xyxy = sorted_arms[0, 0:2]
                        box_centers_xyxy = sorted_boxes[:, :2]
                        arm_tensor = fingertip_xyxy - eye_xyxy
                        box_tensor = box_centers_xyxy - eye_xyxy
                        cos_sim = F.cosine_similarity(arm_tensor, box_tensor,
                                                      dim=1)

                        normalized_sorted_arms_xyxy = sorted_arms / torch.tensor(
                            [W, H, W, H], device=device)
                        normalized_sorted_boxes_xyxy = sorted_boxes / torch.tensor(
                            [W, H, W, H], device=device)

                        if PRINT_PREDICTIONS_AT_BREAKPOINT:
                            print("'" + img_name + "'", ',',
                                  normalized_sorted_boxes_xyxy, ',',
                                  normalized_sorted_arms_xyxy[0], ',', giou,
                                  ',', sorted_align_cost)
                            breakpoint()

                if SAVE_EVALUATION_PREDICTIONS:
                    # initialize CLIP model
                    if SAVE_CLIP_SCORES:
                        clip_model, preprocess = clip.load("ViT-B/32",
                                                           device=device)
                        sentence = self.refexp_gt.pull_item_sentence(image_id)

                    normalized_sorted_boxes_xyxy = sorted_boxes / torch.tensor(
                        [W, H, W, H], device=device)

                    if 'arms' in prediction:
                        normalized_sorted_arms_xyxy = sorted_arms / torch.tensor(
                            [W, H, W, H], device=device)
                        top_arm = normalized_sorted_arms_xyxy[0]
                    else:
                        top_arm = torch.tensor([-1, -1, -1, -1], device=normalized_sorted_boxes_xyxy.device)

                    for i in range(len(normalized_sorted_boxes_xyxy)):
                        # Find out one prediction for current image
                        current_image_name = img_name
                        current_box = normalized_sorted_boxes_xyxy[i]
                        current_arm = top_arm
                        current_box_score = sorted_scores[i]
                        current_giou = giou[i]
                        current_align_cost = sorted_align_cost[i]
                        box_xmin, box_ymin, box_xmax, box_ymax = current_box
                        arm_xmin, arm_ymin, arm_xmax, arm_ymax = current_arm

                        # Find out CLIP scores for current prediction
                        if SAVE_CLIP_SCORES:
                            abs_xmin, abs_ymin, abs_xmax, abs_ymax = \
                            sorted_boxes[i]
                            abs_xmin, abs_ymin, abs_xmax, abs_ymax = int(
                                abs_xmin), int(abs_ymin), int(abs_xmax), int(
                                abs_ymax)
                            # make sure coordinate does not exceed image width or height
                            abs_xmin = max(0, abs_xmin)
                            abs_ymin = max(0, abs_ymin)
                            abs_xmax = min(W, abs_xmax - 1)
                            abs_ymax = min(H, abs_ymax - 1)
                            # using box coordinates to get current patch of image
                            current_patch = img[abs_ymin:abs_ymax + 1,
                                            abs_xmin:abs_xmax + 1, :]
                            current_patch = torch.tensor(current_patch,
                                                         device=device).permute(
                                2, 0, 1)
                            # convert tensor to PIL
                            current_patch = transform(current_patch)
                            # CLIP preprocess patch
                            current_patch = preprocess(current_patch).unsqueeze(
                                0).to(device)
                            # TODO: process the sentence for better clip scores
                            processed_sentence = sentence
                            # CLIP tokenize text. Two classes: is something AND nothing
                            text = clip.tokenize(
                                [processed_sentence, 'nothing']).to(device)
                            # compute CLIP score for current patch
                            with torch.no_grad():
                                image_features = clip_model.encode_image(
                                    current_patch)
                                text_features = clip_model.encode_text(text)
                                logits_per_image, logits_per_text = clip_model(
                                    current_patch, text)
                                probs = logits_per_image.softmax(
                                    dim=-1).cpu().numpy()
                            # Append clip score and sentence to list
                            clip_score_list.append(probs[0, 0])
                            sentence_list.append(sentence)
                        else:
                            clip_score_list.append(-1)
                            sentence_list.append('n/a')

                        # Append one prediction to lists
                        image_name_list.append(current_image_name)

                        box_xmin_list.append(box_xmin.item())
                        box_ymin_list.append(box_ymin.item())
                        box_xmax_list.append(box_xmax.item())
                        box_ymax_list.append(box_ymax.item())

                        arm_xmin_list.append(arm_xmin.item())
                        arm_ymin_list.append(arm_ymin.item())
                        arm_xmax_list.append(arm_xmax.item())
                        arm_ymax_list.append(arm_ymax.item())

                        box_score_list.append(current_box_score)
                        giou_list.append(current_giou.item())
                        align_cost_list.append(current_align_cost.item())

                for thresh_iou in self.thresh_iou:
                    if max(giou[0]) >= thresh_iou:
                        dataset2score["yourefit"][thresh_iou] += 1.0

                dataset2count['yourefit'] += 1.0

                status_string = '    ' + '[' + str(current_image_index + 1) + '/' + str(total_num_images) + ']'
                progressBar(current_image_index, total_num_images, status_string)
                current_image_index += 1


            # Create a dataframe to store all predictions
            if SAVE_EVALUATION_PREDICTIONS:
                df = pd.DataFrame({'image_name': image_name_list,
                                   'box_xmin': box_xmin_list,
                                   'box_ymin': box_ymin_list,
                                   'box_xmax': box_xmax_list,
                                   'box_ymax': box_ymax_list,
                                   'arm_xmin': arm_xmin_list,
                                   'arm_ymin': arm_ymin_list,
                                   'arm_xmax': arm_xmax_list,
                                   'arm_ymax': arm_ymax_list,
                                   'box_score': box_score_list,
                                   'giou': giou_list,
                                   'align_cost': align_cost_list,
                                   'clip_score': clip_score_list,
                                   'sentence': sentence_list})
                df = df.astype({'image_name': 'str',
                                'box_xmin': 'float',
                                'box_ymin': 'float',
                                'box_xmax': 'float',
                                'box_ymax': 'float',
                                'arm_xmin': 'float',
                                'arm_ymin': 'float',
                                'arm_xmax': 'float',
                                'arm_ymax': 'float',
                                'box_score': 'float',
                                'giou': 'float',
                                'align_cost': 'float',
                                'clip_score': 'float',
                                'sentence': 'str'})

                # Save df to disk
                if not os.path.exists(prediction_dir):
                    os.mkdir(prediction_dir)
                df.to_csv(prediction_dir + '/' + prediction_file_name,
                          index=False)

            for key, value in dataset2score.items():
                for thresh_iou in self.thresh_iou:
                    try:
                        value[thresh_iou] /= dataset2count[key]
                    except:
                        pass
            results = {}
            for key, value in dataset2score.items():
                results[key] = sorted([v for k, v in value.items()],
                                      reverse=True)
                print(
                    f" Dataset: {key} - Precision @ 0.25, 0.5, 0.75: {results[key]} \n")

            return results
        return None
