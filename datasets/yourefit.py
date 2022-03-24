# -*- coding: utf-8 -*-

"""
YouRefIt referring image PyTorch dataset.
"""
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
# import h5py
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from collections import OrderedDict
sys.path.append('.')
import operator
import argparse
import collections
import logging
import json
import pickle5 as pickle
import re
import util.dist as dist
from IPython import embed
#sys.modules['utils'] = utils
from transformers import RobertaTokenizerFast
sys.path.append('/DATA2/cxx/mdetr/datasets')
from .yourefit_token import match_pos
import copy
from util.box_ops import generalized_box_iou

cv2.setNumThreads(0)


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
            positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


class DatasetNotFoundError(Exception):
    pass

class ReferDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'yourefit': {'splits': ('train', 'val', 'test')}
    }
    
    def __init__(self, data_root, split_root='data', dataset='yourefit', 
                 transform=None, augment=False, device=None, return_idx=False, testmode=False,
                 split='train', max_query_len=128, lstm=False, bert_model='bert-base-uncased',args=None):
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
        self.augment=augment
        self.return_idx=return_idx
        self.tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)
        if self.dataset == 'yourefit':
            self.dataset_root = osp.join(self.data_root, 'yourefit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.arm_dir = osp.join(self.dataset_root, 'arms.json')
            with open(self.arm_dir,"r") as f:
                self.arm_data = json.load(f)
        else:   ## refcoco, etc.
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
            imgset_file = osp.join(self.dataset_root,'{0}_id.txt'.format(split))
            
            with open(imgset_file,'r') as f:
                while True:
                    img_name = f.readline()
                    if img_name:
                        self.images.append(img_name[:-1])
                    else:
                        break
            f.close()
        
        for split_idx in range(len(self.images)):
            self.image_files[self.images[split_idx]] = split_idx

    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))


    def pull_item_box(self, idx,return_img= False):
        
        #img_file, _, bbox, phrase, attri = self.images[idx]
        ## box format: to x1y1x2y2
        img_name = self.images[idx]
        pickle_file = osp.join(osp.join(self.dataset_root,'pickle'),img_name+'.p')
        pick = pickle.load(open(pickle_file, "rb" ))
        bbox = pick['bbox']
        bbox = np.array(bbox, dtype=int) #x1y1x2y2    
        img = None
        if return_img:
            img_path = osp.join(self.im_dir, img_name+'.jpg')
            img = cv2.imread(img_path)
        return bbox, img_name,img


    def pull_item(self, idx):
        
        img_name = self.images[idx]
        ## box format: to x1y1x2y2
        pickle_file = osp.join(osp.join(self.dataset_root,'pickle'),img_name+'.p')
        pick = pickle.load(open(pickle_file, "rb" ))
        bbox = pick['bbox']
        target_word = pick['anno_target']
        phrase = pick['anno_sentence']
        token_pos = match_pos(phrase,target_word)
        token_pos = [token_pos]
        bbox = np.array(bbox, dtype=int) #x1y1x2y2
        img_path = osp.join(self.im_dir, img_name+'.jpg')
        #img = cv2.imread(img_path)
        img = Image.open(img_path).convert('RGB')
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
        pt = cv2.resize(pt, (256,256))
        pt = np.reshape(pt, (3, 256, 256))
        arm = self.arm_data[img_name]
        return img, pt, ht, phrase, bbox, token_pos, arm, img_name

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, pt, ht, phrase, bbox,token_pos, arm,img_name = self.pull_item(idx)
        # phrase = phrase.decode("utf-8").encode().lower()
        phrase = phrase.lower()
        ## seems a bug in torch transformation resize, so separate in advance
        w,h = img.size
        target = {}
        target['orig_size'] = torch.tensor([h,w])
        target['size'] = torch.tensor([h,w]) 
        target['boxes'] = torch.tensor(bbox, dtype=torch.float32).unsqueeze(0)
        target['caption'] = phrase
        target['img_name'] = img_name
        target['image_id'] = torch.tensor([idx])
        target['tokens_positive'] = token_pos
        target['labels'] = torch.tensor([1])
        target['arm'] = torch.tensor(arm).flatten()
        #print(target['arm'].shape)
        assert target['arm'].shape[0] ==4

        assert len(target["boxes"]) == len(target["tokens_positive"])
        tokenized = self.tokenizer(phrase, return_tensors="pt")
        target["positive_map"] = create_positive_map(tokenized, target["tokens_positive"])
        target['ht_map'] = torch.tensor(ht).unsqueeze(0)
        if self.transform is not None:
            img,target = self.transform(img,target)
        target['dataset_name'] = 'yourefit'
        return img, target


class YouRefItEvaluator_old(object):
    def __init__(self, ref_dataset, iou_types, k=(1, 2, 10), thresh_iou=0.25,draw=False):
        assert isinstance(k, (list, tuple))
        ref_dataset = copy.deepcopy(ref_dataset)
        self.refexp_gt = ref_dataset
        self.iou_types = iou_types
        self.img_ids = self.refexp_gt.image_files.keys()
        self.predictions = {}
        self.k = k
        self.thresh_iou = thresh_iou
        self.draw=draw

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
                
                bbox,img_name,img = self.refexp_gt.pull_item_box(image_id,self.draw)
                
                prediction = self.predictions[image_id]
                assert prediction is not None
                sorted_scores_boxes = sorted(
                    zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True
                )
                sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
                sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
                converted_bbox = bbox

                giou = generalized_box_iou(sorted_boxes, torch.as_tensor(converted_bbox).view(-1, 4))
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
                print(f" Dataset: {key} - Precision @ 1, 5, 10: {results[key]} \n")

            return results
        return None



def draw_box(img,box,img_name,arm=None,gt_box=None,box1=None):
    pt1 = (int(box[0]),int(box[1]))
    pt2 = (int(box[2]),int(box[3]))
    cv2.rectangle(img,pt1,pt2,(0,255,0),4) # green
    if arm is not None:
        pt3 = (int(arm[0]),int(arm[1]))
        pt4 = (int(arm[2]),int(arm[3]))
        cv2.line(img,pt3,pt4,(0,0,255),5) #red
    if gt_box is not None:
        pt5 = (int(gt_box[0]),int(gt_box[1]))
        pt6 = (int(gt_box[2]),int(gt_box[3]))
        cv2.rectangle(img,pt5,pt6,(255,0,0),4) #blue
    if box1 is not None: # second best box
        pt7 = (int(box1[0]),int(box1[1]))
        pt8 = (int(box1[2]),int(box1[3]))
        cv2.rectangle(img,pt7,pt8,(255,255,0),4) #light blue
    cv2.imwrite(img_name,img)

import torch.nn.functional as F
def aligned_score(arm,box):
    arm_tensor = torch.Tensor(arm[2:])-torch.Tensor(arm[:2])
    box_center = [(box[0]+box[2])/2,(box[1]+box[3])/2]
    box_tensor = torch.Tensor(box_center)-torch.Tensor(arm[:2])
    cos_sim = F.cosine_similarity(arm_tensor,box_tensor,dim=0)
    return cos_sim


def aligned_scores(arm,boxes):
    arm_tensor = torch.Tensor(arm[2:])-torch.Tensor(arm[:2])
    box_center = [(boxes[:,0]+boxes[:,2])/2,(boxes[:,1]+boxes[:,3])/2]
    box_tensor = torch.vstack(box_center).transpose(0,1)-torch.Tensor(arm[:2])
    cos_sim = F.cosine_similarity(arm_tensor,box_tensor,dim=1)
    return cos_sim

class YouRefItEvaluator(object):
    def __init__(self, ref_dataset, iou_types, k=1,thresh_iou=(0.25,0.5,0.75),draw=True):
        ref_dataset = copy.deepcopy(ref_dataset)
        self.refexp_gt = ref_dataset
        self.iou_types = iou_types
        self.img_ids = self.refexp_gt.image_files.keys()
        self.predictions = {}
        self.k = k
        self.thresh_iou = thresh_iou
        self.draw=draw

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
            
            for image_id in self.predictions.keys():
                #print('Evaluating{0}...'.format(image_id))
                
                gt_bbox,img_name,img = self.refexp_gt.pull_item_box(image_id,self.draw)

                prediction = self.predictions[image_id]
                assert prediction is not None
                sorted_scores_boxes = sorted(
                    zip(prediction["scores"].tolist(), prediction["boxes"].tolist()), reverse=True
                )
                sorted_scores, sorted_boxes = zip(*sorted_scores_boxes)
                sorted_boxes = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_boxes])
                giou = generalized_box_iou(sorted_boxes, torch.as_tensor(gt_bbox).view(-1, 4))

                tmp_idx = prediction["arms_scores"].argmax()
                arm_token_pair[img_name] = tmp_idx
                torch.save(arm_token_pair,'arm_token_pair.pth')

                if 'arms' in prediction:
                    sorted_scores_arms = sorted(
                        zip(prediction["arms_scores"].tolist(), prediction["arms"].tolist()), reverse=True
                    )
                    sorted_arm_scores, sorted_arms = zip(*sorted_scores_arms)
                    sorted_arms = torch.cat([torch.as_tensor(x).view(1, 4) for x in sorted_arms])
                    #if self.draw:
                        #if giou[0]<0.5:
                            #draw_box(img,sorted_boxes[0],'good_pred/'+img_name+'.jpg',sorted_arms[0])#,gt_bbox,sorted_boxes[1])
                            #draw_box(img,sorted_boxes[1],'check_pred/'+img_name+'_1.jpg')
                            #draw_box(img,sorted_boxes[0],'gt_vis/'+img_name+'.jpg',sorted_arms[0],gt_bbox) #,sorted_boxes[1])
                            #draw_box(img,gt_bbox,'gt_vis/'+img_name+'.jpg',sorted_arms[0])
                # aligned_score = aligned_scores(sorted_arms[0],sorted_boxes[:,:10])

                # print(aligned_score[0])
                # aligned_idx = 0
                # for i in range(5):
                #     if aligned_score[i]>0.9:
                #         aligned_idx = i
                #         break
                # if aligned_score[0]>0.9:
                #     aligned_idx = 0    
                # else:
                #     aligned_idx = aligned_score.argmax()
                        
                for thresh_iou in self.thresh_iou:
                    if max(giou[0]) >= thresh_iou:
                        dataset2score["yourefit"][thresh_iou] += 1.0
                        
                dataset2count['yourefit'] += 1.0

            for key, value in dataset2score.items():
                for thresh_iou in self.thresh_iou:
                    try:
                        value[thresh_iou] /= dataset2count[key]
                    except:
                        pass
            results = {}
            for key, value in dataset2score.items():
                results[key] = sorted([v for k, v in value.items()],reverse=True)
                print(f" Dataset: {key} - Precision @ 0.25, 0.5, 0.75: {results[key]} \n")

            return results
        return None
