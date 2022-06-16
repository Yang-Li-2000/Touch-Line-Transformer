# Copyright (c): Anonymous Authors. Licensed under the Apache License 2.0. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from MDETR (https://github.com/ashkamath/mdetr)
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Dict, Iterable, Optional

import torch
import torch.nn
import torch.optim
from datasets.yourefit import YouRefItEvaluator

import util.dist as dist
from datasets.clevrref import ClevrRefEvaluator
from datasets.coco_eval import CocoEvaluator
from datasets.flickr_eval import FlickrEvaluator
# from datasets.phrasecut_eval import PhrasecutEvaluator
from datasets.refexp import RefExpEvaluator
from util.metrics import MetricLogger, SmoothedValue
from util.misc import targets_to
from util.optim import adjust_learning_rate, update_ema
import time
from models.mdetr import get_pose_loss

from magic_numbers import *
import pandas as pd
import temp_vars

def train_one_epoch(
        model: torch.nn.Module,
        criterion: Optional[torch.nn.Module],
        contrastive_criterion: Optional[torch.nn.Module],
        qa_criterion: Optional[torch.nn.Module],
        weight_dict: Dict[str, float],
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        args,
        max_norm: float = 0,
        model_ema: Optional[torch.nn.Module] = None,
):
    model.train()
    if criterion is not None:
        criterion.train()
    if contrastive_criterion is not None:
        contrastive_criterion.train()
    if qa_criterion is not None:
        qa_criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr",
                            SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_backbone",
                            SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("lr_text_encoder",
                            SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    # Create lists to store the outputs of mdetr
    list_for_pred_scores = []
    list_for_pred_classes = []
    list_for_img_names = []
    list_for_xmin = []
    list_for_ymin = []
    list_for_xmax = []
    list_for_ymax = []

    current_batch_index = 0

    num_training_steps = int(len(data_loader) * args.epochs)
    for i, batch_dict in enumerate(
            metric_logger.log_every(data_loader, print_freq, header,
                                    args.output_dir)):
        curr_step = epoch * len(data_loader) + i
        samples = batch_dict["samples"].to(device)
        positive_map = batch_dict["positive_map"].to(
            device) if "positive_map" in batch_dict else None
        targets = batch_dict["targets"]
        answers = {k: v.to(device) for k, v in batch_dict[
            "answers"].items()} if "answers" in batch_dict else None
        captions = [t["caption"] for t in targets]

        targets = targets_to(targets, device)

        pafs = None
        yourefit = True
        target_arms = None
        if yourefit:
            paf_list = []
            target_arms = []
            for target in targets:
                paf_list.append(target['ht_map'])
                if target['arm'].shape[0] == 4:
                    target['arm'] = target['arm'].unsqueeze(0)
                target_arms.append(target['arm'])
            from util.misc import NestedTensor
            pafs = NestedTensor.from_tensor_list(paf_list)

        loss_dict = {}
        memory_cache = None
        target_arm = None
        pred_arm = None
        pose_out = None
        if args.masks:
            outputs = model(samples, captions)
        else:
            # First pass through the model
            memory_cache, pose_out = model(samples,
                                           captions=captions,
                                           encode_and_save=True,
                                           paf_samples=pafs)
            # ***'s implementation of arm loss computation
            if pose_out is not None and PREDICT_POSE_USING_A_DIFFERENT_MODEL:
                for k in range(3):
                    pose_loss, target_arm, pred_arm = \
                        get_pose_loss(pose_out['{0}_arms'.format(k)],
                                      pose_out['{0}_arm_score'.format(k)],
                                      target_arms,
                                      k)    # Shape:[4, 10, 4], [4, 10, 2], 4
                    loss_dict.update(pose_loss)

            # Second pass through the model.
            #   Pass in the encodings of memory_cache['tokenized'].
            #   Otherwise, memory_cache['tokenized']._encodings passed in can
            #   sometimes get lost inside the function below.
            #   Specifically, memory_cache['tokenized']._encodings inside the
            #   function below sometimes become empty if not pass in
            #   the encodings of memory_cache['tokenized'] and manually set it
            #   in the function below.
            outputs = model(samples, captions, encode_and_save=False,
                            memory_cache=memory_cache,
                            arm_query=target_arm,
                            encodings_of_tokenized=memory_cache['tokenized']._encodings)

            # ***'s implementation of adding pred_arm ot outputs
            if pose_out is not None and PREDICT_POSE_USING_A_DIFFERENT_MODEL:
                outputs.update({'pred_arm': pred_arm})  # last layer
                outputs.update(pose_out)

            # Get the predicted arm and compute arm loss using the same transformer
            # TODO: this should be computed before the arm-box-align loss,
            #  so that the predicted arms with shortest l1 distanced can be
            #  used for computing the arm-box-align loss
            if args.pose and not PREDICT_POSE_USING_A_DIFFERENT_MODEL:
                # This '2' was hard-coded by ***
                i = 2
                arm = outputs['{0}_arms'.format(i)]
                arm_class = outputs['{0}_arm_score'.format(i)]
                pose_loss, processed_target_arm, pred_arm = \
                    get_pose_loss(arm, arm_class, target_arms, i)
                loss_dict.update(pose_loss)
                outputs.update({'pred_arm': pred_arm})

            # Save predictions to lists and write them into a file
            if SAVE_MDETR_PREDICTIONS:
                for j in range(len(targets)):
                    current_pred_logits = outputs['pred_logits'][
                        j].detach().cpu()
                    current_pred_boxes = outputs['pred_boxes'][j].detach().cpu()
                    current_img_name = targets[j]['img_name']

                    values, indices = current_pred_logits.softmax(-1).max(-1)

                    num_boxes = current_pred_logits.shape[0]

                    # Append each prediction for current image separately into lists
                    for k in range(num_boxes):
                        score = values[k]
                        pred_class = indices[k]

                        # cxcywh to xmin, ymin, xmax, ymax
                        box = current_pred_boxes[k]
                        cx, cy, w, h = box
                        xmin, ymin, xmax, ymax = [cx - w / 2, cy - h / 2,
                                                  cx + w / 2, cy + h / 2]

                        list_for_pred_scores.append(score.item())
                        list_for_pred_classes.append(pred_class.item())
                        list_for_img_names.append(current_img_name)
                        list_for_xmin.append(xmin.item())
                        list_for_ymin.append(ymin.item())
                        list_for_xmax.append(xmax.item())
                        list_for_ymax.append(ymax.item())

                # ???
                del outputs
                metric_logger.update(loss=-1)
                metric_logger.update(lr=-1)
                metric_logger.update(lr_backbone=-1)
                metric_logger.update(lr_text_encoder=-1)

                # process lists and store contents into a dataframe
                if i == len(data_loader) - 1:
                    print("Storing MDETR Predictions to Lists")
                    df = pd.DataFrame({'img_name': list_for_img_names,
                                       'pred_score': list_for_pred_scores,
                                       'pred_class': list_for_pred_classes,
                                       'xmin': list_for_xmin,
                                       'ymin': list_for_ymin,
                                       'xmax': list_for_xmax,
                                       'ymax': list_for_ymax
                                       })
                    df.astype({'img_name': 'str',
                               'pred_score': 'float',
                               'pred_class': 'int',
                               'xmin': 'float',
                               'ymin': 'float',
                               'xmax': 'float',
                               'ymax': 'float'})

                    # save dataframe to disk as a csv file
                    print("Writing MDETR Predictions to Disk")
                    file_name = 'mdetr_predictions_training_set.csv'
                    print(file_name)
                    df.to_csv(file_name, index=False)
                    raise NotImplementedError('Done')
                continue

            if pose_out is not None:
                outputs.update({'pred_arm': pred_arm})  # last layer
                outputs.update(pose_out)

        if criterion is not None:
            loss_dict.update(
                criterion(outputs, targets, positive_map, args=args))

        if contrastive_criterion is not None:
            assert memory_cache is not None
            contrastive_loss = contrastive_criterion(
                memory_cache["text_pooled_op"], memory_cache["img_pooled_op"])
            loss_dict["contrastive_loss"] = contrastive_loss

        if qa_criterion is not None:
            answer_losses = qa_criterion(outputs, answers)
            loss_dict.update(answer_losses)

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if
                     k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in
                                      loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in
                                    loss_dict_reduced.items() if
                                    k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        if not CALCULATE_COS_SIM:
            optimizer.step()

        adjust_learning_rate(
            optimizer,
            epoch,
            curr_step,
            num_training_steps=num_training_steps,
            args=args,
        )
        if model_ema is not None:
            update_ema(model, model_ema, args.ema_decay)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(lr_backbone=optimizer.param_groups[1]["lr"])
        metric_logger.update(lr_text_encoder=optimizer.param_groups[2]["lr"])

        current_batch_index += 1
        if TRAIN_EARLY_STOP and current_batch_index >= TRAIN_EARLY_STOP_COUNT:
            break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
        model: torch.nn.Module,
        criterion: Optional[torch.nn.Module],
        contrastive_criterion: Optional[torch.nn.Module],
        qa_criterion: Optional[torch.nn.Module],
        postprocessors: Dict[str, torch.nn.Module],
        weight_dict: Dict[str, float],
        data_loader,
        evaluator_list,
        device: torch.device,
        args,
):
    model.eval()
    if criterion is not None:
        criterion.eval()
    if contrastive_criterion is not None:
        contrastive_criterion.eval()
    if qa_criterion is not None:
        qa_criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    eval_count = 0

    for batch_dict in metric_logger.log_every(data_loader, 10, header,
                                              args.output_dir):
        samples = batch_dict["samples"].to(device)
        positive_map = batch_dict["positive_map"].to(
            device) if "positive_map" in batch_dict else None
        targets = batch_dict["targets"]
        answers = {k: v.to(device) for k, v in batch_dict[
            "answers"].items()} if "answers" in batch_dict else None
        captions = [t["caption"] for t in targets]
        img_names = [t["img_name"] for t in targets]

        targets = targets_to(targets, device)

        pafs = None
        yourefit = True
        target_arms = None
        if yourefit:
            paf_list = []
            target_arms = []
            for target in targets:
                paf_list.append(target['ht_map'])
                if target['arm'].shape[0] == 4:
                    target['arm'] = target['arm'].unsqueeze(0)
                target_arms.append(target['arm'])
            from util.misc import NestedTensor
            pafs = NestedTensor.from_tensor_list(paf_list)

        loss_dict = {}
        memory_cache = None
        target_arm = None
        pred_arm = None
        pose_out = None

        # First pass through the model
        memory_cache, pose_out = model(samples, captions, encode_and_save=True,
                                       paf_samples=pafs, img_names=img_names)
        # ***'s implementation of arm loss computation
        if args.pose and PREDICT_POSE_USING_A_DIFFERENT_MODEL:
            if pose_out is not None:
                for i in range(3):
                    pose_loss, target_arm, pred_arm = \
                        get_pose_loss(pose_out['{0}_arms'.format(i)],
                                      pose_out['{0}_arm_score'.format(i)],
                                      target_arms, i)
                    loss_dict.update(pose_loss)

        # Second pass through the model
        outputs = model(samples, captions, encode_and_save=False,
                        memory_cache=memory_cache, arm_query=target_arm,
                        img_names=img_names)

        # ***'s implementation of adding pred_arm ot outputs
        if args.pose and PREDICT_POSE_USING_A_DIFFERENT_MODEL:
            if pose_out is not None:
                outputs.update({'pred_arm': pred_arm})  # last layer
                outputs.update(pose_out)

        # Get the predicted arm and compute arm loss using the same transformer
        if args.pose and not PREDICT_POSE_USING_A_DIFFERENT_MODEL:
            # This '2' was hard-coded by ***
            i = 2
            pose_loss, target_arm, pred_arm = \
                get_pose_loss(outputs['{0}_arms'.format(i)],
                              outputs['{0}_arm_score'.format(i)],
                              target_arms, i)
            loss_dict.update(pose_loss)
            outputs.update({'pred_arm': pred_arm})


        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, positive_map))

        if contrastive_criterion is not None:
            assert memory_cache is not None
            contrastive_loss = contrastive_criterion(
                memory_cache["text_pooled_op"], memory_cache["img_pooled_op"])
            loss_dict["contrastive_loss"] = contrastive_loss

        if qa_criterion is not None:
            answer_losses = qa_criterion(outputs, answers)
            loss_dict.update(answer_losses)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in
                                    loss_dict_reduced.items() if
                                    k in weight_dict}
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in
                                      loss_dict_reduced.items()}
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )

        if not args.no_detection:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets],
                                            dim=0)
            results = postprocessors["bbox"](outputs, orig_target_sizes,
                                             img_names=img_names, targets=targets)
            if "segm" in postprocessors.keys():
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors["segm"](results, outputs,
                                                 orig_target_sizes,
                                                 target_sizes)

            flickr_res = [] if "flickr_bbox" in postprocessors.keys() else None
            if "flickr_bbox" in postprocessors.keys():
                image_ids = [t["original_img_id"] for t in targets]
                sentence_ids = [t["sentence_id"] for t in targets]
                items_per_batch_element = [t["nb_eval"] for t in targets]
                positive_map_eval = batch_dict["positive_map_eval"].to(device)
                flickr_results = postprocessors["flickr_bbox"](
                    outputs, orig_target_sizes, positive_map_eval,
                    items_per_batch_element
                )
                assert len(flickr_results) == len(image_ids) == len(
                    sentence_ids)
                for im_id, sent_id, output in zip(image_ids, sentence_ids,
                                                  flickr_results):
                    flickr_res.append(
                        {"image_id": im_id, "sentence_id": sent_id,
                         "boxes": output})

            phrasecut_res = None
            if "phrasecut" in postprocessors.keys():
                phrasecut_res = postprocessors["phrasecut"](results)
                assert len(targets) == len(phrasecut_res)
                for i in range(len(targets)):
                    phrasecut_res[i]["original_id"] = targets[i]["original_id"]
                    phrasecut_res[i]["task_id"] = targets[i]["task_id"]

            res = {target["image_id"].item(): output for target, output in
                   zip(targets, results)}
            for evaluator in evaluator_list:
                if isinstance(evaluator, FlickrEvaluator):
                    evaluator.update(flickr_res)
                else:
                    evaluator.update(res)

        eval_count += 1
        if EVAL_EARLY_STOP and eval_count >= EVAL_EARLY_STOP_COUNT:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    for evaluator in evaluator_list:
        evaluator.synchronize_between_processes()

    if CALCULATE_COS_SIM:
        print()
        print('num_images:', temp_vars.image_count)
        print("gt:", temp_vars.gt_cos_sim / temp_vars.image_count)
        print('pred:', temp_vars.pred_cos_sim / temp_vars.image_count)
        print()
        breakpoint()

    refexp_res = None
    flickr_res = None
    phrasecut_res = None
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            evaluator.accumulate()
            evaluator.summarize()

        elif isinstance(evaluator, (RefExpEvaluator, ClevrRefEvaluator)):
            refexp_res = evaluator.summarize()
        elif isinstance(evaluator, FlickrEvaluator):
            flickr_res = evaluator.summarize()
        elif isinstance(evaluator, YouRefItEvaluator):
            yourefit_res = evaluator.summarize()
    # accumulate predictions from all images

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    for evaluator in evaluator_list:
        if isinstance(evaluator, CocoEvaluator):
            if "bbox" in postprocessors.keys():
                stats["coco_eval_bbox"] = evaluator.coco_eval[
                    "bbox"].stats.tolist()
            if "segm" in postprocessors.keys():
                stats["coco_eval_masks"] = evaluator.coco_eval[
                    "segm"].stats.tolist()

    if refexp_res is not None:
        stats.update(refexp_res)

    if flickr_res is not None:
        stats["flickr"] = flickr_res

    if phrasecut_res is not None:
        stats["phrasecut"] = phrasecut_res

    if yourefit_res is not None:
        stats.update(yourefit_res)

    return stats
