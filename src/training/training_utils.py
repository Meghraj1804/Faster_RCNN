import sys
import torch
import torch.nn as nn
import torchvision
import math
import numpy as np

from src.exception.exception import CustomException
from src.logging.logger import logging



def apply_regression_pred_to_anchors_or_proposals(box_transform_pred, anchors_or_proposals):

    try:
        box_transform_pred = box_transform_pred.reshape(
            box_transform_pred.size(0), -1, 4)
        
        # Get cx, cy, w, h from x1,y1,x2,y2
        w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
        h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
        center_x = anchors_or_proposals[:, 0] + 0.5 * w
        center_y = anchors_or_proposals[:, 1] + 0.5 * h
        
        dx = box_transform_pred[..., 0]
        dy = box_transform_pred[..., 1]
        dw = box_transform_pred[..., 2]
        dh = box_transform_pred[..., 3]
        
        # dh -> (num_anchors_or_proposals, num_classes)
        
        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=math.log(1000.0 / 16))
        dh = torch.clamp(dh, max=math.log(1000.0 / 16))
        
        pred_center_x = dx * w[:, None] + center_x[:, None]
        pred_center_y = dy * h[:, None] + center_y[:, None]
        pred_w = torch.exp(dw) * w[:, None]
        pred_h = torch.exp(dh) * h[:, None]
        # pred_center_x -> (num_anchors_or_proposals, num_classes)
        
        pred_box_x1 = pred_center_x - 0.5 * pred_w
        pred_box_y1 = pred_center_y - 0.5 * pred_h
        pred_box_x2 = pred_center_x + 0.5 * pred_w
        pred_box_y2 = pred_center_y + 0.5 * pred_h
        
        pred_boxes = torch.stack((
            pred_box_x1,
            pred_box_y1,
            pred_box_x2,
            pred_box_y2),
            dim=2)
        # pred_boxes -> (num_anchors_or_proposals, num_classes, 4)
        return pred_boxes
    except Exception as e:
            raise CustomException(e,sys)

def clamp_boxes_to_image_boundary(boxes, image_shape):
    
    try:
        boxes_x1 = boxes[..., 0]
        boxes_y1 = boxes[..., 1]
        boxes_x2 = boxes[..., 2]
        boxes_y2 = boxes[..., 3]
        height, width = image_shape[-2:]
        boxes_x1 = boxes_x1.clamp(min=0, max=width)
        boxes_x2 = boxes_x2.clamp(min=0, max=width)
        boxes_y1 = boxes_y1.clamp(min=0, max=height)
        boxes_y2 = boxes_y2.clamp(min=0, max=height)
        boxes = torch.cat((
            boxes_x1[..., None],
            boxes_y1[..., None],
            boxes_x2[..., None],
            boxes_y2[..., None]),
            dim=-1)
        return boxes
    except Exception as e:
            raise CustomException(e,sys)

def get_iou(boxes1, boxes2):
    try:    
        # Area of boxes (x2-x1)*(y2-y1)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (N,)
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (M,)
        
        # Get top left x1,y1 coordinate
        x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # (N, M)
        y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # (N, M)
        
        # Get bottom right x2,y2 coordinate
        x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # (N, M)
        y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # (N, M)
        
        intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)  # (N, M)
        union = area1[:, None] + area2 - intersection_area  # (N, M)
        iou = intersection_area / union  # (N, M)
        return iou
    except Exception as e:
            raise CustomException(e,sys)

def boxes_to_transformation_targets(ground_truth_boxes, anchors_or_proposals):
    
    try:
        # Get center_x,center_y,w,h from x1,y1,x2,y2 for anchors
        widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
        heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
        center_x = anchors_or_proposals[:, 0] + 0.5 * widths
        center_y = anchors_or_proposals[:, 1] + 0.5 * heights
        
        # Get center_x,center_y,w,h from x1,y1,x2,y2 for gt boxes
        gt_widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
        gt_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
        gt_center_x = ground_truth_boxes[:, 0] + 0.5 * gt_widths
        gt_center_y = ground_truth_boxes[:, 1] + 0.5 * gt_heights
        
        targets_dx = (gt_center_x - center_x) / widths
        targets_dy = (gt_center_y - center_y) / heights
        targets_dw = torch.log(gt_widths / widths)
        targets_dh = torch.log(gt_heights / heights)
        regression_targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return regression_targets
    except Exception as e:
            raise CustomException(e,sys)

def sample_positive_negative(labels, positive_count, total_count):
    try:
        positive = torch.where(labels >= 1)[0]
        negative = torch.where(labels == 0)[0]

        # Convert inputs to Python ints (important!)
        positive_count = int(positive_count)
        total_count = int(total_count)

        # Compute required sample counts
        num_pos = min(positive_count, int(positive.numel()))
        num_neg = min(total_count - num_pos, int(negative.numel()))

        # Final safety conversion
        num_pos = int(num_pos)
        num_neg = int(num_neg)

        # ---- POSITIVE SAMPLING ----
        pos_total = int(positive.numel())
        if pos_total == 0:
            perm_positive_idxs = torch.empty(0, dtype=torch.long, device=labels.device)
        else:
            perm_positive_idxs = torch.randperm(pos_total, device=labels.device)[:num_pos]

        # ---- NEGATIVE SAMPLING ----
        neg_total = int(negative.numel())
        if neg_total == 0:
            perm_negative_idxs = torch.empty(0, dtype=torch.long, device=labels.device)
        else:
            perm_negative_idxs = torch.randperm(neg_total, device=labels.device)[:num_neg]

        pos_idxs = positive[perm_positive_idxs]
        neg_idxs = negative[perm_negative_idxs]

        sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
        sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)

        sampled_pos_idx_mask[pos_idxs] = True
        sampled_neg_idx_mask[neg_idxs] = True

        return sampled_neg_idx_mask, sampled_pos_idx_mask
    except Exception as e:
            raise CustomException(e,sys)

def transform_boxes_to_original_size(boxes, new_size, original_size):

    try:
        ratios = [
            torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
            / torch.tensor(s, dtype=torch.float32, device=boxes.device)
            for s, s_orig in zip(new_size, original_size)
        ]
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim=1)
    except Exception as e:
            raise CustomException(e,sys)
        
def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area'):
    
    try:
        gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
        gt_labels = sorted(gt_labels)
        all_aps = {}
        # average precisions for ALL classes
        aps = []
        for idx, label in enumerate(gt_labels):
            # Get detection predictions of this class
            cls_dets = [
                [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
                if label in im_dets for im_dets_label in im_dets[label]
            ]
            
            # Sort them by confidence score
            cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])
            
            # For tracking which gt boxes of this class have already been matched
            gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
            # Number of gt boxes for this class for recall calculation
            num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
            tp = [0] * len(cls_dets)
            fp = [0] * len(cls_dets)
            
            # For each prediction
            for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
                # Get gt boxes for this image and this label
                im_gts = gt_boxes[im_idx][label]
                max_iou_found = -1
                max_iou_gt_idx = -1
                
                # Get best matching gt box
                for gt_box_idx, gt_box in enumerate(im_gts):    
                    
                    # gt_box =  [np.float32(12.0), np.float32(310.0), np.float32(83.0), np.float32(361.0)]
                    gt_tensor = torch.tensor(gt_box, dtype=torch.float32).unsqueeze(0)
                    
                    detection_pred = det_pred[:-1]
                    detection_pred = torch.tensor(detection_pred, dtype=torch.float32).unsqueeze(0)
                    
                    gt_box_iou = get_iou(detection_pred, gt_tensor)
                    gt_box_iou = float(gt_box_iou)
                    # gt_box_iou = get_iou(det_pred[:-1], gt_box)
                    if gt_box_iou > max_iou_found:
                        max_iou_found = gt_box_iou
                        max_iou_gt_idx = gt_box_idx
                # TP only if iou >= threshold and this gt has not yet been matched
                if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                    fp[det_idx] = 1
                    # gt_matched[im_idx][max_iou_gt_idx] = True
                else:
                    tp[det_idx] = 1
                    # If tp then we set this gt box as matched
                    gt_matched[im_idx][max_iou_gt_idx] = True
            # Cumulative tp and fp
            tp = np.cumsum(tp)
            fp = np.cumsum(fp)
            
            eps = np.finfo(np.float32).eps
            recalls = tp / np.maximum(num_gts, eps)
            precisions = tp / np.maximum((tp + fp), eps)

            if method == 'area':
                recalls = np.concatenate(([0.0], recalls, [1.0]))
                precisions = np.concatenate(([0.0], precisions, [0.0]))
                
                # Replace precision values with recall r with maximum precision value
                # of any recall value >= r
                # This computes the precision envelope
                for i in range(precisions.size - 1, 0, -1):
                    precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
                # For computing area, get points where recall changes value
                i = np.where(recalls[1:] != recalls[:-1])[0]
                # Add the rectangular areas to get ap
                ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
            elif method == 'interp':
                ap = 0.0
                for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                    # Get precision values for recall values >= interp_pt
                    prec_interp_pt = precisions[recalls >= interp_pt]
                    
                    # Get max of those precision values
                    prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                    ap += prec_interp_pt
                ap = ap / 11.0
            else:
                raise ValueError('Method can only be area or interp')
            if num_gts > 0:
                aps.append(ap)
                all_aps[label] = ap
            else:
                all_aps[label] = np.nan
        # compute mAP at provided iou threshold
        mean_ap = sum(aps) / len(aps)
        return mean_ap, all_aps
    except Exception as e:
        raise CustomException(e,sys)












































