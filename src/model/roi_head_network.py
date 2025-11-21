import sys
import torch
import torch.nn as nn
import torchvision

from src.exception.exception import CustomException
from src.logging.logger import logging

from src.entity.config_entity import ModelParamConfig
from src.training import training_utils


class ROIHead(nn.Module):
    def __init__(self,model_config:ModelParamConfig):
        super(ROIHead,self).__init__()
        self.model_config = model_config
        
        self.fc6 = nn.Linear(self.model_config.roi_backbone_out_channels 
                             * self.model_config.roi_pool_size 
                             * self.model_config.roi_pool_size,
                                self.model_config.roi_fc_inner_dim)
        
        self.fc7 = nn.Linear(self.model_config.roi_fc_inner_dim,
                            self.model_config.roi_fc_inner_dim)
        
        self.cls_layer = nn.Linear(self.model_config.roi_fc_inner_dim,
                                    self.model_config.num_classes)
        
        self.bbox_reg_layer = nn.Linear(self.model_config.roi_fc_inner_dim,
                                        self.model_config.num_classes * 4)
        
        torch.nn.init.normal_(self.cls_layer.weight, std=0.01)
        torch.nn.init.constant_(self.cls_layer.bias, 0)

        torch.nn.init.normal_(self.bbox_reg_layer.weight, std=0.001)
        torch.nn.init.constant_(self.bbox_reg_layer.bias, 0)
        
    def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):
        
        try:
            # Get IOU Matrix between gt boxes and proposals
            iou_matrix = training_utils.get_iou(gt_boxes, proposals)
            # For each gt box proposal find best matching gt box
            best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
            background_proposals = (best_match_iou < self.model_config.roi_iou_threshold) & (best_match_iou >= self.model_config.roi_low_bg_iou)
            ignored_proposals = best_match_iou < self.model_config.roi_low_bg_iou
            
            # Update best match of low IOU proposals to -1
            best_match_gt_idx[background_proposals] = -1
            best_match_gt_idx[ignored_proposals] = -2
            
            # Get best marching gt boxes for ALL proposals
            # Even background proposals would have a gt box assigned to it
            # Label will be used to ignore them later
            matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)]
            
            # Get class label for all proposals according to matching gt boxes
            labels = gt_labels[best_match_gt_idx.clamp(min=0)]
            labels = labels.to(dtype=torch.int64)
            
            # Update background proposals to be of label 0(background)
            labels[background_proposals] = 0
            
            # Set all to be ignored anchor labels as -1(will be ignored)
            labels[ignored_proposals] = -1
            
            return labels, matched_gt_boxes_for_proposals
        except Exception as e:
            raise CustomException(e,sys)
    
    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        
        try:
            # remove low scoring boxes
            keep = torch.where(pred_scores > self.model_config.roi_scrore_threshold)[0]
            pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
            
            # Remove small boxes
            min_size = 16
            ws, hs = pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]
            keep = (ws >= min_size) & (hs >= min_size)
            keep = torch.where(keep)[0]
            pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
            
            # Class wise nms
            keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
            for class_id in torch.unique(pred_labels):
                curr_indices = torch.where(pred_labels == class_id)[0]
                curr_keep_indices = torch.ops.torchvision.nms(pred_boxes[curr_indices],
                                                            pred_scores[curr_indices],
                                                            self.model_config.roi_nms_threshold)
                keep_mask[curr_indices[curr_keep_indices]] = True
            keep_indices = torch.where(keep_mask)[0]
            post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(descending=True)[1]]
            keep = post_nms_keep_indices[:self.model_config.roi_topk_detection]
            pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
            return pred_boxes, pred_labels, pred_scores
        except Exception as e:
            raise CustomException(e,sys)
        
    def forward(self, feat, proposals, image_shape, target):
        
        try:
            if target is not None:
                proposals = torch.cat([proposals, target['bboxes']], dim=0)
                
                gt_boxes = target['bboxes']
                gt_labels = target['labels']
                
                labels, matched_gt_boxes_for_proposals = self.assign_target_to_proposals(proposals, gt_boxes, gt_labels)
                
                num_positive_samples = int(self.model_config.roi_batch_size * self.model_config.roi_pos_fraction)
                
                sampled_neg_idx_mask, sampled_pos_idx_mask = training_utils.sample_positive_negative(labels,
                                                                                    positive_count=num_positive_samples,
                                                                                    total_count=self.model_config.roi_batch_size)
                
                sampled_idxs = torch.where(sampled_pos_idx_mask | sampled_neg_idx_mask)[0]
                
                proposals = proposals[sampled_idxs]
                labels = labels[sampled_idxs]
                matched_gt_boxes_for_proposals = matched_gt_boxes_for_proposals[sampled_idxs]
                regression_targets = training_utils.boxes_to_transformation_targets(matched_gt_boxes_for_proposals, proposals)
                
            size = feat.shape[-2:]
            possible_scales = []
            for s1, s2 in zip(size, image_shape):
                approx_scale = float(s1) / float(s2)
                scale = 2 ** float(torch.tensor(approx_scale).log2().round())
                possible_scales.append(scale)
            assert possible_scales[0] == possible_scales[1]
                
            # ROI pooling and call all layers for prediction
            proposal_roi_pool_feats = torchvision.ops.roi_pool(feat, [proposals],
                                                            output_size=self.model_config.roi_pool_size,
                                                            spatial_scale=possible_scales[0])
            proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)
            box_fc_6 = torch.nn.functional.relu(self.fc6(proposal_roi_pool_feats))
            box_fc_7 = torch.nn.functional.relu(self.fc7(box_fc_6))
            cls_scores = self.cls_layer(box_fc_7)
            box_transform_pred = self.bbox_reg_layer(box_fc_7)
            # cls_scores -> (proposals, num_classes)
            # box_transform_pred -> (proposals, num_classes * 4)
            ##############################################
            
            num_boxes, num_classes = cls_scores.shape
            box_transform_pred = box_transform_pred.reshape(num_boxes, num_classes, 4)
            frcnn_output = {}
            if target is not None:
                classification_loss = torch.nn.functional.cross_entropy(cls_scores, labels)
                
                # Compute localization loss only for non-background labelled proposals
                fg_proposals_idxs = torch.where(labels > 0)[0]
                # Get class labels for these positive proposals
                fg_cls_labels = labels[fg_proposals_idxs]
                
                localization_loss = torch.nn.functional.smooth_l1_loss(
                    box_transform_pred[fg_proposals_idxs, fg_cls_labels],
                    regression_targets[fg_proposals_idxs],
                    beta=1/9,
                    reduction="sum",
                )
                localization_loss = localization_loss / labels.numel()
                frcnn_output['frcnn_classification_loss'] = classification_loss
                frcnn_output['frcnn_localization_loss'] = localization_loss
            
            if self.training:
                return frcnn_output
            else:
                device = cls_scores.device
                # Apply transformation predictions to proposals
                pred_boxes = training_utils.apply_regression_pred_to_anchors_or_proposals(box_transform_pred, proposals)
                pred_scores = torch.nn.functional.softmax(cls_scores, dim=-1)
                
                # Clamp box to image boundary
                pred_boxes = training_utils.clamp_boxes_to_image_boundary(pred_boxes, image_shape)
                
                # create labels for each prediction
                pred_labels = torch.arange(num_classes, device=device)
                pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)
                
                # remove predictions with the background label
                pred_boxes = pred_boxes[:, 1:]
                pred_scores = pred_scores[:, 1:]
                pred_labels = pred_labels[:, 1:]
                
                # pred_boxes -> (number_proposals, num_classes-1, 4)
                # pred_scores -> (number_proposals, num_classes-1)
                # pred_labels -> (number_proposals, num_classes-1)
                
                # batch everything, by making every class prediction be a separate instance
                pred_boxes = pred_boxes.reshape(-1, 4)
                pred_scores = pred_scores.reshape(-1)
                pred_labels = pred_labels.reshape(-1)
                
                pred_boxes, pred_labels, pred_scores = self.filter_predictions(pred_boxes, pred_labels, pred_scores)
                
                frcnn_output['boxes'] = pred_boxes
                frcnn_output['scores'] = pred_scores
                frcnn_output['labels'] = pred_labels
                
                return frcnn_output
        except Exception as e:
            raise CustomException(e,sys)    