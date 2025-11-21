import sys
import torch
import torch.nn as nn
import torchvision

from src.exception.exception import CustomException
from src.logging.logger import logging
from src.model.rpn_network import RegionProposalNetwork
from src.model.roi_head_network import ROIHead

from src.entity.config_entity import ModelParamConfig
from src.training import training_utils


class FasterRCNN(nn.Module):
    def __init__(self,model_config:ModelParamConfig):
        super(FasterRCNN, self).__init__()
        self.model_config = model_config
        
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.backbone = vgg16.features[:-1]
        
        self.rpn = RegionProposalNetwork(model_config)
        
        self.roi_head = ROIHead(model_config)
        
        # freeze first 10 layers of backbone 
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        
        self.image_mean = model_config.image_mean
        self.image_std = model_config.image_std
        
        self.min_size = model_config.min_image_size
        self.max_size = model_config.max_image_size     
    
    def normalize_resize_image_and_boxes(self, image, bboxes):
        
        try:
            dtype, device = image.dtype, image.device
            # Normalize
            mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
            std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
            image = (image - mean[:, None, None]) / std[:, None, None]
            

            # Original size
            h, w = image.shape[-2:]
            
            # Compute scaling factor
            min_size = min(h, w)
            max_size = max(h, w)
            scale = min(float(self.min_size) / min_size, float(self.max_size) / max_size)
            
            # Resize image
            image = image.unsqueeze(0)  # add batch dim for interpolate
            new_h = int(h * scale + 0.5)
            new_w = int(w * scale + 0.5)
            image = torch.nn.functional.interpolate(
                image,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)  # remove batch dim

            # Resize boxes if provided
            if bboxes is not None:
                ratio_h = new_h / h
                ratio_w = new_w / w
                xmin, ymin, xmax, ymax = bboxes.unbind(1)
                xmin = xmin * ratio_w
                xmax = xmax * ratio_w
                ymin = ymin * ratio_h
                ymax = ymax * ratio_h
                bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)

            return image, bboxes
        except Exception as e:
            raise CustomException(e,sys)
        
        
    
    def forward(self, images, targets=None):
        try:
        # Save original image shapes to transform boxes back after inference
            old_shapes = [img.shape[-2:] for img in images]

            # Normalize and optionally resize images and boxes
            images_resized = []
            targets_resized = []
            # if self.training and targets is not None:
            if targets is not None:
                for img, tgt in zip(images, targets):
                    
                    img, bboxes = self.normalize_resize_image_and_boxes(img, tgt['bboxes'])
                    images_resized.append(img)
                    tgt_resized = tgt.copy()
                    tgt_resized['bboxes'] = bboxes
                    targets_resized.append(tgt_resized)
                targets = targets_resized
            # else:
            elif targets is None:
                for img in images:
                    img, _ = self.normalize_resize_image_and_boxes(img, None)
                    images_resized.append(img)
            images = images_resized

            # Pass images through backbone
            feats = [self.backbone(img.unsqueeze(0)) for img in images]  # add batch dim

            # RPN: generate proposals for each image
            rpn_outputs = []
            proposals_list = []
            for img, feat, tgt in zip(images, feats, targets if targets else [None]*len(images)):
                rpn_out = self.rpn(img, feat, tgt)
                rpn_outputs.append(rpn_out)
                proposals_list.append(rpn_out['proposals'])

            # ROI Head: Fast R-CNN head
            frcnn_outputs = []
            for feat, proposals, img, tgt, old_shape in zip(
                feats, proposals_list, images, targets if targets else [None]*len(images), old_shapes
            ):
                out = self.roi_head(feat, proposals, img.shape[-2:], tgt)
                
                if not self.training and targets is None:
                    # Transform boxes back to original image dimensions
                    out['boxes'] = training_utils.transform_boxes_to_original_size(out['boxes'], img.shape[-2:], old_shape)
                frcnn_outputs.append(out)
            # print('frcnn_type = ',len(frcnn_outputs))
            return rpn_outputs, frcnn_outputs
        except Exception as e:
            raise CustomException(e,sys)
