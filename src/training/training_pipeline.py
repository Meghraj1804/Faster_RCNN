import torch
import argparse
import os
import numpy as np
import yaml
import sys
import random
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.model.fast_rcnn_network import FasterRCNN
from src.entity.config_entity import  ModelParamConfig, TrainerParamConfig, DataIngestionConfig
from src.data_loader.load_voc_data import VOCDataset
from tqdm import tqdm
# from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from src.training import training_utils


class TrainingPipeline:
    def __init__(self,data_ingestion_config:DataIngestionConfig,
                 model_config:ModelParamConfig,
                 trainer_config:TrainerParamConfig):
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.data_ingestion_config = data_ingestion_config
        
        self.faster_rcnn = FasterRCNN(self.model_config)
        
        self.faster_rcnn.to(model_config.device)
        
        self.optimizer = torch.optim.SGD(lr=self.trainer_config.lr,
                                params=filter(lambda p: p.requires_grad,self.faster_rcnn.parameters()),
                                weight_decay=5E-4,
                                momentum=0.9)
        
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.trainer_config.lr_steps, gamma=0.1)
        
        logging.info('Model Initiated Successfully')
        
        self.acc_steps = self.trainer_config.acc_steps
        self.epochs = self.trainer_config.epochs
        self.step_count = 1
        
    def train(self, dataloader:DataLoader):
        self.faster_rcnn.train()
        train_loss = []
        self.optimizer.zero_grad()
        

        
        try:
            for im, target, fname in tqdm(dataloader):
                
                images = [img.float().to(self.model_config.device) for img in im]
                targets = [
                    {
                        'bboxes': t['bboxes'].float().to(self.model_config.device),
                        'labels': t['labels'].long().to(self.model_config.device)
                    }
                    for t in target
                ]
                
                rpn_output, frcnn_output = self.faster_rcnn(images, targets)
                
                rpn_loss = rpn_output[0]['rpn_classification_loss'] + rpn_output[0]['rpn_localization_loss']
                frcnn_loss = frcnn_output[0]['frcnn_classification_loss'] + frcnn_output[0]['frcnn_localization_loss']
                loss = rpn_loss + frcnn_loss
                loss = loss / self.acc_steps
                loss.backward()
                train_loss.append(loss.item())
                if self.step_count % self.acc_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.step_count += 1
            
                # self.optimizer.step()
                # self.optimizer.zero_grad()
            
            return sum(train_loss)/len(train_loss)
        except Exception as e:
            raise CustomException(e,sys)
    
    def validate(self, dataloader: DataLoader):
        try:
            self.faster_rcnn.eval()
            samples = len(dataloader) * self.data_ingestion_config.batch_size
            validation_loss = []

            with torch.no_grad():
                for im, target, fname in tqdm(dataloader):

                    images = [img.float().to(self.model_config.device) for img in im]
                    targets = [
                        {
                            'bboxes': t['bboxes'].float().to(self.model_config.device),
                            'labels': t['labels'].long().to(self.model_config.device)
                        }
                        for t in target
                    ]

                    # Forward pass only
                    rpn_output, frcnn_output = self.faster_rcnn(images, targets)
                    rpn_loss = rpn_output[0]['rpn_classification_loss'] +  rpn_output[0]['rpn_localization_loss']
                    frcnn_loss = frcnn_output[0]['frcnn_classification_loss'] + frcnn_output[0]['frcnn_localization_loss']
                    loss = rpn_loss + frcnn_loss
                    # loss = loss / self.acc_steps
                    validation_loss.append(loss.item())
                    
            return sum(validation_loss)/len(validation_loss)
        except Exception as e:
            raise CustomException(e,sys)
    
    
    def evaluate_map(self, dataloader: DataLoader, label2idx, idx2label):

        try:
            self.faster_rcnn.eval()
            gts = []
            preds = []

            for im, target, fname in tqdm(dataloader):

                images = [img.float().to(self.model_config.device) for img in im]

                # GT targets (keep on CPU)
                targets = [
                    {
                        'bboxes': t['bboxes'].cpu().numpy(),
                        'labels': t['labels'].cpu().numpy()
                    }
                    for t in target
                ]

                # Only detection, no GT passed
                rpn_output, frcnn_output = self.faster_rcnn(images, None)

                # Each batch contains 1 image → first output
                boxes = frcnn_output[0]['boxes'].detach().cpu().numpy()
                labels = frcnn_output[0]['labels'].detach().cpu().numpy()
                scores = frcnn_output[0]['scores'].detach().cpu().numpy()

                # Prepare dicts for this image
                pred_boxes = {cls: [] for cls in label2idx}
                gt_boxes   = {cls: [] for cls in label2idx}

                # -------------------------
                # Add PREDICTIONS
                # -------------------------
                for box, label, score in zip(boxes, labels, scores):
                    cls_name = idx2label[int(label)]
                    x1, y1, x2, y2 = box.tolist()
                    pred_boxes[cls_name].append([x1, y1, x2, y2, float(score)])

                # -------------------------
                # Add GROUND TRUTH
                # -------------------------
                for t in targets:
                    for box, label in zip(t['bboxes'], t['labels']):
                        cls_name = idx2label[int(label)]
                        x1, y1, x2, y2 = box.tolist()
                        gt_boxes[cls_name].append([x1, y1, x2, y2])

                preds.append(pred_boxes)
                gts.append(gt_boxes)

            # Compute mAP
            mean_ap, all_aps = training_utils.compute_map(preds, gts, method='interp')
            return mean_ap

        except Exception as e:
            raise CustomException(e, sys)

    
    def save_model(self, epoch):
        
        ckpt_name = f'best_{epoch}'
        model_path = os.path.join(self.trainer_config.model_dir, ckpt_name + ".pth")
        torch.save(self.faster_rcnn.state_dict(), model_path)
        print('model_saved')
        