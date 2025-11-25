import torch
import numpy as np
import cv2
import argparse
import random
import os
import yaml
from tqdm import tqdm
from PIL import Image
import torchvision
from src.entity.config_entity import ModelParamConfig,TrainingPipelineConfig, DataIngestionConfig
from src.model.fast_rcnn_network import FasterRCNN
from src.data_loader.load_voc_data import VOCDataset
from torch.utils.data.dataloader import DataLoader
from src.training.training_utils import get_iou, compute_map



class Infer:
    def __init__(self):
        self.train_pipeline_config = TrainingPipelineConfig()
        self.data_ingestion_config = DataIngestionConfig(self.train_pipeline_config)
        self.model_param_config = ModelParamConfig(self.data_ingestion_config)

    def load_model(self,model_path:str )->FasterRCNN:
        faster_rcnn_model = FasterRCNN(self.model_param_config)
        device = self.model_param_config.device
        faster_rcnn_model.eval()
        faster_rcnn_model.to(device)
        faster_rcnn_model.load_state_dict(torch.load(model_path,
                                                    map_location=device))
        return faster_rcnn_model


    def infer(self,model:FasterRCNN, image_path:str):
        if not os.path.exists('samples'):
            os.mkdir('samples')
            
        im = Image.open(image_path)
        im = torchvision.transforms.ToTensor()(im)
        im = im.unsqueeze(0).float().to(ModelParamConfig(self.data_ingestion_config).device)
        # Getting predictions from trained model
        rpn_output, frcnn_output = model(im, None)
        # print('out = ',frcnn_output)
        boxes = frcnn_output[0]['boxes']
        labels = frcnn_output[0]['labels']
        scores = frcnn_output[0]['scores']
        image = cv2.imread(image_path)
        # image_copy = image.copy()
        # id = scores.argmax()
        
        # box = boxes[id]
        # score = scores[id]
        
        # x1, y1, x2, y2 = box.detach().cpu().numpy()
        # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # # Draw bounding box
        # image = cv2.rectangle(img = image,
        #                     pt1 = (x1,y1),
        #                     pt2 = (x2,y2),
        #                     color = (0,0,255),
        #                     thickness=2) # type: ignore

        # output_path = 'samples/output_frcnn_.jpg'
        # cv2.imwrite(output_path, image)
        
        # return output_path
        
        
        for box, label, score in zip(boxes, labels, scores):
            if score < 0.8:
                continue  # skip low-confidence detections

            x1, y1, x2, y2 = map(int, box)

            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255), 2
            )

        output_path = 'samples/output_frcnn_.jpg'
        cv2.imwrite(output_path, image)

        return output_path
            
            
            
            
                
                



                
                
