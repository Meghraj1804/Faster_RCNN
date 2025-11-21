from src.exception.exception import CustomException
from src.logging.logger import logging

from src.entity.config_entity import TrainerParamConfig, TrainingPipelineConfig, DataIngestionConfig, ModelParamConfig
from src.data_loader.load_voc_data import VOCDataset

# from src.model.fast_rcnn_network import FasterRCNN
from src.training.training_pipeline import TrainingPipeline

from torch.utils.data import DataLoader

import torch
import os
from tqdm import tqdm


import sys


def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    try:
        train_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(train_pipeline_config)
        logging.info('Training Pipeline initiated')
        
        train_voc = VOCDataset('train', 
                        image_path = data_ingestion_config.train_image_path,
                        annotation_path = data_ingestion_config.train_annotation_path,
                        classes=data_ingestion_config.classes,
                        num_classes=data_ingestion_config.num_classes)
    
        train_dataset = DataLoader(train_voc,
                            batch_size=data_ingestion_config.batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)
        
        logging.info('Train data loaded successfully')
        
        val_voc = VOCDataset('val', 
                        image_path = data_ingestion_config.val_image_path,
                        annotation_path = data_ingestion_config.val_annotation_path,
                        classes=data_ingestion_config.classes,
                        num_classes=data_ingestion_config.num_classes)
    
        val_dataset = DataLoader(val_voc,
                            batch_size=data_ingestion_config.batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)
        
        logging.info('validation data loaded successfully')
        
        test_voc = VOCDataset('test', 
                        image_path = data_ingestion_config.test_image_path,
                        annotation_path = data_ingestion_config.test_annotation_path,
                        classes=data_ingestion_config.classes,
                        num_classes=data_ingestion_config.num_classes)
    
        test_dataset = DataLoader(test_voc,
                            batch_size=data_ingestion_config.batch_size,
                            shuffle=True,
                            collate_fn=collate_fn)
        
        logging.info('test data loaded successfully')
        
        model_config = ModelParamConfig(data_ingestion_config)
        
        trainer_config = TrainerParamConfig(train_pipeline_config)
        
        model = TrainingPipeline( data_ingestion_config, model_config, trainer_config )
        
        logging.info('Training Started')
        
        
        
        
        
        for i in range(trainer_config.epochs):
            print('_'*100)
            print('Epoch = ',i)
            rpn_model_classification_losses = []
            rpn_model_localization_losses = []
            frcnn_model_classification_losses = []
            frcnn_model_localization_losses = []
            
            print('Training')
            train_loss = model.train(train_dataset)
            print(f'Training Loss = {train_loss}')
            
            print('Validation')
            validation_loss = model.validate(val_dataset)
            print(f'Validation Loss = {validation_loss}')
            
            print('Evaluation')
            mAP = model.evaluate_map(test_dataset, test_voc.label2idx, test_voc.idx2label)
            print('mAP = ',mAP)           
        
            # if i%3 == 0:
            model.save_model(i)
            
        model.save_model(trainer_config.epochs)
            
            
            
    except Exception as e:
        raise CustomException(e, sys)