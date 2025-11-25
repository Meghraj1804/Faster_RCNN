import os
from datetime import datetime
from src import constants
import torch


class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.artifact_name = constants.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name,timestamp)
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.timestamp: str = timestamp
        
        
        

class DataIngestionConfig:
    def __init__(self,train_pipeline_config:TrainingPipelineConfig):
        
        self.train_image_path:str = constants.TRAIN_IMAGE_DIR
        self.train_annotation_path:str = constants.TRAIN_ANNOTATION_DIR
        
        self.test_image_path:str = constants.TEST_IMAGE_DIR
        self.test_annotation_path:str = constants.TEST_ANNOTATION_DIR
        
        self.val_image_path:str = constants.VAL_IMAGE_DIR
        self.val_annotation_path:str = constants.VAL_ANNOTATION_DIR
        
        self.batch_size = constants.BATCH_SIZE
        
        self.classes = constants.CLASSES
        self.num_classes = len(self.classes) + 1 # all classes + background class
        
class ModelParamConfig:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        
        self.image_channels = constants.IMAGE_CHANNELS
        self.min_image_size = constants.MIN_IMAGE_SIZE
        self.max_image_size = constants.MAX_IMAGE_SIZE
        self.image_mean = constants.IMAGE_MEAN
        self.image_std = constants.IMAGE_STANDARD_DEVIATION
        
        self.num_classes = data_ingestion_config.num_classes
        self.classes = data_ingestion_config.classes
        
        self.rpn_backbone_out_channels = constants.RPN_BACKBONE_OUT_CHANNELS
        self.rpn_aspects_ratios = constants.RPN_ASPECT_RATIOS
        self.rpn_scales = constants.RPN_SCALES
        self.rpn_background_threshold = constants.RPN_BACKGROUND_THRESHOLD
        self.rpn_foreground_threshold = constants.RPN_FOREGROUND_THRESHOLD
        self.rpn_nms_threshold = constants.RPN_NMS_THRESHOLD
        self.rpn_train_prenms_topk = constants.RPN_TRAIN_PRENMS_TOPK
        self.rpn_test_prenms_topk = constants.RPN_TEST_PRENMS_TOPK
        self.rpn_train_topk = constants.RPN_TRAIN_TOPK
        self.rpn_test_topk = constants.RPN_TEST_TOPK
        self.rpn_batch_size = constants.RPN_BATCH_SIZE
        self.rpn_pos_fraction = constants.RPN_POS_FRACTION
        
        self.roi_backbone_out_channels = constants.ROI_BACKBONE_OUT_CHANNELS
        self.roi_iou_threshold = constants.ROI_IOU_THRESHOLD
        self.roi_low_bg_iou = constants.ROI_LOW_BG_IOU
        self.roi_pool_size = constants.ROI_POOL_SIZE
        self.roi_nms_threshold = constants.ROI_NMS_THRESHOLD
        self.roi_topk_detection = constants.ROI_TOPK_DETECTION
        self.roi_scrore_threshold = constants.ROI_SCORE_THRESHOLD
        self.roi_batch_size = constants.ROI_BATCH_SIZE
        self.roi_pos_fraction = constants.ROI_POS_FRACTION
        self.roi_fc_inner_dim = constants.ROI_FC_INNER_DIM
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
class TrainerParamConfig:
    def __init__(self,trinaing_pipline_config:TrainingPipelineConfig):
        self.task_name:str = constants.TASK_NAME
        self.seed = constants.SEED
        self.acc_steps = constants.ACC_STEPS
        self.epochs = constants.EPOCHS
        self.lr_steps = constants.LR_STEPS
        self.lr = constants.LR
        self.ckpt_name = constants.CKPT_NAME
        
        self.model_dir = os.path.join(trinaing_pipline_config.artifact_dir, self.task_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, self.ckpt_name + ".pth")
         