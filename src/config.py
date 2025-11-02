import os
import numpy as np
import torch

from pathlib import Path
root_path = Path(__file__).parent.parent.absolute()

class Config:
    # Hardware
    NUM_WORKERS = os.cpu_count()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset url
    DATA_URL = "https://github.com/lngquoctrung/vehicle-license-plate-recognition/releases/download/v1.0.0"

    # Directories
    DATA_DIR = str(root_path / "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    MODEL_DIR = str(root_path / "models")
    CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

    # Image configurations
    IMAGE_SIZE = (416, 416)
    MEAN_NORMALIZATION = np.array([0.485, 0.456, 0.406])
    STD_NORMALIZATION = np.array([0.229, 0.224, 0.225])

    # Model configurations
    NUM_CLASSES = 2
    IOU_THRESHOLD = 0.5
    SCORE_THRESHOLD = 0.3

    # Training configurations
    BATCH_SIZE = 8
    EPOCHES = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Optimizer configurations
    MOMENTUM = 0.9

    # Learning Rate Scheduler configurations
    STEP_SIZE = 10
    GAMMA = 0.1

    # Warmup configurations
    WARMUP_EPOCHS = 5

    # Early Stopping configurations
    EARLY_STOPPING_PATIENCE = 15

    # Data Augmentation configurations
    USE_AUGMENTATION = True
    AUGMENTATION_PROBABILITY = 0.5

    # Multi-scale Testing
    TEST_SCALES = [0.8, 1.0, 1.2]

    @classmethod
    def to_dict(cls):
        return {
            "num_workers": cls.NUM_WORKERS,
            "device": cls.DEVICE,
            "data_url": cls.DATA_URL,
            "data_dir": cls.DATA_DIR,
            "raw_data_dir": cls.RAW_DATA_DIR,
            "processed_data_dir": cls.PROCESSED_DATA_DIR,
            "model_dir": cls.MODEL_DIR,
            "checkpoint_dir": cls.CHECKPOINT_DIR,
            "image_size": cls.IMAGE_SIZE,
            "mean_normalization": cls.MEAN_NORMALIZATION,
            "std_normalization": cls.STD_NORMALIZATION,
            "num_classes": cls.NUM_CLASSES,
            "iou_threshold": cls.IOU_THRESHOLD,
            "score_threshold": cls.SCORE_THRESHOLD,
            "batch_size": cls.BATCH_SIZE,
            "epoches": cls.EPOCHES,
            "learning_rate": cls.LEARNING_RATE,
            "weight_decay": cls.WEIGHT_DECAY,
            "momentum": cls.MOMENTUM,
            "step_size": cls.STEP_SIZE,
            "gamma": cls.GAMMA,
            "warmup_epochs": cls.WARMUP_EPOCHS,
            "early_stopping_patience": cls.EARLY_STOPPING_PATIENCE,
            "use_augmentation": cls.USE_AUGMENTATION,
            "augmentation_probability": cls.AUGMENTATION_PROBABILITY,
            "test_scales": cls.TEST_SCALES,
        }