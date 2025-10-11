import os
import numpy as np
import torch

NUM_WORKERS = os.cpu_count()
DATA_URL = "https://github.com/lngquoctrung/vehicle-license-plate-recognition/releases/download/v1.0.0"
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODEL_DIR = "models"
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
FINAL_STATE_DIR = os.path.join(MODEL_DIR, "finals")

IMAGE_SIZE = (224, 224)
MEAN_NORMALIZATION = np.array([0.485, 0.456, 0.406])
STD_NORMALIZATION = np.array([0.229, 0.224, 0.225])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 2
BATCH_SIZE = 16
EPOCHES = 30
IOU_THRESHOLD = 0.5

LEARNING_RATE = 5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
STEP_SIZE = 10
GAMMA = 0.1