import os
import numpy as np

NUM_WORKERS = os.cpu_count()

DATA_URL = "https://raw.githubusercontent.com/lngquoctrung/vehicle-license-plate-recognition/refs/heads/main/large_file_parts"
DATA_DIR = "data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

IMAGE_SIZE = (240, 240)
MEAN_NORMALIZATION = np.array([0.485, 0.456, 0.406])
STD_NORMALIZATION = np.array([0.229, 0.224, 0.225])

BATCH_SIZE = 32