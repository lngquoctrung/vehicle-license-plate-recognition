"""Model utilities for loading and managing ML models"""

import torch
import os
import requests
import sys

from io import BytesIO
from pathlib import Path
from functools import lru_cache

# Add project root to path
root_dir = str(Path(__file__).parent.parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import NUM_CLASSES, DEVICE
from src.model import FTFasterRCNN

@lru_cache(maxsize=1)
def load_model():
    """
    Load model from checkpoints with caching to avoid reloading
    Uses lru_cache instead of st.cache_resource
    """
    try:
        checkpoints = torch.load("/app/model/faster_rcnn_checkpoint.pt", map_location=DEVICE)
        model = FTFasterRCNN(num_classes=NUM_CLASSES, freeze_backbone=True)
        model.load_state_dict(checkpoints["state_dict"])
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def get_model_info():
    """Return model information"""
    info = {
        "Model": "Faster R-CNN",
        "Backbone": "ResNet-50",
        "Classes": NUM_CLASSES,
        "Device": DEVICE,
        "Input Size": "Flexible",
        "Framework": "PyTorch"
    }
    return info
