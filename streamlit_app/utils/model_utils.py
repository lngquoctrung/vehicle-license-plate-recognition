import streamlit as st
import torch
import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
root_dir = str(Path(__file__).parent.parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
model_path = f"{root_dir}/model_checkpoint/"
model_filepath = f"{model_path}/faster_rcnn_checkpoints.pt"

from src.config import NUM_CLASSES, DEVICE
from src.model import FTFasterRCNN
from src.utils import create_directory

def download_model():
    if not os.path.exists(model_path):
        create_directory(model_path)
    subprocess.run([
        "curl", "-L", "https://github.com/lngquoctrung/vehicle-license-plate-recognition/releases/download/v1.0.0/best_faster_rcnn_checkpoints.pt", 
        "-o", model_filepath
    ])

@st.cache_resource
def load_model():
    """
    Load model from checkpoints with caching to avoid reloading
    """
    # Download model if not exists
    if not os.path.exists(model_filepath):
        download_model()
    # Read model
    try:
        model = FTFasterRCNN(num_classes=NUM_CLASSES, freeze_backbone=True)
        checkpoints = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoints["state_dict"])
        model.eval()
        
        st.success(f"✅ Model loaded from: {model_path}")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def get_model_info():
    """
    Return model information
    """
    info = {
        "Model": "Faster R-CNN",
        "Backbone": "ResNet-50",
        "Classes": NUM_CLASSES,
        "Device": DEVICE,
        "Input Size": "Flexible",
        "Framework": "PyTorch"
    }
    return info
