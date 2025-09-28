import streamlit as st
import torch
import os
from pathlib import Path
import sys

# Add project root to path
root_dir = str(Path(__file__).parent.parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import NUM_CLASSES, DEVICE
from src.model import FTFasterRCNN

@st.cache_resource
def load_model():
    """
    Load model from checkpoints with caching to avoid reloading
    """
    try:
        model = FTFasterRCNN(num_classes=NUM_CLASSES, freeze_backbone=True)
        
        # Search for checkpoint file
        checkpoint_paths = [
            f"{root_dir}/models/final/best_faster_rcnn_checkpoints.pt",
            f"{root_dir}/models/checkpoints/faster_rcnn_checkpoints.pt",
        ]
        
        checkpoint_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                checkpoint_path = path
                break
                
        if checkpoint_path is None:
            raise FileNotFoundError("Checkpoint file not found")
            
        checkpoints = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoints["state_dict"])
        model.eval()
        
        st.success(f"✅ Model loaded from: {checkpoint_path}")
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
