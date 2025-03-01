import sys
import os

sys.path.insert

import torch

from src.model import FTFasterRCNN
from torchvision import transforms

if __name__ == "__main__":
    image_filenames = os.listdir("./images")
    
    print(image_filenames)
    pass