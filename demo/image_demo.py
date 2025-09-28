import sys
import os
import warnings
# Add the project folder to environmental variable PYTHONPATH
from pathlib import Path

root_dir = str(Path(__file__).parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)
warnings.filterwarnings("ignore")

import torch
import cv2
import numpy as np
from PIL import Image

from src.model import FTFasterRCNN
from src.config import NUM_CLASSES, DEVICE, MEAN_NORMALIZATION, STD_NORMALIZATION, IMAGE_SIZE
from torchvision import transforms

def load_model(state_path):
    """
        Load model from the checkpoints
    """
    model = FTFasterRCNN(num_classes=NUM_CLASSES, freeze_backbone=True)
    checkpoints = torch.load(state_path)
    model.load_state_dict(checkpoints["state_dict"])
    return model

def preprocess_data(image_path):
    """
        Read and preprocess image
    """
    # Read image
    image = cv2.imread(image_path)
    # Convert image color
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert numpy to tensor
    image = Image.fromarray(image)

    # Transform image
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_NORMALIZATION, std=STD_NORMALIZATION)
    ])

    # Apply image transformation
    image = transform(image)

    return image

def predict(model, image):
    # Add model to GPU
    model.eval()
    model.to(DEVICE)

    # Add batch dimension
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(image)

    return {key: value.cpu() for key, value in prediction[0].items()}

def display_image_and_boxes(image, prediction, score_threshold=0.5, display_scale=2.0):
    window_name = "Result"
    
    # Get boxes và scores from prediction
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    
    # Filter boxes have score > threshold
    high_score_indices = torch.where(scores > score_threshold)[0]
    filtered_boxes = boxes[high_score_indices]
    filtered_scores = scores[high_score_indices]

    # Restore image
    original_image = image * torch.tensor(STD_NORMALIZATION)[:, None, None] + torch.tensor(MEAN_NORMALIZATION)[:, None, None]

    # Convert from tensor to numpy
    numpy_image = original_image.numpy()
    numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)

    # Transpose image dimension
    transposed_image = np.transpose(numpy_image, (1, 2, 0))
    
    # Make sure the image is BGR
    transposed_image = cv2.cvtColor(transposed_image, cv2.COLOR_RGB2BGR)

    # Draw bounding boxes
    filtered_boxes = filtered_boxes.round().to(torch.int)
    filtered_boxes = filtered_boxes.numpy()

    # Print the number of boxes were displayed
    print(f"Displaying {len(filtered_boxes)} boxes with scores > {score_threshold}")
    
    for i, box in enumerate(filtered_boxes):
        # Points of rectangle
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))

        # Color của rectangle - Azure color (BGR color for OpenCV)
        color = (0, 235, 255)
        # Thickness của rectangle
        thickness = 2

        # Draw rectangle cho image
        cv2.rectangle(transposed_image, start_point, end_point, color, thickness)
        
        # Hiển thị score
        score_text = f"{filtered_scores[i].item():.2f}"
        cv2.putText(transposed_image, score_text, (int(box[0]), int(box[1]-5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Resize image before displaying
    h, w = transposed_image.shape[:2]
    display_image = cv2.resize(transposed_image, 
                              (int(w * display_scale), int(h * display_scale)), 
                              interpolation=cv2.INTER_AREA)
    
    # Show image
    cv2.imshow(window_name, display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load model
    model = load_model(f"{root_dir}/models/final/best_faster_rcnn_checkpoints.pt")

    # Display selection list
    INPUT_PATH = f"{root_dir}/demo/inputs/images/"
    print("Choose any image from the image folder:")
    print("-"*15)
    image_filenames = os.listdir(INPUT_PATH)
    for idx, image_filename in enumerate(image_filenames):
        print(f"{idx}. ", image_filename)
    print("-"*15)
    image_idx = int(input("Enter the index of image: ").strip())
    image_path = os.path.join(INPUT_PATH, image_filenames[image_idx])

    # Read and preprocess the image
    image = preprocess_data(image_path)

    # Predict
    prediction = predict(model, image)

    # Display image
    display_image_and_boxes(image, prediction, score_threshold=0.4, display_scale=2.0)