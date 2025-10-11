"""Image processing utilities for license plate detection"""

import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
from pathlib import Path

# Add project root to path
root_dir = str(Path(__file__).parent.parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import MEAN_NORMALIZATION, STD_NORMALIZATION, IMAGE_SIZE

def preprocess_image(uploaded_file):
    """
    Preprocess image for model inference
    
    Args:
        uploaded_file: PIL Image or file path
        
    Returns:
        processed_image: Tensor ready for model
        image: Original PIL Image
    """
    try:
        # Handle PIL Image or file path
        if isinstance(uploaded_file, str):
            image = Image.open(uploaded_file)
        elif isinstance(uploaded_file, Image.Image):
            image = uploaded_file
        else:
            # Assume numpy array from Gradio
            image = Image.fromarray(uploaded_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN_NORMALIZATION,
                std=STD_NORMALIZATION
            )
        ])
        
        # Transform image
        processed_image = transform(image).unsqueeze(0)  # Add batch dimension
        
        return processed_image, image
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None, None

def draw_detection_boxes(image, prediction, score_threshold=0.5):
    """
    Draw bounding boxes on image after prediction
    
    Args:
        image: Normalized tensor image
        prediction: Model prediction dict
        score_threshold: Minimum confidence score
        
    Returns:
        transposed_image: Image with bounding boxes (numpy array)
        num_detections: Number of detected boxes
    """
    try:
        # Restore image from normalized tensor
        original_image = (
            image * torch.tensor(STD_NORMALIZATION)[:, None, None] + 
            torch.tensor(MEAN_NORMALIZATION)[:, None, None]
        )
        
        # Convert from tensor to numpy
        numpy_image = original_image.numpy()
        numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
        
        # Transpose image dimensions (C, H, W) -> (H, W, C)
        transposed_image = np.transpose(numpy_image, (1, 2, 0))
        
        # Ensure array is contiguous for OpenCV
        transposed_image = np.ascontiguousarray(transposed_image)
        
        if prediction is None:
            return transposed_image, 0
        
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        
        # Filter boxes with score > threshold
        high_score_indices = torch.where(scores > score_threshold)[0]
        
        if len(high_score_indices) == 0:
            return transposed_image, 0
        
        filtered_boxes = boxes[high_score_indices]
        filtered_scores = scores[high_score_indices]
        
        # Convert boxes to numpy
        filtered_boxes = filtered_boxes.round().to(torch.int)
        filtered_boxes = filtered_boxes.numpy()
        
        # Draw bounding boxes
        for i, box in enumerate(filtered_boxes):
            start_point = (int(box[0]), int(box[1]))
            end_point = (int(box[2]), int(box[3]))
            
            # Cyan color for RGB
            color = (255, 255, 0)
            thickness = 2
            
            # Draw rectangle
            cv2.rectangle(transposed_image, start_point, end_point, color, thickness)
            
            # Display confidence score
            score_text = f"Plate: {filtered_scores[i].item():.3f}"
            text_position = (int(box[0]), max(int(box[1]) - 10, 10))
            cv2.putText(
                transposed_image, 
                score_text, 
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )
        
        return transposed_image, len(filtered_boxes)
        
    except Exception as e:
        print(f"Error drawing bounding boxes: {str(e)}")
        try:
            # Fallback: return original image
            original_image = (
                image * torch.tensor(STD_NORMALIZATION)[:, None, None] + 
                torch.tensor(MEAN_NORMALIZATION)[:, None, None]
            )
            numpy_image = original_image.numpy()
            numpy_image = np.clip(numpy_image * 255, 0, 255).astype(np.uint8)
            transposed_image = np.transpose(numpy_image, (1, 2, 0))
            transposed_image = np.ascontiguousarray(transposed_image)
            return transposed_image, 0
        except:
            # Final fallback
            return np.zeros((224, 224, 3), dtype=np.uint8), 0

def predict_license_plate(model, processed_image):
    """
    Predict license plates from preprocessed image
    
    Args:
        model: Trained model
        processed_image: Preprocessed tensor image
        
    Returns:
        prediction: Dictionary with boxes, scores, labels
    """
    try:
        with torch.no_grad():
            model.eval()
            predictions = model(processed_image)
        
        # Get first prediction (batch size = 1)
        if predictions and len(predictions) > 0:
            return predictions[0]
        else:
            return None
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None
