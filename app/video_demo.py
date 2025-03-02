import sys
import os
import warnings
# Add the project folder to environmental variable PYTHONPATH
sys.path.append(os.path.abspath(os.path.join("..")))
warnings.filterwarnings("ignore")

import torch
import cv2
import numpy as np
import time
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
    model.eval()
    model.to(DEVICE)
    return model

def preprocess_frame(frame):
    """
        Preprocess a video frame
    """
    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)

    # Transform frame
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_NORMALIZATION, std=STD_NORMALIZATION)
    ])

    # Apply transformation
    transformed_image = transform(pil_image)

    return transformed_image

def predict(model, image):
    """
        Make prediction on a single image
    """
    # Add batch dimension
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prediction = model(image)

    return {key: value.cpu() for key, value in prediction[0].items()}

def process_video(model, video_path, output_path=None, score_threshold=0.5, display_scale=1.0, save_video=True):
    """
        Process video and draw bounding boxes on frames
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer if output_path is specified
    if save_video and output_path:
        # Get the output directory
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Calculate the output dimensions
        out_width = int(frame_width * display_scale)
        out_height = int(frame_height * display_scale)
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    else:
        out = None
    
    # Initialize frame counter and timing
    frame_count = 0
    processing_times = []
    
    print(f"Processing video with {total_frames} frames...")
    
    # Process video frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Measure processing time
        start_time = time.time()
        
        # Preprocess frame
        processed_frame = preprocess_frame(frame)
        
        # Get predictions
        prediction = predict(model, processed_frame)
        
        # Get boxes and scores
        boxes = prediction["boxes"]
        scores = prediction["scores"]
        
        # Filter boxes based on score threshold
        high_score_indices = torch.where(scores > score_threshold)[0]
        filtered_boxes = boxes[high_score_indices]
        filtered_scores = scores[high_score_indices]
        
        # Original frame for display
        display_frame = frame.copy()
        
        # Draw bounding boxes on the frame
        for i, box in enumerate(filtered_boxes):
            # Convert box coordinates to integers
            box = box.round().int().numpy()
            
            # Points of rectangle
            start_point = (int(box[0] * frame_width / IMAGE_SIZE[0]), 
                           int(box[1] * frame_height / IMAGE_SIZE[1]))
            end_point = (int(box[2] * frame_width / IMAGE_SIZE[0]), 
                         int(box[3] * frame_height / IMAGE_SIZE[1]))
            
            # Color for rectangle - Azure color (BGR color for OpenCV)
            color = (0, 235, 255)
            # Thickness of rectangle
            thickness = 2
            
            # Draw rectangle
            cv2.rectangle(display_frame, start_point, end_point, color, thickness)
            
            # Display score
            score_text = f"{filtered_scores[i].item():.2f}"
            cv2.putText(display_frame, score_text, 
                       (start_point[0], start_point[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Resize frame for display if needed
        if display_scale != 1.0:
            display_frame = cv2.resize(display_frame, 
                                      (int(frame_width * display_scale), 
                                       int(frame_height * display_scale)), 
                                      interpolation=cv2.INTER_AREA)
        
        # Write to output video if requested
        if out:
            out.write(display_frame)
        
        # Display the frame
        cv2.imshow("Video Detection", display_frame)
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        # Update frame counter and print progress
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%) "
                  f"- {1/processing_time:.1f} FPS")
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if out:
        out.write(display_frame)
        out.release()
    cv2.destroyAllWindows()
    
    # Print performance statistics
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        avg_fps = 1 / avg_time if avg_time > 0 else 0
        print(f"\nProcessing complete.")
        print(f"Average processing time: {avg_time:.4f} seconds per frame")
        print(f"Average FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    # Load model
    model = load_model("../models/final/best_faster_rcnn_checkpoints.pt")

    # Display selection list
    INPUT_PATH = "./inputs/videos/"
    OUTPUT_PATH = "./outputs/videos/"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    print("Choose any video from the video folder:")
    print("-"*15)
    video_filenames = [f for f in os.listdir(INPUT_PATH) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_filenames:
        print("No video files found in the input directory.")
        sys.exit(1)
        
    for idx, video_filename in enumerate(video_filenames):
        print(f"{idx}. ", video_filename)
    print("-"*15)
    
    video_idx = int(input("Enter the index of video: ").strip())
    video_path = os.path.join(INPUT_PATH, video_filenames[video_idx])
    
    # Ask for additional parameters
    score_threshold = float(input("Enter detection score threshold (0.0-1.0, default 0.5): ") or "0.5")
    display_scale = float(input("Enter display scale factor (default 1.5): ") or "1.5")
    save_output = input("Save output video? (y/n, default y): ").lower() != "n"
    
    # Generate output path if saving
    output_path = None
    if save_output:
        output_filename = f"detected_{os.path.splitext(video_filenames[video_idx])[0]}.avi"
        output_path = os.path.join(OUTPUT_PATH, output_filename)
        print(f"Output will be saved to: {output_path}")
    
    # Process the video
    process_video(
        model=model,
        video_path=video_path,
        output_path=output_path,
        score_threshold=score_threshold,
        display_scale=display_scale,
        save_video=save_output
    )