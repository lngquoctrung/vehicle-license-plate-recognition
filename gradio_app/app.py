"""
Vehicle License Plate Recognition
Intelligent License Plate Recognition System with Gradio
"""

import gradio as gr
import sys
from pathlib import Path
import time

# Add project root to path
root_dir = str(Path(__file__).parent.parent.absolute())
dashboard_dir = str(Path(__file__).parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if dashboard_dir not in sys.path:
    sys.path.insert(0, dashboard_dir)

# Import utilities
from utils.model_utils import load_model, get_model_info
from utils.image_processing import preprocess_image, draw_detection_boxes, predict_license_plate

# Load model once at startup
model = load_model()
model_info = get_model_info()

def recognize_license_plate(image, score_threshold):
    """
    Main function to process image and detect license plates
    
    Args:
        image: PIL Image or numpy array
        score_threshold: Confidence threshold for detections
        
    Returns:
        result_image: Image with bounding boxes
        info_text: Detection information
        detection_details: Formatted details
    """
    if image is None:
        return None, "Please upload an image", ""
    
    if model is None:
        return None, "Model not loaded", ""
    
    try:
        # Start timing
        start_time = time.time()
        
        # Preprocess image
        processed_image, original_image = preprocess_image(image)
        
        if processed_image is None:
            return None, "Error preprocessing image", ""
        
        # Predict
        prediction = predict_license_plate(model, processed_image)
        
        # Draw bounding boxes
        result_image, num_detections = draw_detection_boxes(
            processed_image[0], 
            prediction, 
            score_threshold
        )
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Create info text
        if num_detections > 0:
            info_text = f"Detected {num_detections} license plate(s)"
            detection_details = f"""
                ### Detection Results

                - **Number of plates detected:** {num_detections}
                - **Processing time:** {processing_time:.3f}s
                - **Confidence threshold:** {score_threshold}

                The model has detected license plates in the image. 
                Cyan bounding boxes show the detected locations.
            """
        else:
            info_text = "No license plates detected"
            detection_details = f"""
                ### Detection Results

                - **Number of plates detected:** 0
                - **Processing time:** {processing_time:.3f}s
                - **Confidence threshold:** {score_threshold}

                Try lowering the confidence threshold or upload a different image.
            """
        
        return result_image, info_text, detection_details
        
    except Exception as e:
        return None, f"Error: {str(e)}", ""

def create_home_tab():
    """Create Home tab content based on project README"""
    with gr.Tab("Home"):
        gr.Markdown("""
            # Vehicle License Plate Recognition

            ## Intelligent License Plate Recognition System

            This application uses advanced AI technology to recognize license plates from images with high accuracy. Built using **Faster R-CNN** with fine-tuned backbone networks on the **License Plate** dataset from Roboflow.
        """)
        
        # Introduction Section
        gr.Markdown("""
            ### Introduction

            In recent years, AI has achieved outstanding accomplishments in the field of computer vision. In transportation, AI has many practical applications serving human interests such as predicting traffic jams, identifying vehicles participating in traffic, and recognizing license plates.

            This project researches and builds an application for license plate recognition using the **Fine Tuning** technique with pre-trained models such as **ResNet-50** and **MobileNet** as backbones for the **Faster R-CNN** model.
        """)
        
        # Key Features and Technologies
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("""
                    ### Key Features

                    - **Real-time Detection**: Upload images to automatically detect license plates
                    - **High Accuracy**: ~65% accuracy on validation dataset
                    - **Adjustable Threshold**: Customize confidence scores for detection
                    - **GPU Acceleration**: Optimized for CUDA-enabled devices
                    - **Bounding Box Visualization**: Clear detection results with scores
                    - **Fine-tuned Model**: Transfer learning from pre-trained backbones
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("""
                    ### Technologies Used

                    - **Deep Learning Framework**: PyTorch & TorchVision
                    - **Model Architecture**: Faster R-CNN
                    - **Backbone Network**: ResNet-50 (pre-trained)
                    - **Image Processing**: OpenCV, PIL
                    - **Dataset Source**: Roboflow License Plate Dataset
                    - **UI Framework**: Gradio
                    - **Optimization**: SGD with momentum
                """)
        
        # Model Information Section
        gr.Markdown("### Model Information")
        
        model_info_data = [
            ["Architecture", model_info.get("Model", "Faster R-CNN")],
            ["Backbone", model_info.get("Backbone", "ResNet-50")],
            ["Number of Classes", str(model_info.get("Classes", 2))],
            ["Device", str(model_info.get("Device", "CPU/GPU"))],
            ["Input Size", "224 × 224 pixels"],
            ["Framework", model_info.get("Framework", "PyTorch")],
        ]
        
        gr.Dataframe(
            value=model_info_data,
            headers=["Property", "Value"],
            datatype=["str", "str"],
            col_count=(2, "fixed"),
            interactive=False
        )
        
        # Training Configuration
        gr.Markdown("""
            ### Training Configuration

            The model was trained using the following hyperparameters and configuration:
        """)
        
        training_config = [
            ["Optimizer", "SGD (Stochastic Gradient Descent)"],
            ["Learning Rate", "0.0005"],
            ["Momentum", "0.9"],
            ["Weight Decay", "0.0005"],
            ["Batch Size", "16"],
            ["Total Epochs", "30 (trained 25)"],
            ["IoU Threshold", "0.5"],
            ["Image Size", "224 × 224"],
            ["Step Size", "10"],
            ["Gamma", "0.1"],
        ]
        
        gr.Dataframe(
            value=training_config,
            headers=["Parameter", "Value"],
            datatype=["str", "str"],
            col_count=(2, "fixed"),
            interactive=False
        )
        
        gr.Markdown("""
            The model uses the **Intersection over Union (IoU)** algorithm to evaluate the accuracy between predicted bounding boxes and ground truth boxes. IoU calculates the overlap area divided by the union area of two boxes.
        """)
        
        # How to Use Section
        gr.Markdown("""
            ### How to Use

            1. Navigate to the **Image Recognition** tab above
            2. Upload an image containing vehicles with license plates
            3. Adjust the **Confidence Threshold** slider (default: 0.5)
            - Higher values (0.7-0.9): Only show highly confident detections
            - Lower values (0.3-0.5): Show more potential detections
            4. Click **"Detect License Plates"** to process
            5. View results with cyan bounding boxes and confidence scores

            **Best Results**: The model works best with high-resolution images where license plates are clearly visible. For video processing, accuracy increases when vehicles move closer to the camera.
        """)
        
        # Performance & Results
        gr.Markdown("""
            ### Model Performance

            The model was trained on Kaggle with GPU P100 (16GB memory, 12-hour time limit):
        """)
        
        performance_data = [
            ["Training Epochs", "25 (out of 30)"],
            ["Validation Accuracy", "~65%"],
            ["Average Inference Time", "0.33 seconds per frame"],
            ["Average FPS", "~3.0 (video processing)"],
            ["Training Platform", "Kaggle GPU P100"],
        ]
        
        gr.Dataframe(
            value=performance_data,
            headers=["Metric", "Value"],
            datatype=["str", "str"],
            col_count=(2, "fixed"),
            interactive=False
        )
        
        gr.Markdown("""
            **Testing Results**: When tested with external images, the model shows quite good results. With video processing, the model achieves higher scores when vehicle license plates are closer and more visible to the camera.
        """)
        
        # Project Structure
        with gr.Accordion("Project Structure", open=False):
            gr.Markdown("""
                ```
                    ├── assets
                    │   ├── example.jpg
                    │   ├── example.xml
                    │   ├── iou.jpg
                    │   ├── lazy_loading.png
                    │   └── result.png
                    ├── demo
                    │   ├── image_demo.py
                    │   ├── inputs
                    │   │   ├── images
                    │   │   │   ├── sample_image_0.jpg
                    │   │   │   ├── sample_image_1.jpg
                    │   │   │   └── sample_image_2.jpg
                    │   │   └── videos
                    │   │       ├── sample_video_0.mp4
                    │   │       ├── sample_video_1.mp4
                    │   │       └── sample_video_2.mp4
                    │   └── video_demo.py
                    ├── Dockerfile
                    ├── gradio_app
                    │   ├── app.py
                    │   └── utils
                    │       ├── image_processing.py
                    │       ├── __init__.py
                    │       └── model_utils.py
                    ├── large_file_parts
                    │   ├── license-plate-project.zip.part1
                    │   ├── license-plate-project.zip.part10
                    │   ├── license-plate-project.zip.part11
                    │   ├── license-plate-project.zip.part12
                    │   ├── license-plate-project.zip.part13
                    │   ├── license-plate-project.zip.part14
                    │   └── ...
                    ├── models
                    │   ├── checkpoints
                    │   │   └── faster_rcnn_checkpoints.pt
                    │   └── final
                    │       └── best_faster_rcnn_checkpoints.pt
                    ├── notebooks
                    │   ├── experiments.ipynb
                    │   ├── exploration.ipynb
                    │   └── preprocessing.ipynb
                    ├── packages.txt
                    ├── README.md
                    ├── requirements.dev.txt
                    ├── requirements.txt
                    ├── runtime.txt
                    ├── src
                    │   ├── callbacks.py
                    │   ├── config.py
                    │   ├── data_preprocessing.py
                    │   ├── dataset.py
                    │   ├── evaluate.py
                    │   ├── __init__.py
                    │   ├── metrics.py
                    │   ├── model.py
                    │   ├── train.py
                    │   ├── utils.py
                    │   └── visualization.py
                    ├── streamlit_app
                    │   ├── Home.py
                    │   ├── pages
                    │   │   └── Image_Recognition.py
                    │   └── utils
                    │       ├── image_processing.py
                    │       ├── __init__.py
                    │       └── model_utils.py
                    └── tests
                        ├── test_data_preprocessing.py
                        ├── test_dataset.py
                        ├── test_model.py
                        └── test_utils.py
                ```

                **Key Directories**:
                - `strealit_app/`: Streamlit web application interface
                - `gradio_app/`: Gradio web application interface
                - `demo/`: Command-line testing scripts for images and videos
                - `models/`: Trained model weights and checkpoints
                - `src/`: Core modules (Dataset, Model, Training, Metrics)
                - `notebooks/`: Data preprocessing, exploration, and training notebooks
                - `tests/`: Unit tests using unittest library
            """)
        
        # Libraries Used
        with gr.Accordion("Libraries & Dependencies", open=False):
            gr.Markdown("""
                This project uses several powerful libraries for deep learning and computer vision:

                - **PyTorch**: Deep learning framework providing classes, metrics, and optimizers
                - **TorchVision**: Pre-trained models and computer vision utilities
                - **OpenCV**: Specialized tool for image processing (loading, resizing, drawing)
                - **NumPy**: Numerical computing and array operations
                - **Pillow (PIL)**: Image loading and manipulation
                - **Matplotlib**: Data visualization and plotting
                - **Gradio** or **Streamlit**: Web interface for ML model deployment

                PyTorch's **Dataset** and **DataLoader** classes help build efficient data pipelines, avoiding memory overflow issues when handling large image datasets.

                See [requirements.txt](https://github.com/lngquoctrung/vehicle-license-plate-recognition/blob/main/requirements.txt) for specific version details.
            """)
        
        # Installation & Usage
        with gr.Accordion("Installation & Setup", open=False):
            gr.Markdown("""
                ### Prerequisites
                Check if your device supports GPU acceleration:

                ```
                # Check GPU information
                nvidia-smi

                # Check CUDA version
                nvcc --version
                ```

                ### Installation Steps

                ```
                # Clone the repository
                git clone https://github.com/lngquoctrung/vehicle-license-plate-recognition.git

                cd vehicle-license-plate-recognition

                # Create virtual environment
                python3 -m venv .venv

                # Activate environment
                source .venv/bin/activate  # Linux/Mac
                # .venv\\Scripts\\activate  # Windows

                # Install dependencies
                pip install -r requirements.dev.txt
                ```

                ### Running the Application

                **Web Interface (Gradio)**:
                ```
                python app/main.py
                ```

                **Command-line Testing**:
                ```
                # Test with images
                cd demo
                python3 image_demo.py

                # Test with videos
                python3 video_demo.py
                ```

                The command-line interface allows you to select images/videos from the `inputs/` folder, adjust confidence thresholds, display scales, and optionally save output videos.
            """)
        
        # References
        with gr.Accordion("References & Resources", open=False):
            gr.Markdown("""
                **Dataset**:
                - [License Plate Project - Roboflow Universe](https://universe.roboflow.com/test-vaxvp/license-plate-project-adaad)

                **Model & Algorithms**:
                - [Faster R-CNN Explained for Object Detection Tasks](https://www.digitalocean.com/community/tutorials/faster-r-cnn-explained-object-detection)
                - [Intersection over Union (IoU) for Object Detection](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)

                **Framework Documentation**:
                - [PyTorch Official Documentation](https://pytorch.org/)
                - [Gradio Documentation](https://www.gradio.app/)

                **GPU Setup Guides**:
                - [Installing PyTorch with GPU Support on Ubuntu](https://medium.com/@jeanpierre_lv/installing-pytorch-with-gpu-support-on-ubuntu-a-step-by-step-guide-38dcf3f8f266)
                - [Install PyTorch GPU on Windows - Complete Guide](https://www.lavivienpost.com/install-pytorch-gpu-on-windows-complete-guide/)
            """)
        
        # Footer
        gr.Markdown("""
            ---
            <p style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem'>
                Powered by Faster R-CNN & PyTorch | Built with Gradio | Dataset from Roboflow
            </p>
""")

def create_recognition_tab():
    """Create Image Recognition tab"""
    with gr.Tab("Image Recognition"):
        gr.Markdown("""
            ## License Plate Detection from Images

            Select an image containing vehicles for license plate detection.  
            **Supports:** JPG, JPEG, PNG
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    sources=["upload", "clipboard"]
                )
                
                score_threshold = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Only show detections with confidence > this threshold"
                )
                
                detect_btn = gr.Button("Detect License Plates", variant="primary")
                clear_btn = gr.Button("Clear")
                
                # Model info in sidebar
                gr.Markdown("### Model Information")
                model_info_text = "\n".join([f"**{k}:** {v}" for k, v in model_info.items()])
                gr.Markdown(model_info_text)
            
            with gr.Column(scale=2):
                # Output components
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1
                )
                
                output_image = gr.Image(
                    label="Detection Result",
                    type="numpy"
                )
                
                detection_info = gr.Markdown()
        
        # Event handlers
        detect_btn.click(
            fn=recognize_license_plate,
            inputs=[input_image, score_threshold],
            outputs=[output_image, status_text, detection_info]
        )
        
        clear_btn.click(
            fn=lambda: (None, None, "", ""),
            inputs=None,
            outputs=[input_image, output_image, status_text, detection_info]
        )

def main():
    """Main application entry point"""
    
    # Custom CSS
    custom_css = """
    .gradio-container {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .gr-button-primary {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
    }
    """
    
    # Create Gradio Blocks interface
    with gr.Blocks(
        title="Vehicle License Plate Recognition",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:
        
        gr.Markdown("""
            <h1 style='text-align: center; margin-bottom: 1rem'>
                Vehicle License Plate Recognition
            </h1>
            <p style='text-align: center; color: #666; margin-bottom: 2rem'>
                Intelligent License Plate Recognition System
            </p>
        """)
        
        # Create tabs
        create_home_tab()
        create_recognition_tab()
        
        gr.Markdown("""
            ---
            <p style='text-align: center; color: #999; font-size: 0.9em'>
                Powered by Faster R-CNN | Built with Gradio
            </p>
        """)
    
    return demo

if __name__ == "__main__":
    app = main()
    app.launch(
        server_name="0.0.0.0",
        server_port=7001,
        share=False,
        show_error=True
    )
