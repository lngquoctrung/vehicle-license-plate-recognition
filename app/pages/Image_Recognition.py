import streamlit as st
import sys
from pathlib import Path
import time

# Add project root to path
root_dir = str(Path(__file__).parent.parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import utilities
from utils.model_utils import load_model, get_model_info
from utils.image_processing import preprocess_image, draw_detection_boxes, predict_license_plate

# Page config
st.title("📸 License Plate Recognition from Images")

# CSS for this page
st.markdown("""
<style>
.upload-section {
    background-color: #f8f9fa;
    padding: 2rem;
    border-radius: 10px;
    border-left: 5px solid #667eea;
    margin: 1rem 0;
    color: black !important;
}

.result-section {
    background-color: #e7f3ff;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 1rem 0;
}

.stats-box {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}

.detection-info {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Sidebar with model information
with st.sidebar:
    st.markdown("### 🤖 Model Information")
    model_info = get_model_info()
    for key, value in model_info.items():
        st.markdown(f"**{key}:** {value}")
    
    st.markdown("### ⚙️ Settings")
    score_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1,
        help="Only show detections with confidence > this threshold"
    )

# Main content
st.markdown("""
<div class="upload-section">
    <h3>📁 Upload Image</h3>
    <p>Select an image containing vehicles for license plate detection. Supports: JPG, JPEG, PNG</p>
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png'],
    help="Upload an image containing vehicles for license plate detection"
)

if uploaded_file is not None:
    # Display original image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🖼️ Original Image")
        st.image(uploaded_file, caption="Uploaded Image", width="stretch")
    
    # Load model
    with st.spinner("🔄 Loading model..."):
        model = load_model()
    
    if model is not None:
        # Predict button
        if st.button("🎯 Detect License Plates", type="primary", width="stretch"):
            with st.spinner("🔍 Processing and detecting..."):
                # Preprocess image
                processed_image, original_pil = preprocess_image(uploaded_file)
                
                if processed_image is not None:
                    # Predict
                    start_time = time.time()
                    prediction = predict_license_plate(model, processed_image)
                    processing_time = time.time() - start_time
                    
                    if prediction is not None:
                        # Draw bounding boxes
                        result_image, num_detections = draw_detection_boxes(
                            processed_image.squeeze(0), 
                            prediction, 
                            score_threshold
                        )
                        
                        with col2:
                            st.markdown("### 🎯 Detection Results")
                            st.image(result_image, caption="Results with bounding boxes", width="stretch")
                        
                        # Statistics
                        st.markdown("""
                        <div class="result-section">
                            <h3>📊 Detection Statistics</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col3, col4, col5 = st.columns(3)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="stats-box">
                                <h4 style="color: #667eea;">🎯 License Plates</h4>
                                <h2 style="color: #2e7d32;">{num_detections}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                            <div class="stats-box">
                                <h4 style="color: #667eea;">⚡ Processing Time</h4>
                                <h2 style="color: #2e7d32;">{processing_time:.2f}s</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col5:
                            st.markdown(f"""
                            <div class="stats-box">
                                <h4 style="color: #667eea;">🎚️ Threshold</h4>
                                <h2 style="color: #2e7d32;">{score_threshold}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detection details
                        if num_detections > 0:
                            st.markdown("""
                            <div class="detection-info">
                                <h4>✅ Detection Successful!</h4>
                                <p>The model has detected license plates in the image. Cyan bounding boxes show the detected locations.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Individual detection details
                            if prediction["boxes"].shape[0] > 0:
                                st.markdown("### 📋 Detection Details")
                                
                                boxes = prediction["boxes"]
                                scores = prediction["scores"]
                                
                                for i, (box, score) in enumerate(zip(boxes, scores)):
                                    if score > score_threshold:
                                        x1, y1, x2, y2 = box.int().tolist()
                                        width = x2 - x1
                                        height = y2 - y1
                                        
                                        st.markdown(f"""
                                        **Detection {i+1}:**
                                        - **Confidence:** {score:.3f}
                                        - **Position:** ({x1}, {y1}) → ({x2}, {y2})
                                        - **Size:** {width} × {height} pixels
                                        """)
                        else:
                            st.warning("⚠️ No license plates detected with current threshold. Try lowering the threshold or choose a different image.")
                    
                    else:
                        st.error("❌ Unable to perform prediction. Please try again.")
                else:
                    st.error("❌ Unable to process image. Please choose a different image.")
    else:
        st.error("❌ Unable to load model. Please check checkpoint file.")

else:
    # Instructions when no image is uploaded
    st.markdown("""
    ### 📖 How to Use
    
    1. **Upload Image:** Click "Browse files" to select an image from your computer
    2. **Wait for Processing:** The model will load automatically
    3. **Detect:** Click "Detect License Plates" to start processing
    4. **View Results:** The result image will display with cyan bounding boxes
    
    ### 💡 Tips for Best Results:
    - Choose images with clear license plates
    - Ensure plates are not obstructed
    - Use adequate lighting (not too dark or bright)
    - Avoid extremely angled shots
    """)

# Footer
st.markdown("---")
st.markdown("*Developed by Vehicle License Plate Recognition System*")
