import streamlit as st
import sys
from pathlib import Path

# Add project root to path
root_dir = str(Path(__file__).parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

st.set_page_config(
    page_title="Vehicle License Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.welcome-section {
    background-color: #f8f9fa;
    padding: 2rem;
    border-radius: 10px;
    border-left: 5px solid #667eea;
    margin: 2rem 0;
    color: black !important;
}

.feature-card {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 1rem 0;
    border-left: 4px solid #667eea;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🚗 Vehicle License Plate Recognition</h1><p>Intelligent License Plate Recognition System</p></div>', unsafe_allow_html=True)

# Welcome section
st.markdown("""
<div class="welcome-section">
    <h2>👋 Welcome to the Vehicle License Plate Recognition System!</h2>
    <p>This application uses advanced AI technology to recognize license plates from images with high accuracy.</p>
</div>
""", unsafe_allow_html=True)

# Features
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3>📸 Image Recognition</h3>
        <p>Upload images to automatically detect license plates</p>
        <ul>
            <li>Supports JPG, PNG formats</li>
            <li>Automatic detection with bounding boxes</li>
            <li>Confidence score display</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3>🎯 High Accuracy</h3>
        <p>Uses specially trained Faster R-CNN model</p>
        <ul>
            <li>Fine-tuned for Vietnamese license plates</li>
            <li>Fast and accurate processing</li>
            <li>Supports multiple plate types</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Instructions
st.markdown("### 📋 How to Use")
st.markdown("""
1. **Select "Image Recognition"** from the sidebar
2. **Upload an image** containing vehicles for license plate detection
3. **Click "Detect License Plate"** to process
4. **View results** with bounding boxes and confidence scores

> 💡 **Tip**: Choose images with clear, unobstructed license plates for best results
""")

# Sidebar info
st.sidebar.success("📌 Select a page from the left menu to get started")
st.sidebar.info("🔧 Version: 1.0.0")
