"""
═══════════════════════════════════════════════════════════════
AERIAL OBJECT DETECTION & CLASSIFICATION - STREAMLIT APP
═══════════════════════════════════════════════════════════════
Professional app for Bird vs Drone classification and detection
Cloud-ready version with Google Drive integration
Color Palette: #450693, #8C00FF, #FF3F7F, #FFC400
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import torch
import io
import time
import os
import gdown
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION & CLOUD SETUP
# ═══════════════════════════════════════════════════════════════

# Google Drive file IDs - REPLACE THESE WITH YOUR ACTUAL FILE IDs
GDRIVE_CONFIG = {
    'classification_model': {
        'file_id': '1hItUzrDMO-nAzpLFpAHC4kBdReYS2c4N',
        'output': 'Models/best_model.h5'
    },
    'detection_model': {
        'file_id': '1K-pJncda11qY1JKi9TuOsPtqTiVBoM2Z',
        'output': 'Models/YOLOv8s.pt'
    }
}


def download_from_gdrive(file_id, output_path):
    """Download model from Google Drive"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Check if file already exists
        if os.path.exists(output_path):
            return True
        
        # Download from Google Drive
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, output_path, quiet=False)
        
        return os.path.exists(output_path)
    except Exception as e:
        st.error(f"Error downloading from Google Drive: {str(e)}")
        return False

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Aerial Object AI",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with improved text visibility
st.markdown("""
<style>
    /* Color Palette Variables */
    :root {
        --primary: #450693;
        --secondary: #8C00FF;
        --accent1: #FF3F7F;
        --accent2: #FFC400;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* Headers */
    h1 {
        color: #FFC400 !important;
        text-align: center;
        font-weight: 800 !important;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.8);
        padding: 20px 0;
    }
    
    h2 {
        color: #FFC400 !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #FF3F7F;
        padding-bottom: 10px;
        margin-top: 30px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
    }
    
    h3 {
        color: #FFC400 !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.6);
    }
    
    h4 {
        color: #8C00FF !important;
        font-weight: 600 !important;
    }
    
    /* General text visibility - WHITE */
    p, span, div, label {
        color: #FFFFFF !important;
    }
    
    /* Sidebar - WHITE TEXT */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #450693 0%, #8C00FF 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stRadio div {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    
    /* Radio buttons - WHITE */
    .stRadio > label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    .stRadio div[role="radiogroup"] label {
        color: #FFFFFF !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #8C00FF 0%, #FF3F7F 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 700 !important;
        font-size: 16px !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(140, 0, 255, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 63, 127, 0.6);
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(69, 6, 147, 0.3);
        border: 2px dashed #8C00FF;
        border-radius: 15px;
        padding: 20px;
    }

    [data-testid="stFileUploader"] label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* File uploader internal text - GREY */
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] div[data-testid="stMarkdownContainer"] {
        color: #808080 !important;
    }

    /* Uploaded file name - GREY */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"] {
        color: #808080 !important;
    }
            
    /* Info/Success/Warning Boxes */
    .stAlert {
        background: rgba(140, 0, 255, 0.15) !important;
        border-left: 4px solid #8C00FF !important;
        border-radius: 8px;
        color: white !important;
    }
    
    .stSuccess {
        background: rgba(0, 255, 0, 0.1) !important;
        border-left: 4px solid #00FF00 !important;
        color: white !important;
    }
    
    .stWarning {
        background: rgba(255, 196, 0, 0.15) !important;
        border-left: 4px solid #FFC400 !important;
        color: white !important;
    }
    
    .stError {
        background: rgba(255, 0, 0, 0.15) !important;
        border-left: 4px solid #FF0000 !important;
        color: white !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #FFC400 !important;
        font-size: 32px !important;
        font-weight: 800 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }
    
    /* Cards/Containers */
    .css-1r6slb0, .css-12oz5g7 {
        background: rgba(69, 6, 147, 0.15);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(140, 0, 255, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(69, 6, 147, 0.3);
        border-radius: 10px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #FFC400 !important;
        border-radius: 8px;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8C00FF 0%, #FF3F7F 100%);
        color: white !important;
    }
    
    /* Custom Cards */
    .result-card {
        background: linear-gradient(135deg, rgba(69, 6, 147, 0.3) 0%, rgba(140, 0, 255, 0.2) 100%);
        border: 2px solid #8C00FF;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(140, 0, 255, 0.3);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #8C00FF 0%, #FF3F7F 100%);
        color: white !important;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 28px !important;
        font-weight: 800 !important;
        margin: 20px 0;
        box-shadow: 0 6px 25px rgba(255, 63, 127, 0.6);
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .stat-box {
        background: rgba(255, 196, 0, 0.15);
        border-left: 4px solid #FFC400;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: white !important;
    }
    
    .stat-box b {
        color: #FFC400 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #FF3F7F !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #8C00FF 0%, #FF3F7F 100%);
    }
    
    /* Caption text */
    .stCaption {
        color: #B0B0B0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(140, 0, 255, 0.2) !important;
        color: #FFC400 !important;
        font-weight: 600 !important;
    }
    
    /* Slider */
    .stSlider label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_classification_model():
    """Load TensorFlow classification model from Google Drive"""
    try:
        model_path = GDRIVE_CONFIG['classification_model']['output']
        
        # Download if not exists
        if not os.path.exists(model_path):
            with st.spinner("Downloading classification model from Google Drive..."):
                success = download_from_gdrive(
                    GDRIVE_CONFIG['classification_model']['file_id'],
                    model_path
                )
                if not success:
                    return None, None
        
        # Check if file actually exists and has content
        if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
            st.error(f"Model file is missing or corrupted: {model_path}")
            return None, None
        
        # FIXED: Simplified loading strategy with proper error handling
        model = None
        loading_method = None
        
        # Method 1: Standard TensorFlow Keras (works with .h5 and compatible .keras files)
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            loading_method = "TensorFlow Keras"
            st.success("✅ Model loaded successfully")
        except Exception as e1:
            # Method 2: Try with tf.saved_model if it's a SavedModel format
            try:
                model = tf.saved_model.load(model_path)
                # Wrap in a callable interface
                model = model.signatures['serving_default']
                loading_method = "TensorFlow SavedModel"
                st.success("✅ Model loaded as SavedModel")
            except Exception as e2:
                st.error("❌ Failed to load model with both methods")
                st.error(f"Method 1 error: {str(e1)[:200]}")
                st.error(f"Method 2 error: {str(e2)[:200]}")
                
                st.error("### 🔴 SOLUTION REQUIRED")
                st.markdown("""
                **Your model format is incompatible. Please re-save your model:**
                
                ```python
                import tensorflow as tf
                
                # Load your trained model
                model = tf.keras.models.load_model('your_model.keras')
                
                # Save in H5 format (most compatible)
                model.save('best_model.h5', save_format='h5')
                ```
                
                Then upload `best_model.h5` to Google Drive and update the file ID.
                """)
                return None, None
        
        # Compile the model if it's a Keras model
        if hasattr(model, 'compile'):
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.00004),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        
        return model, f"InceptionV3 ({loading_method})"
            
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None, None
        
@st.cache_resource(show_spinner=False)
def load_detection_model():
    """Load YOLOv8 detection model from Google Drive"""
    try:
        model_path = GDRIVE_CONFIG['detection_model']['output']
        
        # Download if not exists
        if not os.path.exists(model_path):
            with st.spinner("Downloading detection model from Google Drive..."):
                success = download_from_gdrive(
                    GDRIVE_CONFIG['detection_model']['file_id'],
                    model_path
                )
                if not success:
                    return None, None
        
        # Load the model
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
        
        model = YOLO(model_path)
        return model, "YOLOv8s"
    except Exception as e:
        st.error(f"Error loading detection model: {str(e)}")
        return None, None

def preprocess_image_classification(image, target_size=(224, 224)):
    """Preprocess image for classification"""
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_classification(model, image):
    """Run classification prediction - returns Python float"""
    processed_img = preprocess_image_classification(image)
    prediction = model.predict(processed_img, verbose=0)[0][0]
    
    # Convert numpy float32 to Python float immediately
    prediction = float(prediction)
    
    # Binary classification: 0 = bird, 1 = drone
    class_name = "Drone" if prediction > 0.5 else "Bird"
    confidence = prediction if prediction > 0.5 else (1 - prediction)
    
    # Ensure it's Python float, not numpy
    return class_name, float(confidence * 100)

def predict_detection(model, image, conf_threshold=0.25):
    """Run YOLOv8 detection prediction"""
    img_array = np.array(image)
    results = model.predict(img_array, conf=conf_threshold, verbose=False)[0]
    
    img_with_boxes = img_array.copy()
    
    colors = {
        'bird': (0, 255, 0),
        'drone': (255, 0, 0)
    }
    
    detections = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = model.names[cls]
        
        detections.append({
            'class': name,
            'confidence': conf * 100,
            'bbox': (x1, y1, x2, y2)
        })
        
        color = colors[name]
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
        
        label = f"{name.upper()}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_with_boxes, (x1, y1 - label_h - 10), 
                     (x1 + label_w, y1), color, -1)
        
        cv2.putText(img_with_boxes, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img_with_boxes, detections

# ═══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <h1>AERIAL OBJECT AI SYSTEM</h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: white; font-size: 18px; margin-bottom: 30px;'>
        <b style='color: #FFC400;'>Advanced Deep Learning for Bird vs Drone Recognition</b><br>
        <span style='color: #FF3F7F; font-size: 14px; font-weight: 600;'>
            Classification • Detection • Real-Time Analysis
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### CONTROL PANEL")
        st.markdown("---")
        
        # Task selection
        task = st.radio(
            "Select Task:",
            ["Image Classification", "Object Detection"],
            help="Choose between classification or detection mode"
        )
        
        st.markdown("---")
        
        # Model info
        st.markdown("### MODEL INFO")
        if "Classification" in task:
            st.info("**Task:** Binary Classification\n\n**Classes:** Bird, Drone\n\n**Architecture:** Transfer Learning (ResNet50/InceptionV3)")
        else:
            st.info("**Task:** Object Detection\n\n**Model:** YOLOv8s\n\n**Input:** 640×640\n\n**Real-time:** Yes")
        
        st.markdown("---")
        
        # Confidence threshold for detection
        if "Detection" in task:
            conf_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.25,
                step=0.05,
                help="Minimum confidence for detections"
            )
        else:
            conf_threshold = None
        
        st.markdown("---")
        
        # About
        with st.expander("About This App"):
            st.markdown("""
            **Aerial Object AI** uses state-of-the-art deep learning to:
            
            - Classify images as **Bird** or **Drone**
            - Detect and localize objects in real-time
            - Provide confidence scores and analytics
            
            **Applications:**
            - Airport safety monitoring
            - Wildlife protection
            - Security surveillance
            - Airspace management
            """)
        
        # Credits
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #FFC400; font-size: 12px;'>
            <b>Built with Deep Learning</b><br>
            <span style='color: white;'>TensorFlow • PyTorch • YOLOv8 • Streamlit</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if "Classification" in task:
        # ═══════════════════════════════════════════════════════════
        # IMAGE CLASSIFICATION
        # ═══════════════════════════════════════════════════════════
        
        st.markdown("## Image Classification")
        st.markdown("<p style='color: white; font-size: 16px;'>Upload an image to classify it as <b style='color: #00FF00;'>Bird</b> or <b style='color: #FF3F7F;'>Drone</b></p>", unsafe_allow_html=True)
        
        # Load model only when needed
        if 'classification_model' not in st.session_state:
            with st.spinner("Loading classification model..."):
                model, model_name = load_classification_model()
                if model is not None:
                    st.session_state.classification_model = model
                    st.session_state.classification_model_name = model_name
        
        if 'classification_model' not in st.session_state:
            st.error("Failed to load model. Please check your Google Drive configuration.")
            st.info("Make sure you've set the correct Google Drive file IDs in the GDRIVE_CONFIG dictionary.")
            return
        
        st.success(f"Model loaded: **{st.session_state.classification_model_name}**")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG",
            key="classification_uploader"
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Display original image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True)
                st.caption(f"Size: {image.size[0]} × {image.size[1]} pixels")
            
            with col2:
                st.markdown("### AI Analysis")
                
                # Predict button
                if st.button("ANALYZE IMAGE", use_container_width=True):
                    with st.spinner("Analyzing..."):
                        start_time = time.time()
                        class_name, confidence = predict_classification(
                            st.session_state.classification_model, image
                        )
                        inference_time = (time.time() - start_time) * 1000
                    
                    # Ensure confidence is Python float
                    confidence = float(confidence)
                    
                    # Results
                    st.markdown(f"""
                    <div class='prediction-box'>
                        {class_name.upper()}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Confidence", f"{confidence:.2f}%")
                    with metric_col2:
                        st.metric("Time", f"{inference_time:.1f} ms")
                    
                    # Progress bar - Explicit Python float conversion
                    st.markdown("#### Confidence Score")
                    progress_value = confidence / 100.0
                    st.progress(float(progress_value))
                    
                    # Additional info
                    if confidence >= 90:
                        st.success("Very High Confidence - Excellent prediction")
                    elif confidence >= 75:
                        st.info("High Confidence - Good prediction")
                    elif confidence >= 60:
                        st.warning("Moderate Confidence - Review recommended")
                    else:
                        st.error("Low Confidence - Uncertain prediction")
                    
                    # Detailed stats
                    with st.expander("Detailed Statistics"):
                        other_class = "Bird" if class_name == "Drone" else "Drone"
                        other_conf = 100 - confidence
                        
                        st.markdown(f"""
                        <div class='stat-box'>
                            <b>{class_name}:</b> <span style='color: white;'>{confidence:.2f}%</span><br>
                            <b>{other_class}:</b> <span style='color: white;'>{other_conf:.2f}%</span><br>
                            <b>Model:</b> <span style='color: white;'>{st.session_state.classification_model_name}</span><br>
                            <b>Inference Time:</b> <span style='color: white;'>{inference_time:.2f} ms</span>
                        </div>
                        """, unsafe_allow_html=True)
    
    else:
        # ═══════════════════════════════════════════════════════════
        # OBJECT DETECTION
        # ═══════════════════════════════════════════════════════════
        
        st.markdown("## Object Detection")
        st.markdown("<p style='color: white; font-size: 16px;'>Upload an image to detect and localize <b style='color: #00FF00;'>Birds</b> and <b style='color: #FF3F7F;'>Drones</b></p>", unsafe_allow_html=True)
        
        # Load model only when needed
        if 'detection_model' not in st.session_state:
            with st.spinner("Loading detection model..."):
                model, model_name = load_detection_model()
                if model is not None:
                    st.session_state.detection_model = model
                    st.session_state.detection_model_name = model_name
        
        if 'detection_model' not in st.session_state:
            st.error("Failed to load model. Please check your Google Drive configuration.")
            st.info("Make sure you've set the correct Google Drive file IDs in the GDRIVE_CONFIG dictionary.")
            return
        
        st.success(f"Model loaded: **{st.session_state.detection_model_name}**")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG",
            key="detection_uploader"
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file).convert('RGB')
            
            # Display original image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True)
                st.caption(f"Size: {image.size[0]} × {image.size[1]} pixels")
            
            with col2:
                st.markdown("### Detection Results")
                
                # Detect button
                if st.button("DETECT OBJECTS", use_container_width=True):
                    with st.spinner("Detecting objects..."):
                        start_time = time.time()
                        img_with_boxes, detections = predict_detection(
                            st.session_state.detection_model, image, conf_threshold
                        )
                        inference_time = (time.time() - start_time) * 1000
                    
                    # Show detected image
                    st.image(img_with_boxes, use_container_width=True)
                    
                    # Detection statistics
                    num_birds = sum(1 for d in detections if d['class'] == 'bird')
                    num_drones = sum(1 for d in detections if d['class'] == 'drone')
                    total_detections = len(detections)
                    
                    # Metrics
                    st.markdown("### Detection Summary")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("Birds", num_birds)
                    with metric_col2:
                        st.metric("Drones", num_drones)
                    with metric_col3:
                        st.metric("Time", f"{inference_time:.0f} ms")
                    
                    if total_detections == 0:
                        st.warning("No objects detected. Try lowering the confidence threshold.")
                    else:
                        st.success(f"Detected **{total_detections}** object(s)")
                        
                        # Detailed detections
                        with st.expander("Detection Details"):
                            for idx, det in enumerate(detections, 1):
                                color = "#00FF00" if det['class'] == 'bird' else "#FF0000"
                                
                                st.markdown(f"""
                                <div class='stat-box'>
                                    <b>Detection #{idx}</b><br>
                                    <b>Class:</b> <span style='color: {color}; font-weight: bold;'>{det['class'].upper()}</span><br>
                                    <b>Confidence:</b> <span style='color: white;'>{det['confidence']:.2f}%</span><br>
                                    <b>Bounding Box:</b> <span style='color: white;'>{det['bbox']}</span>
                                </div>
                                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()






