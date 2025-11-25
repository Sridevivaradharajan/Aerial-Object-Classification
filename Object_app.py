"""
═══════════════════════════════════════════════════════════════
AERIAL OBJECT DETECTION & CLASSIFICATION - STREAMLIT APP
═══════════════════════════════════════════════════════════════
Professional app for Bird vs Drone classification and detection
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

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION & STYLING
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
    
    /* General text visibility */
    p, span, div {
        color: #E0E0E0 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #450693 0%, #8C00FF 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-weight: 600 !important;
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
        color: #FFC400 !important;
        font-weight: 600 !important;
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
        color: white !important;
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
        color: white !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_classification_model():
    """Load TensorFlow classification model"""
    try:
        # Try loading the best model from Streamlit app folder
        model = tf.keras.models.load_model(r'D:\Profession\Labmentix\Aerial Object classification\Object Classification\results\streamlit_app\best_model.keras')
        return model, "Best Model (Transfer Learning)"
    except:
        try:
            # Fallback to ResNet50
            model = tf.keras.models.load_model('models/resnet50_final.keras')
            return model, "ResNet50"
        except:
            try:
                # Fallback to Custom CNN
                model = tf.keras.models.load_model('models/custom_cnn_final.keras')
                return model, "Custom CNN"
            except:
                st.error("⚠️ No classification model found! Please ensure model files are in the correct directory.")
                return None, None

@st.cache_resource(show_spinner=False)
def load_detection_model():
    """Load YOLOv8 detection model"""
    try:
        # Fix PyTorch compatibility for YOLOv8
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
        
        # Try loading YOLOv8s model first
        model = YOLO('best_bird_drone.pt')
        return model, "YOLOv8s"
    except:
        try:
            # Fallback to alternative path
            model = YOLO(r'D:\Profession\Labmentix\Aerial Object classification\Object Detection\best_bird_drone.pt')
            return model, "YOLOv8s"
        except:
            st.error("⚠️ No detection model found! Please ensure YOLOv8 model file exists.")
            return None, None

def preprocess_image_classification(image, target_size=(224, 224)):
    """Preprocess image for classification"""
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_classification(model, image):
    """Run classification prediction"""
    processed_img = preprocess_image_classification(image)
    prediction = model.predict(processed_img, verbose=0)[0][0]
    
    # Binary classification: 0 = bird, 1 = drone
    class_name = "Drone" if prediction > 0.5 else "Bird"
    confidence = prediction if prediction > 0.5 else (1 - prediction)
    
    # Convert to Python float to avoid numpy float32 issues
    return class_name, float(confidence * 100)

def predict_detection(model, image, conf_threshold=0.25):
    """Run YOLOv8 detection prediction"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Run prediction
    results = model.predict(img_array, conf=conf_threshold, verbose=False)[0]
    
    # Draw bounding boxes
    img_with_boxes = img_array.copy()
    
    colors = {
        'bird': (0, 255, 0),   # Green
        'drone': (255, 0, 0)    # Red
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
        
        # Draw bounding box
        color = colors[name]
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        label = f"{name.upper()}: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_with_boxes, (x1, y1 - label_h - 10), 
                     (x1 + label_w, y1), color, -1)
        
        # Draw label text
        cv2.putText(img_with_boxes, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return img_with_boxes, detections

# ═══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown("""
    <h1>🦅 AERIAL OBJECT AI SYSTEM 🚁</h1>
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
        st.markdown("### ⚙️ CONTROL PANEL")
        st.markdown("---")
        
        # Task selection
        task = st.radio(
            "🎯 Select Task:",
            ["📸 Image Classification", "🔍 Object Detection"],
            help="Choose between classification or detection mode"
        )
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 📊 MODEL INFO")
        if "📸" in task:
            st.info("**Task:** Binary Classification\n\n**Classes:** Bird, Drone\n\n**Architecture:** Transfer Learning (ResNet50/InceptionV3)")
        else:
            st.info("**Task:** Object Detection\n\n**Model:** YOLOv8s\n\n**Input:** 640×640\n\n**Real-time:** ✅")
        
        st.markdown("---")
        
        # Confidence threshold for detection
        if "🔍" in task:
            conf_threshold = st.slider(
                "🎚️ Confidence Threshold",
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
        with st.expander("ℹ️ About This App"):
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
            <b>Built with ❤️ using</b><br>
            <span style='color: white;'>TensorFlow • PyTorch • YOLOv8 • Streamlit</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if "📸" in task:
        # ═══════════════════════════════════════════════════════════
        # IMAGE CLASSIFICATION
        # ═══════════════════════════════════════════════════════════
        
        st.markdown("## 📸 Image Classification")
        st.markdown("<p style='color: white; font-size: 16px;'>Upload an image to classify it as <b style='color: #00FF00;'>Bird</b> or <b style='color: #FF3F7F;'>Drone</b></p>", unsafe_allow_html=True)
        
        # Load model only when needed
        if 'classification_model' not in st.session_state:
            with st.spinner("🔄 Loading classification model..."):
                model, model_name = load_classification_model()
                if model is not None:
                    st.session_state.classification_model = model
                    st.session_state.classification_model_name = model_name
        
        if 'classification_model' not in st.session_state:
            st.error("❌ Failed to load model. Please check model files.")
            return
        
        st.success(f"✅ Model loaded: **{st.session_state.classification_model_name}**")
        
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
                st.markdown("### 🖼️ Original Image")
                st.image(image, use_container_width=True)
                st.caption(f"📐 Size: {image.size[0]} × {image.size[1]} pixels")
            
            with col2:
                st.markdown("### 🤖 AI Analysis")
                
                # Predict button
                if st.button("🚀 ANALYZE IMAGE", use_container_width=True):
                    with st.spinner("🔍 Analyzing..."):
                        start_time = time.time()
                        class_name, confidence = predict_classification(
                            st.session_state.classification_model, image
                        )
                        inference_time = (time.time() - start_time) * 1000
                    
                    # Results
                    st.markdown(f"""
                    <div class='prediction-box'>
                        {class_name.upper()}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("🎯 Confidence", f"{confidence:.2f}%")
                    with metric_col2:
                        st.metric("⚡ Time", f"{inference_time:.1f} ms")
                    
                    # Progress bar - Fixed: convert to float
                    st.markdown("#### 📊 Confidence Score")
                    st.progress(float(confidence / 100))
                    
                    # Additional info
                    if confidence >= 90:
                        st.success("🟢 **Very High Confidence** - Excellent prediction")
                    elif confidence >= 75:
                        st.info("🔵 **High Confidence** - Good prediction")
                    elif confidence >= 60:
                        st.warning("🟡 **Moderate Confidence** - Review recommended")
                    else:
                        st.error("🔴 **Low Confidence** - Uncertain prediction")
                    
                    # Detailed stats
                    with st.expander("📈 Detailed Statistics"):
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
        
        st.markdown("## 🔍 Object Detection")
        st.markdown("<p style='color: white; font-size: 16px;'>Upload an image to detect and localize <b style='color: #00FF00;'>Birds</b> and <b style='color: #FF3F7F;'>Drones</b></p>", unsafe_allow_html=True)
        
        # Load model only when needed
        if 'detection_model' not in st.session_state:
            with st.spinner("🔄 Loading detection model..."):
                model, model_name = load_detection_model()
                if model is not None:
                    st.session_state.detection_model = model
                    st.session_state.detection_model_name = model_name
        
        if 'detection_model' not in st.session_state:
            st.error("❌ Failed to load model. Please check model files.")
            return
        
        st.success(f"✅ Model loaded: **{st.session_state.detection_model_name}**")
        
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
                st.markdown("### 🖼️ Original Image")
                st.image(image, use_container_width=True)
                st.caption(f"📐 Size: {image.size[0]} × {image.size[1]} pixels")
            
            with col2:
                st.markdown("### 🎯 Detection Results")
                
                # Detect button
                if st.button("🔍 DETECT OBJECTS", use_container_width=True):
                    with st.spinner("🔍 Detecting objects..."):
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
                    st.markdown("### 📊 Detection Summary")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("🦅 Birds", num_birds)
                    with metric_col2:
                        st.metric("🚁 Drones", num_drones)
                    with metric_col3:
                        st.metric("⚡ Time", f"{inference_time:.0f} ms")
                    
                    if total_detections == 0:
                        st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")
                    else:
                        st.success(f"✅ Detected **{total_detections}** object(s)")
                        
                        # Detailed detections
                        with st.expander("📋 Detection Details"):
                            for idx, det in enumerate(detections, 1):
                                icon = "🦅" if det['class'] == 'bird' else "🚁"
                                color = "#00FF00" if det['class'] == 'bird' else "#FF0000"
                                
                                st.markdown(f"""
                                <div class='stat-box'>
                                    <b>{icon} Detection #{idx}</b><br>
                                    <b>Class:</b> <span style='color: {color}; font-weight: bold;'>{det['class'].upper()}</span><br>
                                    <b>Confidence:</b> <span style='color: white;'>{det['confidence']:.2f}%</span><br>
                                    <b>Bounding Box:</b> <span style='color: white;'>{det['bbox']}</span>
                                </div>
                                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()