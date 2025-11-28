# **Aerial Object Classification & Detection using AI**

This project focuses on automating the classification and detection of aerial objects—such as **birds**, **drones**, **aircraft**, and **vehicles**—from drone and satellite imagery using deep learning. The system combines **image classification** and **object detection** to deliver accurate, real-time predictions suitable for surveillance, security, wildlife monitoring, and environmental applications.

---

## **Project Overview**

Aerial imagery is widely used in defense, disaster management, airspace monitoring, and smart city planning. However, manually analyzing large volumes of aerial images is time-consuming and error-prone.
This project leverages **Convolutional Neural Networks (CNNs)** and **state-of-the-art object detection models** to build a robust and scalable AI pipeline capable of:

* Classifying the type of object (bird, drone, airplane, etc.)
* Detecting and locating objects using bounding boxes
* Performing real-time inference through a Streamlit interface
* Supporting cloud and edge deployment

---

## **Key Features**

### ✔ **1. Aerial Object Classification**

Uses transfer learning with:

* **ResNet50**
* **CNN**
* **InceptionV3**

Predicts the class of the aerial object with high accuracy.

### ✔ **2. Object Detection System**

Implements advanced detection models:

* **YOLOv8s**

Detects object coordinates with bounding boxes and confidence scores.

### ✔ **3. Real-Time Web Interface (Streamlit)**

* Upload images
* View predictions instantly
* Visualize bounding boxes
* Cloud-compatible (Python 3.10)

### ✔ **4. Optimized for Deployment**

* Lightweight model versions for drones & edge devices
* Modular design for easy scaling
* TensorFlow & Ultralytics-based workflows

---

## **Project Structure**

```
Aerial-Object-AI/
│
├── App.py        # Web interface for cloud deployment
├── Object classification.ipynb # Classification training script
├── Object detection.ipynb     # YOLO/ training script
├── requirements.txt        # Dependencies for Python 3.10
└── README.md               # Project documentation
```

---

## **Technologies Used**

* **Python 3.10**
* **TensorFlow / Keras**
* **PyTorch (for detection models)**
* **Ultralytics YOLO**
* **OpenCV**
* **Streamlit**
* **NumPy / Pandas / Matplotlib**

---

## **How to Run the Project**

### **1. Clone the Repository**

```bash
git clone https://github.com/yourusername/aerial-object-ai.git
cd aerial-object-ai
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run Streamlit App**

```bash
streamlit run streamlit_app.py
```

### **4. For YOLO Training**

```bash
yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```

---

## **Evaluation Metrics**

* **Accuracy** (classification)
* **Precision, Recall, F1-Score**
* **Confusion Matrix**
* **mAP@50 and mAP@50–95** (object detection)
* **Inference Time** (real-time performance)

---

## **Applications**

* Border & drone surveillance
* Airspace management
* Wildlife tracking
* Disaster response
* Smart city monitoring
* Airport safety systems

---

## **Conclusion**

This project provides a powerful, real-time framework that integrates both **classification** and **detection** for aerial imagery. By combining multiple deep learning models and offering a simple, deployable interface, it enhances situational awareness and improves decision-making across various critical domains.


Just ask!
