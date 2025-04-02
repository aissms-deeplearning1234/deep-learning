import streamlit as sy
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from temsorflow.keras.models import Model
from get_data import get_data, read_params
from glob import glob
from urllib.parse import urlparse
import argparse
import os
import cv2
import base64
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras

# Page Configuration
st.set_page_config(page_title="Brain Tumor Classification", layout="wide")
st.markdown("""<style>
            .reportview-container { background: #f5f5f5; }
            .stButton > button { width: 100%; border-radius: 10px; }
            .stImage { text-align: center; }
            </style>
            """, unsafe_allow_html=True)

model = load_model("/models/trained.h5")

# Class Labels

classes = ['no_tumor', 'pituitary_tumor', 'meningioma_tumor', 'glioma_tumor']

def process_image(image):
    image = tf.image.resize(image, [255, 255])
    image /= 225.0
    return image

def generate_gradcam(image, model):
    """Generate Grad-CAM Visualization"""
    # Dummy implementation (replace with actual grad-CAM Logic)
    img_array = np.array(image)
    heatmap = np.uint8(255 * np.random.rand(*img_array.shape[:2]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return Image.fromarray(heatmap)

def download_report(pred_class, confidence):
    """Generate and return a Downloadable Report"""
    report_text = f"""
    Brain Tumor Classification Report
    ---------------------------------
    Prediction : {pred_class}
    Confidence : {confidence:.2f}%
    """
    
    b64 = base64.b64encode(report_text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="report.txt">Download Report</a>'
    return href

# Sidebar for Image Upload
st.sidebar.title("Upload MRI Scan")
uploaded_file = st.sidebar.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    if st.button("classify image"):
        input_tensor = preprocess_image(image)
        output = model.predict(input_tensor)
        pred_idx = np.argmax(output, axis=1)[0]
        confidence = output[0][pred_idx] * 100
        pred_class = classes[pred_idx]

        # Display Results
        col1, col2 = st.columns([2, 1])
        with col1:
            st.success(f"Prediction: {pred_class}")
            st.info(f"Confidence: {confidence:.2f}%")

        with col2:
            gradcam_image = generate_gradcam(image, model)
            st.image(gradcam_image, caption="Grad-CAM Visualization", use_column_width=True)

        # Download Report
        st.markdown(download_report(pred_class, confidence), unsafe_allow_html=True)