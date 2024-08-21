import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# preprocessing image for prediction 
def preprocess_image(image):
    # Convert BytesIO to numpy array
    img_array = np.array(Image.open(image).convert("RGB"))
    # Resize the image
    img_array = tf.image.resize(img_array, (32, 32))
    # Normalize the image
    img_array = img_array / 255.0
    # Expand dimensions to match model input shape
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
     