import streamlit as st
import cv2
import tensorflow as tf
from PIL import Image
import numpy as np

# @st.cache_data(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('model/ai_imageclassifier')
    return model

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

def make_prediction():
    model = load_model()
    img = st.file_uploader("Choose an image...", type="jpg")

    if img is not None:
        try:
            st.image(img, use_column_width=True)
            img_array = preprocess_image(img)
            prediction = model.predict(img_array)

            if prediction > 0.5:
                st.success("Image is Real")
            else:
                st.error("Image is AI Generated")

            

        except Exception as e:
            st.error(f"Error processing image: {e}")

 

