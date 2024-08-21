import streamlit as st
from ui import preprocess_image
from tensorflow.keras.models import load_model

# title for project
st.title("Real vs AI Generated image classifier")

# loading model
model = load_model('app/my_model.h5')

# taking image as input and storing it into img
img = st.file_uploader("Choose an image...", type=["jpg", "webp", "png"])

if img is not None:
        try:
            st.image(img, use_column_width=True)
            img_array = preprocess_image(img)
            prediction = model.predict(img_array)

            if prediction > 0.41:
                st.success("It is a REAL image")
                
            else:
                st.error("It is an AI Generated image")

            

        except Exception as e:
            st.error(f"Error processing image: {e}")

else:
        print("Please upload an image")        
