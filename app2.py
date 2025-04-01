import streamlit as st
import requests
from PIL import Image, ImageOps
import io
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.title("MNIST Digit Classifier API")
st.write("Draw a digit or upload an image and get the predicted digit from the API.")

# Create a drawing canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  
    stroke_width=10,
    stroke_color="#000000",  
    background_color="#FFFFFF",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    image = Image.fromarray((canvas_result.image_data[:, :, :3]).astype('uint8'))
    st.image(image, caption="Your Drawing", width=150)
    
    if st.button("Classify Drawing"):
        with st.spinner("Classifying..."):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            
            response = requests.post("http://127.0.0.1:5000/predict", files={"image": buf.getvalue()})

            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Digit: {result['digit']}")
                st.write(f"Confidence: {result['confidence']:.2f}")
            else:
                st.error("Error in API response")

st.markdown("---")
st.write("DIGIT RECOGNITION")