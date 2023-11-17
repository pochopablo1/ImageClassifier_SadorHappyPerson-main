import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Title of the application
st.title('Image Classifier: Sad or Happy Person.')

# Load the model
model_path = 'models/happysadmodel.h5'
model = load_model(model_path)

# Widget to upload an image
upload_file = st.file_uploader("Upload image file (jpg)", type=["jpg"])

# Function to predict and display the image
def predict(image):
    # Check if the image is loaded successfully
    if image is not None:
        # Show the original image
        st.image(image, caption="Original Image", use_column_width=True)

        # Resize the image
        resize = tf.image.resize(image, (256, 256))

        # Normalize and expand dimensions for prediction
        input_image = np.expand_dims(resize / 255, 0)

        # Make the prediction
        yhat = model.predict(input_image)

        # Print the result
        if yhat[0] > 0.5:
            result = "HAPPY"
        else:
            result = "SAD"

        st.write(f"Prediction: {result}")
    else:
        st.write("Error: Unable to load the image.")

# Button to make the prediction
if upload_file is not None:
    # Read the image using OpenCV
    content = upload_file.getvalue()
    image = cv2.imdecode(np.frombuffer(content, np.uint8), 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predict(image_rgb)
