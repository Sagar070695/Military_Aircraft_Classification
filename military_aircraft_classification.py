import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("keras_model.h5")

# Load labels
label_file = "labels.txt"
with open(label_file, "r") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f.readlines()]

# Streamlit UI
st.title("Military Aircraft Classification")
st.write("Upload an image of a military aircraft, and the model will classify it.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize to match the model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_label = labels[np.argmax(predictions)]

    # Display result
    st.subheader(f"Prediction: {predicted_label}")
    st.bar_chart(predictions[0])
