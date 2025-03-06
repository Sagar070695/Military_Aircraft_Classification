import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

# Load the trained model
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"

# Load model
model = load_model(MODEL_PATH, compile=False)

# Load class labels
with open(LABELS_PATH, "r") as file:
    class_names = [line.strip().split(" ", 1)[1] for line in file.readlines()]

# Streamlit App
st.title("Military Aircraft Classification")
st.write("Upload an image of a military aircraft and let the AI classify it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display result
    st.success(f"Prediction: **{class_name}**")
    st.info(f"Confidence Score: {confidence_score:.2f}")
