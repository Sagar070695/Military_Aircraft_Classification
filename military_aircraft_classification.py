import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
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
    img = np.array(image)  # Convert to NumPy array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Ensure correct colour format
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = img / 255.0  # Normalise
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Prediction and Visualisation
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_label = labels[np.argmax(predictions)]
    confidence_scores = predictions[0]

    # Display result
    st.subheader(f"Prediction: {predicted_label}")

    # Visualise probabilities using matplotlib
    fig, ax = plt.subplots()
    ax.barh(labels, confidence_scores, color="blue")
    ax.set_xlabel("Confidence")
    ax.set_title("Prediction Probabilities")
    st.pyplot(fig)
