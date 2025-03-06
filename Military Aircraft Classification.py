import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("keras_model.h5")

# Load labels
labels = ["F4 Fighter", "F16 Fighter", "Mi26 Helicopter", "Mirage Fighter", "Rafale Fighter"]

def preprocess_image(image):
    """Preprocess the image to fit the model's input requirements."""
    image = image.resize((224, 224))  # Resize image to match model input
    image = np.array(image) / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image):
    """Predict the class of the given image."""
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return labels[predicted_class], confidence

# Streamlit UI
st.title("Military Aircraft Classification")
st.write("Upload an image to classify the military aircraft type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("\nClassifying...")
    
    # Predict
    label, confidence = predict(image)
    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence:.2%}")
