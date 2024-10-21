import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import load_model
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Suppress warnings about deprecated functions
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Define age groups
age_group = {
    0: "YOUNG",
    1: "MIDDLE",
    2: "OLD"
}


# Load the model for prediction
l_model = load_model(os.path.join(
    os.getcwd(), "model/Age-Detector-Model-95.h5"))

# Preprocess test image


def preprocess_test_image(image):
    # Resize the image to the target size
    image_resized = cv2.resize(
        image, (180, 180), interpolation=cv2.INTER_LINEAR)

    # Normalize pixel values to the range [0, 1]
    image_normalized = image_resized / 255.0

    # Expand dimensions for batch processing
    image_expanded = np.expand_dims(image_normalized, axis=0)

    return image_expanded

# Predict age group


def predict_class(pred_model, image):
    processed_img = preprocess_test_image(image)
    pred_probas = pred_model.predict(processed_img)
    pred_class = pred_probas.argmax()
    return list(age_group.keys())[pred_class]

# Streamlit UI


def main():
    st.title("Age Detection APP")

    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read the image and display it
        image = plt.imread(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict the age group
        predicted_age_group = predict_class(l_model, image)

        # Display the predicted age group
        st.success(f"Predicted Age Group: {age_group[predicted_age_group]}")


if __name__ == "__main__":
    main()
