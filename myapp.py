import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Custom function to load the model
@st.cache_resource
def load_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (4, 4), 1, activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, (4, 4), 1, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, (4, 4), 1, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


# Function to resize and predict
def predict(model, image):
    resize = tf.image.resize(image, (32, 32))
    print("Resized Image Shape:", resize.shape)  # Debugging: Print resized image shape
    y_pred = model.predict(np.expand_dims(resize / 255, 0))
    if y_pred < 0.60:
        return st.subheader("REAL IMAGE")
    else:
        return st.subheader("AI GENERATED IMAGE")


def main():
    st.title("Image Classifier")

    st.subheader("***Upload an image for classification***")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the model
        model = load_model()
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="*Uploaded Image*", use_column_width=True)
        # Perform prediction
        prediction = predict(model, np.array(image))
        st.write(f"Predicted class: {prediction}")


if __name__ == "__main__":
    main()
