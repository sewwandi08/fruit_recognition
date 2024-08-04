import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
from PIL import Image
import tensorflow as tf

# TensorFlow Model Prediction
def model_prediction(test_image, resize_dim):
    model_path = "my_model.keras"
    if not os.path.isfile(model_path):
        st.error(f"Model file '{model_path}' not found.")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(resize_dim, resize_dim))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # Return index of max element
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction", "Visualizations"])

# Main Page Logic
if app_mode == "Home":
    st.header("WELCOME TO THE FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "home_img.jpg"
    if os.path.isfile(image_path):
        st.image(image_path)
    else:
        st.error(f"Home image '{image_path}' not found.")
    
    st.write("""
### Discover the Power of AI in Identifying Fruits & Vegetables!
This tool quickly and accurately identifies different fruits and vegetables from images. Whether you're a tech enthusiast, a curious learner, or someone who simply wants to explore the capabilities of AI, this app is designed with you in mind.

**Key Features:**
- *Instant Image Classification*: Upload an image and get instant results.
- *User-Friendly Interface*: Seamless experience with intuitive design.
- *Educational Tool*: Learn about various fruits and vegetables.

Upload an image and let the app do the rest. Experience the future of image recognition today!
""")
    video_path = "veg.mp4"
    if os.path.isfile(video_path):
        st.video(video_path)
    else:
        st.error(f"Video '{video_path}' not found.")

elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of various fruits and vegetables.")
    st.code("Fruits: banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("Vegetables: cucumber, carrot, capsicum, onion, potato, lemon, tomato, radish, beetroot, cabbage, etc.")
    st.subheader("Dataset Structure")
    st.text("The dataset consists of three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

elif app_mode == "Prediction":
    st.header("Model Prediction")
    
    # Sidebar Inputs for Prediction
    st.sidebar.subheader("Prediction Settings")
    resize_dim = st.sidebar.slider("Resize Image to:", min_value=32, max_value=128, value=64, step=8)
    feedback = st.sidebar.text_input("Feedback")
    
    # Main Prediction Area
    with st.container():
        test_image = st.file_uploader("Choose an Image:")
        
        if st.button("Show Image"):
            if test_image is not None:
                st.image(test_image, width=300, use_column_width=True)
            else:
                st.error("Please upload an image first.")

        if st.button("Predict"):
            if test_image is not None:
                # Show progress and status updates
                with st.spinner('Model is making prediction...'):
                    time.sleep(2)  # Simulate time delay for model prediction
                    result_index = model_prediction(test_image, resize_dim)
                    if result_index is None:
                        st.error("Prediction could not be made.")
                    else:
                        progress = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress.progress(i + 1)

                        # Reading Labels
                        label_path = "labels.txt"
                        if not os.path.isfile(label_path):
                            st.error(f"Labels file '{label_path}' not found.")
                        else:
                            with open(label_path) as f:
                                content = f.readlines()
                            labels = [i.strip() for i in content]
                            st.success(f"Model predicts it's a {labels[result_index]}")

                        if feedback:
                            st.write(f"Thank you for your feedback: {feedback}")
            else:
                st.error("Please upload an image first.")

elif app_mode == "Visualizations":
    st.header("Data Visualizations")
    
    # Bar Plot using Matplotlib/Seaborn
    st.subheader("Example: Fruit and Vegetable Count")
    data = {
        'Category': ['Apple', 'Banana', 'Orange', 'Tomato'],
        'Count': [100, 150, 80, 50]
    }
    fig, ax = plt.subplots()
    sns.barplot(x='Category', y='Count', data=data, ax=ax)
    st.pyplot(fig)
    
    # Line Plot for Model Performance
    st.subheader("Example: Model Performance Over Time")
    epochs = np.arange(1, 31)  # 30 epochs, starting from 1 to 30
    accuracy = np.random.rand(len(epochs)) * 0.1 + 0.9  # Random accuracy data with matching length
    loss = np.random.rand(len(epochs)) * 0.1 + 0.2  # Random loss data with matching length

    fig, ax = plt.subplots()
    ax.plot(epochs, accuracy, label='Accuracy', marker='o')
    ax.plot(epochs, loss, label='Loss', marker='o')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metric')
    ax.legend()
    st.pyplot(fig)
