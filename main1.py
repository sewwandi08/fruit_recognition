import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
import time
from PIL import Image
import tensorflow as tf

# Tensorflow Model Prediction
def model_prediction(test_image):
    model_path = "trained_model.h5"
    if not os.path.isfile(model_path):
        st.error(f"Model file '{model_path}' not found.")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About Project", "Prediction", "Visualizations"])

# Main Page
if app_mode == "Home":
    st.header("WELCOME TO THE FRUITS & VEGETABLES RECOGNITION SYSTEM")
    image_path = "home_img.jpg"
    if os.path.isfile(image_path):
        st.image(image_path)
    else:
        st.error(f"Home image '{image_path}' not found.")
    st.write("""
### 

Discover the power of artificial intelligence through our interactive web application. This tool allows you to quickly and accurately identify different fruits and vegetables from images. Whether you're a tech enthusiast, a curious learner, or someone who simply wants to explore the capabilities of AI, this app is designed with you in mind.

*Key Features:*
- *Instant Image Classification*: Upload an image of any fruit or vegetable, and our app will identify it within seconds.
- *User-Friendly Interface*: Enjoy a seamless experience with a clean and intuitive design, making it easy for anyone to use.
- *Educational Tool*: Learn more about various fruits and vegetables as you explore the classification results.

Simply upload your image, and let the app do the rest. Experience the future of image recognition today!
""")
    video_path = "veg.mp4"
    if os.path.isfile(video_path):
        st.video(video_path)
    else:
        st.error(f"Video '{video_path}' not found.")

elif app_mode == "About Project":
    st.header("About Project")
    st.subheader("About Dataset")
    st.text("This dataset contains images of the following food items:")
    st.code("fruits- banana, apple, pear, grapes, orange, kiwi, watermelon, pomegranate, pineapple, mango.")
    st.code("vegetables- cucumber, carrot, capsicum, onion, potato, lemon, tomato, raddish, beetroot, cabbage, lettuce, spinach, soy bean, cauliflower, bell pepper, chilli pepper, turnip, corn, sweetcorn, sweet potato, paprika, jalepeño, ginger, garlic, peas, eggplant.")
    st.subheader("Content")
    st.text("This dataset contains three folders:")
    st.text("1. train (100 images each)")
    st.text("2. test (10 images each)")
    st.text("3. validation (10 images each)")

elif app_mode == "Prediction":
    st.header("Model Prediction")
    
    # Input Widgets in Sidebar
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
                    result_index = model_prediction(test_image)
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
    epochs = np.arange(1, 32)
    accuracy = np.random.rand(10) * 0.1 + 0.9  # Random accuracy data
    loss = np.random.rand(10) * 0.1 + 0.2  # Random loss data
    
    fig, ax = plt.subplots()
    ax.plot(epochs, accuracy, label='Accuracy', marker='o')
    ax.plot(epochs, loss, label='Loss', marker='o')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metric')
    ax.legend()
    st.pyplot(fig)
