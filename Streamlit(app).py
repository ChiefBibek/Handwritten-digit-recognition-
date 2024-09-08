import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('my_model.keras')

# Image preprocessing function
def preprocess_image(image):
    # Convert the uploaded image to a grayscale numpy array
    image = Image.open(image).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image).astype('float32')  # Convert to NumPy array
    image = image.reshape(1, 28, 28,1)  # Reshape to (1, 28, 28, 1)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

# Title and description
st.title('Handwritten Digit Recognition')
img_folder = 'img dataset/'
img_name = ["0.png", "1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png"]

# Subheader and selection
st.subheader('Do you want to use a preset dataset or upload your own image?')

# Radio button for decision
decision = st.radio('Choose one of the following options:', ['Preset dataset', 'Upload your own image'], index=None)

# Logic based on the user's decision
if decision == 'Preset dataset':
    st.header('Select the image to be recognized')
    # Radio buttons for selecting an image from the preset dataset
    selected_img = st.radio('Choose an image:', img_name, index=None)
    # Display the selected image from the dataset
    if selected_img is not None:
        image_path = img_folder + selected_img
        st.image(image_path, width=300)
        img_for_prediction = preprocess_image(image_path)
elif decision == 'Upload your own image':
    # File uploader for user-uploaded image
    selected_img = st.file_uploader('Upload an image:', type=['png', 'jpg', 'jpeg'])
    if selected_img is not None:
        # Display the uploaded image
        st.image(selected_img, width=300)
        # Preprocess the uploaded image
        img_for_prediction = preprocess_image(selected_img)
    else:
        st.warning('Please upload an image.')

# Button for recognizing the selected image
if st.button('Recognize the selected image'):
    if selected_img:
        # Prediction
        img_prob = model.predict(img_for_prediction)
        img_pred = np.argmax(img_prob)
        
        # Display the recognized digit
        st.success(f'The digit in the image is: {img_pred}')

        # Display the processed image using matplotlib
        plt.imshow(img_for_prediction.reshape(28, 28), cmap='gray')
        st.pyplot(plt)
        
        # Debugging output
        st.write('Image Shape:', img_for_prediction.shape)
        # st.write('Image Data (First 10 Pixels):', img_for_prediction[0, :, :, 0].flatten()[:10])
        st.write('Model Prediction Probabilities:', img_prob)
    else:
        st.warning('Please select or upload an image to recognize.')
