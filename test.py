import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Load your Keras model
model_path = 'my_model.keras'  # Replace with the actual path to your model file
model = load_model(model_path)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Take one sample for testing
sample_image = x_test[0]  # Use the first image from the test set
sample_label = y_test[0]  # Corresponding label

def preprocess_image(image_array):
    """
    Preprocesses the image array for prediction.

    Parameters:
    - image_array (np.ndarray): 2D numpy array of shape (height, width).

    Returns:
    - np.ndarray: Preprocessed image array.
    """
    # Normalize the image
    image_array = image_array / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    # Add channel dimension (for grayscale images, use axis=-1)
    image_array = np.expand_dims(image_array, axis=-1)
    return image_array

def predict_with_model(image_array):
    """
    Predicts the class of the input image array using the Keras model.

    Parameters:
    - image_array (np.ndarray): 2D numpy array of shape (height, width).

    Returns:
    - int: Predicted class label.
    """
    preprocessed_image = preprocess_image(image_array)
    # Perform prediction
    predictions = model.predict(preprocessed_image)
    # Get the predicted class
    predicted_class = np.argmax(predictions, axis=-1)[0]
    return predicted_class

# Test with the sample image
predicted_class = predict_with_model(sample_image)
print(f"True label: {sample_label}")
print(f"Predicted class: {predicted_class}")
