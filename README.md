# cam.py


"""
This script captures video from the webcam, detects a central region of interest (ROI), and uses a pre-trained 
Convolutional Neural Network (CNN) model to recognize handwritten digits within that ROI in real-time.

Modules:
- cv2: OpenCV module for video capture and image processing.
- numpy: Module for numerical operations.
- tensorflow.keras.models: Module to load the pre-trained Keras model.

Workflow:
1. Load the pre-trained CNN model from 'CNN_model.keras'.
2. Initialize the webcam for video capture.
3. Define the parameters for a central bounding box within the video frame.
4. Continuously capture frames from the webcam.
5. For each frame:
    a. Resize the frame to the standard size.
    b. Draw a central bounding box on the frame.
    c. Crop the region inside the bounding box.
    d. Preprocess the cropped region to match the input requirements of the CNN model.
    e. Run the CNN model to predict the digit in the cropped region.
    f. Display the detected digit and confidence score on the frame.
6. Display the annotated frame in a window.
7. Exit the loop and close the window when 'q' is pressed.

Functions:
- preprocess_image(image): Preprocesses the input image to match the model's input requirements.

Usage:
- Run the script to start the webcam and see real-time digit recognition within the central bounding box.
- Press 'q' to exit the application.
"""
### Convolutional Neural Network (CNN) Model

The Convolutional Neural Network (CNN) model used in this script is a deep learning model specifically designed for image recognition tasks. CNNs are particularly effective for this purpose due to their ability to capture spatial hierarchies in images through the use of convolutional layers.

#### Model Architecture

The pre-trained CNN model loaded from 'CNN_model.keras' typically consists of the following layers:

1. **Convolutional Layers**: These layers apply a set of filters to the input image, producing feature maps that highlight various aspects of the image, such as edges, textures, and patterns.
2. **Pooling Layers**: These layers reduce the spatial dimensions of the feature maps, retaining the most important information while reducing computational complexity.
3. **Fully Connected Layers**: These layers interpret the features extracted by the convolutional and pooling layers and make the final classification decision.
4. **Output Layer**: This layer produces the final prediction, which in this case is the recognized digit.

#### Training

The CNN model was trained on a large dataset of handwritten digits, such as the MNIST dataset, which contains 60,000 training images and 10,000 test images of digits from 0 to 9. The training process involves adjusting the weights of the network to minimize the error in digit recognition.

#### Preprocessing

Before feeding an image into the CNN model, it must be preprocessed to match the model's input requirements. This typically involves resizing the image to a fixed size (e.g., 28x28 pixels), normalizing pixel values, and reshaping the image to fit the input shape expected by the model.

### Additional Information

- **Accuracy**: The pre-trained CNN model used in this script is highly accurate, often achieving over 99% accuracy on standard handwritten digit datasets.
- **Real-time Performance**: The model is optimized for real-time performance, allowing it to process video frames quickly and provide immediate feedback on the recognized digits.

### Example Output

When the script is run, the webcam captures video frames, and the CNN model predicts the digit within the central bounding box. The detected digit and its confidence score are displayed on the frame. For example:

```
Detected Digit: 5
Confidence: 98.7%
```

This output indicates that the model is 98.7% confident that the digit in the central region is '5'.

### Conclusion

This script demonstrates the practical application of a CNN model for real-time handwritten digit recognition using a webcam. By leveraging the power of deep learning, it provides an accurate and efficient solution for digit recognition tasks.
"""

# streamlit digit recognition app
Script : digit.py

"""
link :https://digitbibek.streamlit.app/
"""

This script is a Streamlit application for handwritten digit recognition. It allows users to either select an image from a preset dataset or upload their own image for recognition. The application uses a pre-trained model to predict the digit in the selected or uploaded image.

Functions:
- preprocess_image(image_path): Preprocesses the image for prediction.
- model.predict(img_for_prediction): Predicts the digit in the preprocessed image.

Components:
- Title and description: Displays the title of the application.
- Subheader and selection: Asks the user whether they want to use a preset dataset or upload their own image.
- Radio button for decision: Allows the user to choose between using a preset dataset or uploading their own image.
- Logic based on the user's decision: Handles the user's choice and displays the appropriate options.
    - If 'Preset dataset' is selected: Displays radio buttons for selecting an image from the preset dataset and shows the selected image.
    - If 'Upload your own image' is selected: Provides a file uploader for the user to upload an image and shows the uploaded image.
- Button for recognizing the selected image: When clicked, it processes the selected or uploaded image, predicts the digit, and displays the result along with the processed image and prediction probabilities.

Dependencies:
- Streamlit (st)
- Matplotlib (plt)
- NumPy (np)
- Pre-trained model for digit recognition

Note:
- Ensure that the 'img dataset/' folder contains the preset images named "0.png" to "9.png".
- The preprocess_image function and the pre-trained model should be defined elsewhere in the code.
"""



# Pre-trained Models
#### CNN Model
CNN_model.keras is the model file which is trained using the MNIST dataset. The model is trained using the CNN model and the accuracy of the model is 99.2%. The model is saved in the keras format.

#### Simple Neural Network Model
my_model.keras is the model file which is trained using the MNIST dataset. The model is trained using the Neural Network model and the accuracy of the model is 98.9%.

# Demo Video

To see the application in action, check out the demo video below:

[![Handwritten Digit Recognition Demo](https://youtu.be/w5hlC74CAaI)](https://youtu.be/w5hlC74CAaI)
