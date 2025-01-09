import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load Keras model
model = load_model('CNN_model.keras')

# Streamlit app title
st.title("Handwritten Digit Recognition")

# Initialize session state for webcam
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

# Define central bounding box parameters
frame_width, frame_height = 640, 480  # Default webcam resolution
box_size = 200  # Size of the central bounding box
box_x1 = (frame_width - box_size) // 2
box_y1 = (frame_height - box_size) // 2
box_x2 = box_x1 + box_size
box_y2 = box_y1 + box_size

# Streamlit video display
frame_placeholder = st.empty()

# Start and Stop buttons
if st.button('Start Webcam'):
    st.session_state.webcam_active = True
if st.button('Stop Webcam'):
    st.session_state.webcam_active = False

# Webcam processing loop
if st.session_state.webcam_active:
    cap = cv2.VideoCapture(0)
    while st.session_state.webcam_active:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to the standard size (optional, depends on model requirements)
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Draw central bounding box
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)

        # Crop the region inside the bounding box
        cropped_region = frame[box_y1:box_y2, box_x1:box_x2]

        # Preprocess the cropped region for the Keras model
        input_image = cv2.resize(cropped_region, (28, 28))  # Resize to model input size (e.g., 28x28 for MNIST)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
        input_image = input_image / 255.0  # Normalize to [0, 1]
        input_image = np.expand_dims(input_image, axis=-1)  # Add channel dimension
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

        # Run Keras model inference
        predictions = model.predict(input_image)
        detected_digit = np.argmax(predictions)
        confidence_score = np.max(predictions)

        if confidence_score < 0.9:
            text = "No digit recognized"
        else:
            text = f"Digit: {detected_digit} ({confidence_score:.2f})"

        # Display detected digit and confidence on the frame
        cv2.putText(frame, text, (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        frame_placeholder.image(frame_rgb)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()