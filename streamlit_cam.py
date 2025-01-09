import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# Load the Keras model
model = load_model('CNN_model.keras')

# Streamlit app title
st.title("Handwritten Digit Recognition")

# Instructions
st.markdown("Click 'Start Webcam' to begin video streaming. Click 'Stop Webcam' to end.")

# Button to start/stop the webcam
if 'run' not in st.session_state:
    st.session_state.run = False

start_button = st.button("Start Webcam")
stop_button = st.button("Stop Webcam")

if start_button:
    st.session_state.run = True

if stop_button:
    st.session_state.run = False

# Streamlit container for video feed
video_feed = st.empty()

# Webcam processing loop
if st.session_state.run:
    cap = cv2.VideoCapture(0)

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not access webcam.")
            break

        # Define bounding box
        frame_height, frame_width = frame.shape[:2]
        box_size = 200
        box_x1 = (frame_width - box_size) // 2
        box_y1 = (frame_height - box_size) // 2
        box_x2, box_y2 = box_x1 + box_size, box_y1 + box_size

        # Draw central bounding box
        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)

        # Crop and preprocess the region of interest (ROI)
        cropped_region = frame[box_y1:box_y2, box_x1:box_x2]
        input_image = cv2.resize(cropped_region, (28, 28))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        input_image = input_image / 255.0  # Normalize to [0, 1]
        input_image = np.expand_dims(input_image, axis=-1)  # Add channel dimension
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

        # Model inference
        predictions = model.predict(input_image)
        detected_digit = np.argmax(predictions)
        confidence_score = np.max(predictions)

        # Display prediction
        text = f"Digit: {detected_digit} ({confidence_score:.2f})" if confidence_score >= 0.9 else "No digit recognized"
        cv2.putText(frame, text, (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update video feed
        video_feed.image(frame_rgb, channels="RGB")

    # Release resources when done
    cap.release()
