from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load your model
model = load_model('my_model.keras')

@app.route('/predict/', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Convert file to image
    img = Image.open(file.stream)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to the input shape of your model
    img = np.array(img)
    img = img.reshape((1, 28, 28, 1))  # Adjust shape as needed
    img = img / 255.0  # Normalize

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)

    return jsonify({'predicted_class': int(predicted_class)})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
