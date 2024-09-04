from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import base64
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained model
model = load_model('my_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['image']
        image = Image.open(io.BytesIO(base64.b64decode(data.split(',')[1]))).convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image)
        image_array = 255 - image_array
        image_array = image_array / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        # Debug: Print the mean value of the image array
        print(f"Mean value of the image array: {np.mean(image_array)}")
        
        if np.mean(image_array) > 0.99:
            return jsonify({'prediction': 'Blank or No Digit Detected'})
        
        prediction = model.predict(image_array).argmax(axis=1)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500


if __name__ == '__main__':
    app.run(debug=True)
