from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io

app = Flask(__name__)

# Load the trained MNIST model
model = tf.keras.models.load_model('emnist_model.h5')

def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = ImageOps.invert(image)  # Invert colors (MNIST has white digits on black background)
    image = np.array(image) / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

@app.route('/predict', methods=['POST'])
def predict_digit():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)
    
    prediction = model.predict(processed_image)
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    return jsonify({"digit": predicted_digit, "confidence": confidence})

if __name__ == '__main__':
    app.run(debug=True)