from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import io
import onnxruntime as ort
import logging
import os

# Create directories if they do not exist
input_dir = 'inputimages'
output_dir = 'outputimages'
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

application = Flask(__name__)
CORS(application)  # Enable CORS for all routes

# Load ONNX model with error handling
try:
    onnx_model_path = r'/var/app/current/27.onnx'
    ort_session = ort.InferenceSession(onnx_model_path)
    logging.info("ONNX model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load ONNX model: {e}")
    raise e

def preprocess_image(image):
    # Resize and normalize the image
    new_image = image.resize((224, 224))
    new_image_array = np.array(new_image) / 255.0
    new_image_array = np.expand_dims(new_image_array, axis=0).astype(np.float32)
    return new_image_array

def predict(image_array):
    # Run the ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: image_array}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0]

@application.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    input_image_path = os.path.join(input_dir, image_file.filename)
    
    try:
        # Save the uploaded image to the inputimages directory
        image_file.save(input_image_path)
    except IOError:
        return jsonify({'error': 'Failed to save image'}), 500

    try:
        # Open the image from the inputimages directory
        image = Image.open(input_image_path)
    except IOError:
        return jsonify({'error': 'Invalid image format'}), 400

    # Preprocess the image
    image_array = preprocess_image(image)

    # Predict using the ONNX model
    try:
        prediction = predict(image_array)
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

    # Process the prediction to generate the output image
    try:
        predicted_points = prediction[0].reshape(-1, 2).astype(int)
        resized_image = cv2.resize(np.array(image), (224, 224))
        cv2.polylines(resized_image, [predicted_points], isClosed=True, color=(0, 255, 0), thickness=1)
    except Exception as e:
        logging.error(f"Image processing error: {e}")
        return jsonify({'error': 'Image processing failed'}), 500

    # Save output image
    output_image_filename = 'output_' + image_file.filename
    output_image_path = os.path.join(output_dir, output_image_filename)
    cv2.imwrite(output_image_path, resized_image)

    # Return the relative path of the output image along with the filename
    output_image_url = f"/output/{output_image_filename}"
    return jsonify({'output_image_url': output_image_url})

@application.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(output_dir, filename)

@application.route('/')
def index():
    return send_from_directory('templates', 'index.html')

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=80)
