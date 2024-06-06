from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import io
import onnxruntime as ort
import logging
import os
import requests
from io import BytesIO

# Create directories
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

@application.route('/predict', methods=['GET', 'POST'])
def predict_route():
    if request.method == 'GET':
        return jsonify({'message': 'Send a POST request with an image URL to get predictions'}), 200
    
    if 'image_url' not in request.form:
        return jsonify({'error': 'No image URL provided'}), 400
    
    image_url = request.form['image_url']

    try:
        # Fetch the image from the provided URL
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch image from URL'}), response.status_code

        # Open the image from the response content
        image = Image.open(BytesIO(response.content))
    except Exception as e:
        logging.error(f"Error fetching or opening image: {e}")
        return jsonify({'error': 'Failed to fetch or open image from URL'}), 500

    # Get height and width parameters from the request in cm
    polyline_height_cm = float(request.form.get('polyline_height_cm'))
    polyline_width_cm = float(request.form.get('polyline_width_cm'))

    # Rest of your code remains the same...

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

        # Calculate bounding box coordinates
        min_x, min_y = np.min(predicted_points, axis=0)
        max_x, max_y = np.max(predicted_points, axis=0)

        # Calculate width and height of bounding box
        bb_width = max_x - min_x
        bb_height = max_y - min_y

        # Calculate scale factors for resizing
        scale_x = polyline_width_px / bb_width
        scale_y = polyline_height_px / bb_height

        # Scale the predicted points
        scaled_points = ((predicted_points - [min_x, min_y]) * [scale_x, scale_y]).astype(int)

        # Create a blank image with specified dimensions
        resized_image = np.zeros((polyline_height_px, polyline_width_px, 3), dtype=np.uint8)

        # Draw polyline on the image
        cv2.polylines(resized_image, [scaled_points], isClosed=True, color=(0, 255, 0), thickness=1)
    except Exception as e:
        logging.error(f"Image processing error: {e}")
        return jsonify({'error': 'Image processing failed'}), 500

    # Convert the processed image to bytes
    is_success, buffer = cv2.imencode(".png", resized_image)
    if not is_success:
        return jsonify({'error': 'Failed to encode image'}), 500
    
    io_buf = io.BytesIO(buffer)

    return send_file(io_buf, mimetype='image/png')

@application.route('/')
def index():
    return send_from_directory('templates', 'index.html')

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=80)
