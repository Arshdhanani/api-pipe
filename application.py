from flask import Flask, request, jsonify, send_file, send_from_directory, abort
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import io
import onnxruntime as ort
import logging

application = Flask(__name__)
CORS(application)  # Enable CORS for all routes, restrict in production
application.logger.setLevel(logging.INFO)

# Load ONNX model
onnx_model_path = r'/var/app/current/model.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

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
        return jsonify({'message': 'Send a POST request with an image to get predictions'}), 200

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    try:
        image = Image.open(image_file.stream)
    except IOError:
        application.logger.error('Invalid image format or corrupted file.')
        return jsonify({'error': 'Invalid image format'}), 400

    # Preprocess the image
    image_array = preprocess_image(image)

    # Predict using the ONNX model
    prediction = predict(image_array)

    # Process the prediction to generate the output image
    predicted_points = prediction[0].reshape(-1, 2).astype(int)
    resized_image = cv2.resize(np.array(image), (224, 224))
    cv2.polylines(resized_image, [predicted_points], isClosed=True, color=(0, 255, 0), thickness=1)

    # Convert the processed image to bytes
    is_success, buffer = cv2.imencode(".png", resized_image)
    if not is_success:
        application.logger.error('Failed to encode image.')
        return jsonify({'error': 'Failed to encode image'}), 500
    
    io_buf = io.BytesIO(buffer)
    return send_file(io_buf, mimetype='image/png')

@application.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@application.errorhandler(500)
def handle_500(error):
    application.logger.error('Server Error: %s', (error))
    return jsonify({"error": "Internal server error"}), 500

@application.errorhandler(Exception)
def handle_exception(e):
    application.logger.error('Unhandled Exception: %s', (e))
    return jsonify({"error": "Something went wrong"}), 500

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8000, debug=True)
