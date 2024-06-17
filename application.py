from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import io
import onnxruntime as ort
import logging
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from botocore.client import Config
import os

# Initialize the S3 client
s3_client = boto3.client('s3', config=Config(signature_version='s3v4'))

# Set the S3 bucket name from environment variable
S3_BUCKET_NAME = os.getenv('s3flask ')

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
    image_filename = image_file.filename
    
    try:
        # Read the image file into a PIL Image
        image = Image.open(image_file)
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
        
        # Convert the processed image to bytes
        _, buffer = cv2.imencode('.png', resized_image)
        output_image_bytes = io.BytesIO(buffer)
    except Exception as e:
        logging.error(f"Image processing error: {e}")
        return jsonify({'error': 'Image processing failed'}), 500

    # Save input image and output image to S3
    try:
        input_image_s3_key = f"inputimages/{image_filename}"
        s3_client.upload_fileobj(image_file, S3_BUCKET_NAME, input_image_s3_key)
        
        output_image_s3_key = f"outputimages/output_{image_filename}"
        s3_client.upload_fileobj(output_image_bytes, S3_BUCKET_NAME, output_image_s3_key)
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"S3 credentials error: {e}")
        return jsonify({'error': 'S3 credentials error'}), 500
    except Exception as e:
        logging.error(f"Failed to upload to S3: {e}")
        return jsonify({'error': 'Failed to upload to S3'}), 500

    # Return the relative path of the output image along with the filename
    output_image_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{output_image_s3_key}"
    return jsonify({'output_image_url': output_image_url})

@application.route('/output/<filename>')
def output_file(filename):
    output_image_s3_key = f"outputimages/{filename}"
    try:
        output_image = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=output_image_s3_key)
        return send_file(io.BytesIO(output_image['Body'].read()), mimetype='image/png')
    except Exception as e:
        logging.error(f"Failed to retrieve file from S3: {e}")
        return jsonify({'error': 'Failed to retrieve file from S3'}), 500

@application.route('/')
def index():
    return send_from_directory('templates', 'index.html')

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=80)
