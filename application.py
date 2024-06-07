from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

app = Flask(__name__)
CORS(app)

# Paths for directories
input_dir = 'input_images'
output_dir = 'output_images'
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Load the ONNX model
onnx_model_path = r'C:\Users\dhana\Desktop\updated\27.onnx'
session = ort.InferenceSession(onnx_model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    image_file = request.files['image']
    polyline_height_cm = request.form['polyline_height_cm']
    polyline_width_cm = request.form['polyline_width_cm']
    
    # Print the polyline dimensions
    print(f"Polyline Height: {polyline_height_cm} cm, Width: {polyline_width_cm} cm")

    # Save input image
    input_image_path = os.path.join(input_dir, image_file.filename)
    image_file.save(input_image_path)

    # Preprocess the image
    new_image = Image.open(input_image_path).resize((224, 224))
    new_image_array = np.array(new_image) / 255.0  # Normalize the image
    new_image_array = np.expand_dims(new_image_array, axis=0).astype(np.float32)  # Add batch dimension and convert to float32

    # Prepare input for the ONNX model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Make predictions
    prediction = session.run([output_name], {input_name: new_image_array})[0]

    # Reshape and convert predicted points
    predicted_points = prediction[0].reshape(-1, 2).astype(int)

    # Load the original image for visualization
    resized_image = cv2.resize(cv2.imread(input_image_path), (224, 224))

    # Draw the predicted points
    for point in predicted_points:
        cv2.circle(resized_image, tuple(point), radius=2, color=(0, 255, 0), thickness=-1)

    # Save the output image
    output_image_path = os.path.join(output_dir, 'output_' + image_file.filename)
    cv2.imwrite(output_image_path, resized_image)

    # Send the output image back to the client
    return send_file(output_image_path, mimetype='image/jpeg')

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
