import json
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
from helpers import extract_features

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('../model/logistic_regression_model.pkl')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        request_data = request.get_json()

        # Log the request data
        logger.info('Received request data: %s', request_data)

        # Extract the necessary features for prediction using the extract_features function
        features = extract_features(request_data)

        # Convert features to a 2D array for model prediction
        features_array = np.array([features])

        # Make prediction
        prediction = model.predict(features_array)
        predicted_class = int(prediction[0])

        # Define prediction result based on the model output
        prediction_result = 'accepted' if predicted_class == 1 else 'rejected'

        # Log the prediction result
        logger.info('Prediction: %s', prediction_result)

        # Return prediction as JSON response
        response_data = {'prediction': prediction_result}
        return jsonify(response_data), 200

    except Exception as e:
        # Log and return an error message in case of an exception
        logger.error('Error predicting: %s', str(e))
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)
