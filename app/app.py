from flask import Flask, render_template, request, jsonify
import json
import joblib
import numpy as np
from helpers import extract_features  # Import the extract_features function

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('../model/logistic_regression_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    request_data = request.get_json()

    print(request_data )

    # Extract the necessary features for prediction using the extract_features function
    #  features = extract_features(request_data)

    # Convert features to a 2D array for model prediction
    # features_array = np.array([features])

    # Make prediction
    # prediction = model.predict(features_array)
    # predicted_class = int(prediction[0])

    # Return prediction as JSON response
    response_data = {'prediction': 'accepted'}
    # return json.dumps(response_data), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True)
