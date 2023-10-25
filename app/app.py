import json
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
from flask import request


app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('../model/logistic_regression_model.pkl')

# Define a dictionary to store the test data
test_data = {}

# Load test data from JSON and populate the test_data dictionary
with open('./static/testdata.json', 'r') as json_file:
    test_data = json.load(json_file)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

import numpy as np

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Access the JSON data sent from the client
        request_data = request.get_json()

        # Create a list of values from the dictionary and convert them to a 2D array
        input_data = [list(request_data.values())]
        logger.info(input_data )


        # Make prediction using the pre-trained model
         # prediction = model.predict(input_data)
         # predicted_class = prediction

        # Define the "action taken" based on the model output
         # action_taken = 'accepted' if predicted_class == 1 else 'rejected'

        # Log the prediction result
         # logger.info('Action taken: %s', action_taken)

        # Return prediction as JSON response
         # response_data = {'action_taken': action_taken}
        return jsonify(input_data), 200

if __name__ == '__main__':
    app.run(debug=True)

