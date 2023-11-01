import json
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
from flask import request
import numpy as np
import requests
import json
import logging
import numpy as np
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model_xgboost = joblib.load('../notebooks/Christinas/EDA_stepwise/xgboost_model.pkl') # excepts 33 features:

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


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Access the JSON data sent from the client
            request_data = request.get_json()
            logger.info('request_data: %s', request_data)

            # Convert the request data to a pandas DataFrame
            input_df = pd.DataFrame(request_data, index=[0])

            # Make prediction using the pre-trained model
            prediction = model_xgboost.predict(input_df)

            # Define the "action taken" based on the model output
            action_taken = 'rejected' if prediction == 0 else 'accepted'
            logger.info('Action taken: %s', action_taken)

            # Return prediction as JSON response
            response_data = {'action_taken': action_taken}
            return jsonify(response_data), 200
        except Exception as e:
            logger.error('Prediction Error: %s', str(e))
            return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)

