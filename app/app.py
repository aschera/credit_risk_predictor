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

# Define the expected feature names in the correct order
expected_feature_names = ['census_tract', 'loan_type', 'lien_status', 'open_end_line_of_credit', 'loan_amount', 'combined_loan_to_value_ratio', 'interest_rate', 'total_loan_costs', 'origination_charges', 'loan_term', 'negative_amortization', 'interest_only_payment', 'balloon_payment', 'other_nonamortizing_features', 'property_value', 'occupancy_type', 'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest', 'total_units', 'income', 'debt_to_income_ratio', 'applicant_credit_score_type', 'co_applicant_credit_score_type', 'applicant_ethnicity_1', 'co_applicant_ethnicity_1', 'applicant_race_1', 'applicant_race_2', 'co_applicant_race_1', 'co_applicant_race_2', 'applicant_sex', 'co_applicant_sex', 'applicant_age', 'co_applicant_age']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:

            # ----------------------------------------------------------------#
            # --------------------------Test----------------------------------#
            '''
            # Load the data from the NumPy array file
            row1_data = np.load('../notebooks/Christinas/EDA_stepwise/row1.npy')

            # Make sure the loaded data is in the right format
            input_data_test = row1_data .reshape(1, -1)  # Reshape to match the expected input shape

            # Make prediction using the pre-trained model
            prediction1 = model_xgboost.predict(input_data_test)

            # Define the "action taken" based on the model output
            action_taken1 = 'accepted' if prediction1 == 1 else 'rejected'
            logger.info('Prediction1: %s', prediction1)
            logger.info('Action taken1: %s', action_taken1)
            
            '''

            
            # ----------------------------------------------------------------#
            # --------------------------Form----------------------------------#
            
            # Access the JSON data sent from the client
            request_data = request.get_json()

            # Map expected feature names to values in request_data
            input_data = [float(request_data.get(feature, 0)) for feature in expected_feature_names]

            # Convert the input data to a NumPy array
            input_data = np.array(input_data).reshape(1, -1)
            logger.info('input_data: %s', input_data)

            # Make prediction using the pre-trained model
            prediction = model_xgboost.predict(input_data)

            # Define the "action taken" based on the model output
            action_taken = 'accepted' if prediction == 1 else 'rejected'
            
            # Return prediction as JSON response
            response_data = {'action_taken': action_taken}

            return jsonify(response_data), 200
        except Exception as e:
            logging.error('Prediction Error: %s', str(e))
            return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)

