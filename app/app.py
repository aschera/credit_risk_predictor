import json
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
from flask import request


app = Flask(__name__)

# Load the pre-trained model
model_xgboost = joblib.load('./xgboost_model.pkl') # excepts 33 features:

expected_features = ['census_tract', 'loan_type', 'lien_status', 'open_end_line_of_credit', 
                     'loan_amount', 'combined_loan_to_value_ratio', 'interest_rate', 'total_loan_costs', 
                     'origination_charges', 'loan_term', 'negative_amortization', 'interest_only_payment', 
                     'balloon_payment', 'other_nonamortizing_features', 'property_value', 'occupancy_type', 
                     'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest', 
                     'total_units', 'income', 'debt_to_income_ratio', 'applicant_credit_score_type', 
                     'co_applicant_credit_score_type', 'applicant_ethnicity_1', 'co_applicant_ethnicity_1', 
                     'applicant_race_1', 'applicant_race_2', 'co_applicant_race_1', 
                     'co_applicant_race_2', 'applicant_sex', 'co_applicant_sex', 'applicant_age', 
                     'co_applicant_age']

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
        logger.info(input_data)

        # Create a dictionary that maps feature names to values
        input_dict = dict(zip(expected_features, input_data[0]))

        # Create prepared_data by accessing values in the expected order
        prepared_data = [input_dict[feature] for feature in expected_features]

        # Make prediction using the pre-trained model
        prediction = model_xgboost.predict([prepared_data])
        predicted_class = prediction

        # Define the "action taken" based on the model output
        action_taken = 'accepted' if predicted_class == 1 else 'rejected'

        # Log the prediction result
        logger.info('Action taken: %s', action_taken)

        # Return prediction as JSON response
        response_data = {'action_taken': action_taken}
        return jsonify(response_data), 200


if __name__ == '__main__':
    app.run(debug=True)

