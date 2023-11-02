import json
import logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib
import pickle
from flask import jsonify



app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('../notebooks/Christinas/EDA_stepwise/xgboost_model.pkl', 'rb'))

# Define the order of columns
column_order = [
    'census_tract', 'action_taken', 'loan_type', 'lien_status',
    'open_end_line_of_credit', 'loan_amount', 'combined_loan_to_value_ratio',
    'interest_rate', 'total_loan_costs', 'origination_charges', 'loan_term',
    'negative_amortization', 'interest_only_payment', 'balloon_payment',
    'other_nonamortizing_features', 'property_value', 'occupancy_type',
    'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest',
    'total_units', 'income', 'debt_to_income_ratio', 'applicant_credit_score_type',
    'co_applicant_credit_score_type', 'applicant_ethnicity_1', 'co_applicant_ethnicity_1',
    'applicant_race_1', 'applicant_race_2', 'co_applicant_race_1', 'co_applicant_race_2',
    'applicant_sex', 'co_applicant_sex', 'applicant_age', 'co_applicant_age'
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')



import numpy as np

@app.route('/predict', methods=['POST'])
def predict():
    init_features = []

    
    init_features = request.form.values()
    
    app.logger.info('Received data: %s', init_features)

    if init_features:
        final_features = np.array(init_features)  # Create a NumPy array directly
        final_features = final_features.reshape(1, -1)  # Reshape to match expected input shape
        app.logger.info('final_features: %s', final_features)
        

        prediction = model.predict(final_features)
        prediction = prediction.tolist()  # Convert the ndarray to a list
        app.logger.info('Prediction: %s', prediction)
        return jsonify({"prediction": prediction})
    else:
        app.logger.info('Invalid input values')
        return jsonify({"error": "Invalid input values"})



if __name__ == '__main__':
    app.run(debug=True)



# INFO:app:Received data: [2.0, 3.0, 2.0, 3.0]
# INFO:app:final_features: [array([2., 3., 2., 3.])]