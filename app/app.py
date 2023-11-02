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
import pickle

app = Flask(__name__)

# Load the pre-trained model
# model_xgboost = joblib.load('../notebooks/Christinas/EDA_stepwise/xgboost_model.pkl')
model = pickle.load(open('../notebooks/Christinas/EDA_stepwise/xgboost_model.pkl', 'rb')) # Load the trained model

# Define the order of columns
column_order = ['census_tract', 'action_taken', 'loan_type', 'lien_status', 
                'open_end_line_of_credit', 'loan_amount', 'combined_loan_to_value_ratio', 
                'interest_rate', 'total_loan_costs', 'origination_charges', 'loan_term', 
                'negative_amortization', 'interest_only_payment', 'balloon_payment', 
                'other_nonamortizing_features', 'property_value', 'occupancy_type', 
                'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest', 
                'total_units', 'income', 'debt_to_income_ratio', 'applicant_credit_score_type', 
                'co_applicant_credit_score_type', 'applicant_ethnicity_1', 'co_applicant_ethnicity_1', 
                'applicant_race_1', 'applicant_race_2', 'co_applicant_race_1', 'co_applicant_race_2', 
                'applicant_sex', 'co_applicant_sex', 'applicant_age', 'co_applicant_age']


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    init_features = []
    final_features = []

    if request.method == 'POST':
        for column_name in column_order:
            x = request.form.get(column_name, 'undefined')
            if x.strip().lower() == 'undefined':
                float_value = 0.0
            try:
                float_value = float(x)
                init_features.append(float_value)
            except ValueError:
                float_value = 0.0
                pass

        logger.info('init_features: %s', init_features)

        if init_features:
            final_features = [np.array(init_features)]
            prediction = model.predict(final_features)
            
            logger.info('prediction: %s', prediction)

            return render_template('index.html', prediction_text='Prediction: {}'.format(prediction))
        else:
            # Handle the case where no valid numeric values were found
            return render_template('index.html', prediction_text='Invalid input values')

if __name__ == '__main__':
    app.run(debug=True)