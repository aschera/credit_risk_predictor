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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

column_mapping = {
                0: 'census_tract',
                1: 'loan_type',
                2: 'lien_status',
                3: 'open_end_line_of_credit',
                4: 'loan_amount',
                5: 'combined_loan_to_value_ratio',
                6: 'interest_rate',
                7: 'total_loan_costs',
                8: 'origination_charges',
                9: 'loan_term',
                10: 'negative_amortization',
                11: 'interest_only_payment',
                12: 'balloon_payment',
                13: 'other_nonamortizing_features',
                14: 'property_value',
                15: 'occupancy_type',
                16: 'manufactured_home_secured_property_type',
                17: 'manufactured_home_land_property_interest',
                18: 'total_units',
                19: 'income',
                20: 'debt_to_income_ratio',
                21: 'applicant_credit_score_type',
                22: 'co_applicant_credit_score_type',
                23: 'applicant_ethnicity_1',
                24: 'co_applicant_ethnicity_1',
                25: 'applicant_race_1',
                26: 'applicant_race_2',
                27: 'co_applicant_race_1',
                28: 'co_applicant_race_2',
                29: 'applicant_sex',
                30: 'co_applicant_sex',
                31: 'applicant_age',
                32: 'co_applicant_age'
            }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
            
            '''
            # Access the JSON data sent from the client
            request_data = request.get_json()

            # Create a dictionary with feature names and values
            input_data_dict = {}

            for feature in range(33):
                input_data_dict[feature] = float(request_data.get(feature, 0))

            # Create a DataFrame from the dictionary
            input_data_df = pd.DataFrame(input_data_dict, index=[0])

            # Rename the columns using the mapping dictionary
            input_data_df.rename(columns=column_mapping, inplace=True)

            # save to test
            input_data_df.to_csv('./your_output_file.csv', index=False)

            logger.info('input_data_df: %s', input_data_df)

            # Make prediction using the pre-trained model
            prediction = model_xgboost.predict(input_data_df)
            logger.info('prediction: %s', prediction)

            # Define the "action taken" based on the model output
            action_taken = 'accepted' if prediction == 1 else 'rejected'

            # Return prediction as JSON response
            response_data = {'action_taken': action_taken}

            return jsonify(response_data), 200
            
            '''
            if request.method == 'POST':
                # Print the keys in the request.form dictionary
                print(request.form.keys())
    
                # Access data from the form fields
                #applicant_age = int(request.form['applicant_age'])
                #applicant_race_1 = int(request.form['applicant_race_1'])
                #applicant_ethnicity_1 = int(request.form['applicant_ethnicity_1'])
                #income = float(request.form['income'])
                
                # You can now use these variables in your prediction logic
                
                # For example, you can create a list of these values:
                #init_features = [applicant_age, applicant_race_1, applicant_ethnicity_1, income]

                # Convert the list of values to a NumPy array
                #final_features = np.array(init_features)

                #logger.info('final_features: %s', final_features)
                
                # Now, you can use final_features to make predictions
                # Replace the following line with your actual prediction code
                prediction = 'test'
                
                return render_template('index.html', prediction_text='Prediction: {}'.format(prediction))


        # prediction = model.predict(final_features) # Make a prediction


if __name__ == '__main__':
    app.run(debug=True)