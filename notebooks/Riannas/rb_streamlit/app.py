import logging
from flask import Flask, render_template, request
import pickle
from flask import jsonify
from xgboost import XGBClassifier
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = XGBClassifier()
model = pickle.load(open('xgboost_model_not_scaled.pkl', 'rb'))



# Define the order of columns
column_order = [
    'loan_type', 'lien_status',
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Initialize an empty list for input features
        init_features = []

        # Get JSON data from the POST request
        data = request.get_json()

        # Check if the provided JSON contains all the required columns
        if set(data.keys()) != set(column_order):
            return jsonify({"error": f"Input data must contain {len(column_order)} keys matching column names"})

        # Populate the list dynamically based on the order of columns
        for column_name in column_order:
            feature_value = data.get(column_name)
            if feature_value is not None:
                init_features.append(float(feature_value))

        # Perform the prediction
        prediction = model.predict(np.array(init_features).reshape(1, -1))

        # Set the prediction_text variable
        if prediction[0] == 1:
            prediction_text = "APPROVED"
        else:
            prediction_text = "DENIED"

        return jsonify({"prediction_text": prediction_text})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)