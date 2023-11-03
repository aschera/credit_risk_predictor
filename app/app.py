import logging
from flask import Flask, render_template, request
import pickle
from flask import jsonify
from xgboost import XGBClassifier
import numpy as np
import joblib
import shap

app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('../model/xgboost_model_not_scaled.pkl', 'rb'))

# Load the explainer from the file
explainer = joblib.load('../model/explainer.pkl')

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
    
    # Initialize an empty list for input features
    init_features = []

    # Populate the list dynamically based on the order of columns
    for column_name in column_order:
        feature_value = request.form.get(column_name)
        if feature_value is not None:
            init_features.append(float(feature_value))

    # Ensure that the list has the same number of values as the column_order
    if len(init_features) != len(column_order):
        return jsonify({"error": f"Input data must contain {len(column_order)} float values"})

    # Perform the prediction
    prediction = model.predict([init_features])
    
    # Convert the prediction to a regular Python list
    prediction = prediction.tolist()

    # Set the prediction_text variable
    prediction_text = f"Prediction: {prediction[0]}"

    # Use the explainer to generate explanations
    explanation = explainer.shap_values(init_features)

    # Return a JSON response with both prediction_text and explanation
    return jsonify({"prediction_text": prediction_text, "explanation": explanation.tolist()})




if __name__ == '__main__':
    app.run(debug=True)
