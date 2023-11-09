import logging
import flask
from flask import Flask, render_template, request
import pickle
from flask import jsonify
import numpy as np
import shap
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the pre-trained model
current_directory = os.path.dirname(os.path.abspath(__file__))
relative_model_path = os.path.join(
    current_directory,
    "..",
    "Streamlit",
    "streamlit-m",
    "static",
    "xgboost_model_not_scaled.pkl"
)

model = pickle.load(open(relative_model_path, 'rb'))

# Load the SHAP explainer
explainer = shap.TreeExplainer(model)

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
    
    # Convert to a 2D array
    init_features_2d = np.array([init_features])  

    # Perform the prediction
    prediction = model.predict([init_features])
    
    # Convert the prediction to a regular Python list
    prediction = prediction.tolist()

    # Set the prediction_text variable
    prediction_text = f"Prediction: {prediction[0]}"

    # Use the explainer to generate explanations for the sample row
    explanation = explainer.shap_values(init_features_2d)

    # Convert the explanation to a regular Python list
    explanation = explanation.tolist()

    # Return a JSON response with both prediction_text and explanation and column names.
    return jsonify({"prediction_text": prediction_text, "explanation": explanation, "features": column_order })

if __name__ == '__main__':
    app.run(debug=True)
