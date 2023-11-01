from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('logistic_regression_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    request_data = request.get_json()

    # Extract the necessary features for prediction
    features = [
        request_data['dummyData1']['applicantData']['applicant_age'],
        request_data['dummyData1']['applicantData']['income'],
        request_data['dummyData1']['applicantData']['applicant_race_1'],
        request_data['dummyData1']['applicantData']['applicant_sex'],
        request_data['dummyData1']['applicantData']['applicant_ethnicity_1'],
        request_data['dummyData1']['applicantData']['lenderCredits'],
        request_data['dummyData1']['applicantData']['debt_to_income_ratio'],
        request_data['dummyData1']['applicantData']['applicant_credit_score_type'],

        request_data['dummyData1']['loanDetails']['loan_amount'],
        request_data['dummyData1']['loanDetails']['interest_rate'],
        request_data['dummyData1']['loanDetails']['totalpointsandfees'],
        request_data['dummyData1']['loanDetails']['loan_term'],
        request_data['dummyData1']['loanDetails']['discountPoints'],
        request_data['dummyData1']['loanDetails']['prepaymentPenaltyTerm'],
        request_data['dummyData1']['loanDetails']['negative_amortization'],
        request_data['dummyData1']['loanDetails']['total_loan_costs'],
        request_data['dummyData1']['loanDetails']['loan_type'],
        request_data['dummyData1']['loanDetails']['loanpurpose'],
        request_data['dummyData1']['loanDetails']['originationCharges'],
        request_data['dummyData1']['loanDetails']['interest_only_payment'],
        request_data['dummyData1']['loanDetails']['balloon_payment'],
        request_data['dummyData1']['loanDetails']['other_nonamortizing_features']
    ]


    # Convert features to a 2D array for model prediction
    features_array = np.array([features])

    # Make prediction
    prediction = model.predict(features_array)
    predicted_class = prediction[0]

    # Return prediction as JSON response
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
