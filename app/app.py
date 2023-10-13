import json
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('../model/logistic_regression_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    # Get the JSON data from the request
    request_data = request.get_json()

    print(request_data)

    # Extract the necessary features for prediction
    features = [
        
        np.int64(request_data['dummyData1']['applicantData']['age']),
        np.int64(request_data['dummyData1']['applicantData']['income']),
        np.int64(request_data['dummyData1']['applicantData']['race']),
        np.int64(request_data['dummyData1']['applicantData']['sex']),
        np.int64(request_data['dummyData1']['applicantData']['ethnicity']),
        np.int64(request_data['dummyData1']['applicantData']['lenderCredits']),
        np.int64(request_data['dummyData1']['applicantData']['debtToIncomeRatio']),
        np.int64(request_data['dummyData1']['applicantData']['creditScore']),
        np.int64(request_data['dummyData1']['loanDetails']['loanAmount']),
        np.int64(request_data['dummyData1']['loanDetails']['interestRate']),
        np.int64(request_data['dummyData1']['loanDetails']['totalpointsandfees']),
        np.int64(request_data['dummyData1']['loanDetails']['loanterm']),
        np.int64(request_data['dummyData1']['loanDetails']['discountPoints']),
        np.int64(request_data['dummyData1']['loanDetails']['prepaymentPenaltyTerm']),
        np.int64(request_data['dummyData1']['loanDetails']['negativeAmortization']),
        np.int64(request_data['dummyData1']['loanDetails']['totalloancosts']),
        np.int64(request_data['dummyData1']['loanDetails']['loantype']),
        np.int64(request_data['dummyData1']['loanDetails']['loanpurpose']),
        np.int64(request_data['dummyData1']['loanDetails']['originationCharges']),
        np.int64(request_data['dummyData1']['loanDetails']['interestOnlyPayment']),
        np.int64(request_data['dummyData1']['loanDetails']['balloonPayment']),
        np.int64(request_data['dummyData1']['loanDetails']['otherNonamortizingFeatures'])
    ]


    # Convert features to a 2D array for model prediction
    features_array = np.array([features])

    # Make prediction
    prediction = model.predict(features_array)
    predicted_class = prediction[0]
    
    print(features_array)
    print(predicted_class)

    # Return prediction as JSON response
    response_data = {'prediction': int(predicted_class)}  # Convert to int
    return json.dumps(response_data), 200, {'Content-Type': 'application/json'}

if __name__ == '__main__':
    app.run(debug=True)
