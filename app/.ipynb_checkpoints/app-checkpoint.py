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
        request_data['dummyData1']['applicantData']['age'],
        request_data['dummyData1']['applicantData']['income'],
        request_data['dummyData1']['applicantData']['race'],
        request_data['dummyData1']['applicantData']['sex'],
        request_data['dummyData1']['applicantData']['ethnicity'],
        request_data['dummyData1']['applicantData']['lenderCredits'],
        request_data['dummyData1']['applicantData']['debtToIncomeRatio'],
        request_data['dummyData1']['applicantData']['creditScore'],

        request_data['dummyData1']['loanDetails']['loanAmount'],
        request_data['dummyData1']['loanDetails']['interestRate'],
        request_data['dummyData1']['loanDetails']['totalpointsandfees'],
        request_data['dummyData1']['loanDetails']['loanterm'],
        request_data['dummyData1']['loanDetails']['discountPoints'],
        request_data['dummyData1']['loanDetails']['prepaymentPenaltyTerm'],
        request_data['dummyData1']['loanDetails']['negativeAmortization'],
        request_data['dummyData1']['loanDetails']['totalloancosts'],
        request_data['dummyData1']['loanDetails']['loantype'],
        request_data['dummyData1']['loanDetails']['loanpurpose'],
        request_data['dummyData1']['loanDetails']['originationCharges'],
        request_data['dummyData1']['loanDetails']['interestOnlyPayment'],
        request_data['dummyData1']['loanDetails']['balloonPayment'],
        request_data['dummyData1']['loanDetails']['otherNonamortizingFeatures']
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
