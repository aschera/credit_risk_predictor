import numpy as np

def extract_features(request_data):
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
    return features
