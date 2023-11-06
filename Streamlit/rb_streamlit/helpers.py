import numpy as np

def extract_features(request_data):
    # Extract the necessary features for prediction
    features = [
        np.int64(request_data['dummyData1']['applicantData']['applicant_age']),
        np.int64(request_data['dummyData1']['applicantData']['income']),
        np.int64(request_data['dummyData1']['applicantData']['applicant_race_1']),
        np.int64(request_data['dummyData1']['applicantData']['applicant_sex']),
        np.int64(request_data['dummyData1']['applicantData']['applicant_ethnicity_1']),
        np.int64(request_data['dummyData1']['applicantData']['lenderCredits']),
        np.int64(request_data['dummyData1']['applicantData']['debt_to_income_ratio']),
        np.int64(request_data['dummyData1']['applicantData']['applicant_credit_score_type']),
        np.int64(request_data['dummyData1']['loanDetails']['loan_amount']),
        np.int64(request_data['dummyData1']['loanDetails']['interest_rate']),
        np.int64(request_data['dummyData1']['loanDetails']['totalpointsandfees']),
        np.int64(request_data['dummyData1']['loanDetails']['loan_term']),
        np.int64(request_data['dummyData1']['loanDetails']['discountPoints']),
        np.int64(request_data['dummyData1']['loanDetails']['prepaymentPenaltyTerm']),
        np.int64(request_data['dummyData1']['loanDetails']['negative_amortization']),
        np.int64(request_data['dummyData1']['loanDetails']['total_loan_costs']),
        np.int64(request_data['dummyData1']['loanDetails']['loan_type']),
        np.int64(request_data['dummyData1']['loanDetails']['loanpurpose']),
        np.int64(request_data['dummyData1']['loanDetails']['originationCharges']),
        np.int64(request_data['dummyData1']['loanDetails']['interest_only_payment']),
        np.int64(request_data['dummyData1']['loanDetails']['balloon_payment']),
        np.int64(request_data['dummyData1']['loanDetails']['other_nonamortizing_features'])
    ]
    return features
