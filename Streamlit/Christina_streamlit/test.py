import streamlit as st
import pickle

# Load the trained model
with open('../../model/xgboost_model_not_scaled.pkl', 'rb') as model_file:
    loan_model = pickle.load(model_file)

st.title("Classifying Loan")
st.markdown("Model to classify loan applications into accepted or declined")

st.header("Features")
col1, col2 = st.columns(2)

user_inputs = {
    "Loan Type": st.selectbox("Loan Type", ["Type 1", "Type 2", "Type 3"], key="Loan Type"),
    "Lien Status": st.selectbox("Lien Status", ["Status 1", "Status 2"], key="Lien Status"),
    "Open End Line of Credit": st.selectbox("Open End Line of Credit", ["Option 1", "Option 2"], key="Open End Line of Credit"),
    "Loan Amount": st.slider("Loan Amount", 10000, 40000, 1000, key="Loan Amount"),
    "Combined Loan to Value Ratio": st.slider("Combined Loan to Value Ratio", 80, 95, 5, key="Combined Loan to Value Ratio"),
    "Interest Rate": st.slider("Interest Rate", 3.5, 4.5, 0.1, key="Interest Rate"),
    "Total Loan Costs": st.slider("Total Loan Costs", 500, 1500, 100, key="Total Loan Costs"),
    "Origination Charges": st.slider("Origination Charges", 200, 600, 50, key="Origination Charges"),
    "Loan Term": st.slider("Loan Term", 15, 45, 5, key="Loan Term"),
    "Negative Amortization": st.selectbox("Negative Amortization", ["Yes", "No"], key="Negative Amortization"),
    "Interest Only Payment": st.selectbox("Interest Only Payment", ["Yes", "No"], key="Interest Only Payment"),
    "Balloon Payment": st.selectbox("Balloon Payment", ["Yes", "No"], key="Balloon Payment"),
    "Other Non-amortizing Features": st.selectbox("Other Non-amortizing Features", ["Yes", "No"], key="Other Non-amortizing Features"),
    "Property Value": st.slider("Property Value", 100000, 300000, 10000, key="Property Value"),
    "Occupancy Type": st.selectbox("Occupancy Type", ["Primary Residence", "Second Home", "Investment Property"], key="Occupancy Type"),
    "Manufactured Home Secured Property Type": st.selectbox("Manufactured Home Secured Property Type", ["Type A", "Type B", "Type C"], key="Manufactured Home Secured Property Type"),
    "Manufactured Home Land Property Interest": st.selectbox("Manufactured Home Land Property Interest", ["Interest 1", "Interest 2"], key="Manufactured Home Land Property Interest"),
    "Total Units": st.slider("Total Units", 1, 3, 1, key="Total Units"),
    "Income": st.slider("Income", 30000, 50000, 1000, key="Income"),
    "Debt to Income Ratio": st.slider("Debt to Income Ratio", 30, 50, 1, key="Debt to Income Ratio"),
    "Applicant Credit Score Type": st.selectbox("Applicant Credit Score Type", ["Type 1", "Type 2", "Type 3"], key="Applicant Credit Score Type"),
    "Co-applicant Credit Score Type": st.selectbox("Co-applicant Credit Score Type", ["Type 1", "Type 2", "Type 3"], key="Co-applicant Credit Score Type"),
    "Applicant Ethnicity 1": st.selectbox("Applicant Ethnicity 1", ["Ethnicity 1", "Ethnicity 2"], key="Applicant Ethnicity 1"),
    "Co-applicant Ethnicity 1": st.selectbox("Co-applicant Ethnicity 1", ["Ethnicity 1", "Ethnicity 2"], key="Co-applicant Ethnicity 1"),
    "Applicant Race 1": st.selectbox("Applicant Race 1", ["Race 1", "Race 2"], key="Applicant Race 1"),
    "Applicant Race 2": st.selectbox("Applicant Race 2", ["Race 1", "Race 2"], key="Applicant Race 2"),
    "Co-applicant Race 1": st.selectbox("Co-applicant Race 1", ["Race 1", "Race 2"], key="Co-applicant Race 1"),
    "Co-applicant Race 2": st.selectbox("Co-applicant Race 2", ["Race 1", "Race 2"], key="Co-applicant Race 2"),
    "Applicant Sex": st.selectbox("Applicant Sex", ["Male", "Female"], key="Applicant Sex"),
    "Co-applicant Sex": st.selectbox("Co-applicant Sex", ["Male", "Female"], key="Co-applicant Sex"),
    "Applicant Age": st.slider("Applicant Age", 20, 50, 5, key="Applicant Age"),
    "Co-applicant Age": st.slider("Co-applicant Age", 20, 50, 5, key="Co-applicant Age"),
}

# Ensure a unique key for the "Predict" button
predict_button = st.button("Predict Loan Status", key="predict_button")

if predict_button:
    # Prepare the input data for the model
    input_data = [user_inputs[key] for key in user_inputs]

    # Make a prediction
    # prediction = loan_model.predict([input_data])

    #if prediction[0] == 1:
    st.success("Loan Application Accepted")
    #else:
        #st.error("Loan Application Declined")
