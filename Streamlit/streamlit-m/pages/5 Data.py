import streamlit as st
import pandas as pd

# ----------------------------------------------------------------#
# Christinas paths.
c_final_dataset= 'C:/Users/asche/OneDrive/Dokumenter/repos/Streamlit/streamlit-m/static/final_dataset.csv';
# Markus paths.
# m_final_dataset= 'C:/Users/marku/Documents/Programmering/Credit_Risk_Predictor/credit_risk_predictor/Streamlit/streamlit-m/static/final_dataset.csv';
# ----------------------------------------------------------------#

st.header("Place for Interactive data page")

@st.cache_data()
def load_data():
    df = pd.read_csv(c_final_dataset)
    return df

df = load_data()

#Function to map columns
def map_column(column, labels):
    return df[column].map(labels)

#Label dictionaries
action_taken_labels = {
    1: 'Accepted',
    3: 'Rejected'
}
loan_type_labels = {
    1: 'Conventional',
    2: 'FHA',
    3: 'VA',
    4: 'RHS or FSA'
}

lien_status_labels = {
    1: 'Secured by a first lien',
    2: 'Secured by a subordinate lien'
}

line_of_credit_labels = {
    1: 'Open-end line of credit',
    2: 'Not an open-end line of credit'
}

df["action_taken"] = map_column('action_taken', action_taken_labels)
df['loan_type'] = map_column('loan_type', loan_type_labels)
df['lien_status'] = map_column('lien_status', lien_status_labels)
df['open_end_line_of_credit'] = map_column('open_end_line_of_credit', line_of_credit_labels)


st.dataframe(df, hide_index=True)


column_to_inspect = 'action_taken'
value_counts = df[column_to_inspect].value_counts()
st.write(f"Value Counts for {column_to_inspect}: ")
st.dataframe(value_counts)

with st.sidebar:
    st.header("Filters")
    st.write("Action taken")
    #Filter for action taken
    st.write("Loan type")
    #filter for loan type
    st.write("Loan amount")
    #Filter for loan amount
    st.write("Interestrate")
    st.write("Total loan costs")
    st.write("Loan term")
    st.write("Property value")
    st.write("Occupancy_type")
    st.write("Total units")
    st.write("Income")
    st.write("Applicant ethnicity")
    st.write("Applicant race")
    st.write("Applicant sex")
    st.write("Applicant age")
    st.write("Co-applicant ethnicity")
    st.write("Co-applicant race")
    st.write("Co-applicant sex")
    st.write("Co-applicant age")

