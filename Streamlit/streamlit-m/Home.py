#console command: streamlit run Home.py

import streamlit as st
import subprocess
import os

st.title("Loan Status Predictor")

st.header("Project Aims")
st.write("Our project aims to: ")
st.markdown(" <span style='color: #ff6e55;'> Develop a Discrimination-Aware Predictive Model: </span> Our primary focus is on creating a machine learning model that not only predicts an individual's likelihood of obtaining a home mortgage but also addresses potential instances of discrimination.  ", unsafe_allow_html=True)
st.markdown(" <span style='color: #ff6e55;'> Identify Key Features Impacting Loan Approval: </span> Determine which features from the Home Mortgage Disclosure Act dataset are most likely to increase or decrease said likelihood.  ", unsafe_allow_html=True)
st.markdown(" <span style='color: #ff6e55;'> Empower Users with a User-Friendly Application: </span> Build an application where individuals can input their own information and redeive a preliminary assessment of their loan application.  ", unsafe_allow_html=True)


current_directory = os.path.dirname(os.path.abspath(__file__))

relative_path = os.path.join(
    current_directory,
    "..",
    "..",
    "app",
    "app.py"
)
subprocess.run(["python", relative_path])

