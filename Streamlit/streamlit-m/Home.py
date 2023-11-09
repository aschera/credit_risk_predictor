#console command: streamlit run Home.py

import streamlit as st
import os



st.title("Loan Status Predictor")

st.header("Project Aims")
st.write(""" 
        Our project aims to:
        
         - Develop a predictive machine learning model to assess an individual's likelihood of obtaining a 
         home mortgage loan.  
    
        - Determine which features from the Home Mortgage Disclosure Act dataset are most likely to increase or decrease said likelihood.
        
        - Build an application where individuals can input their own information and redeive a preliminary assessment of their loan application.

        
        """)
