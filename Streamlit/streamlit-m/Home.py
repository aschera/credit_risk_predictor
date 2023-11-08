#console command: streamlit run Home.py

import streamlit as st
import subprocess
import time
subprocess.Popen(["python", "../app/app.py"])

st.title("Credit Risk Predictor")

st.header("Project Description")
st.write(""" Machine learning models are increasingly being applied across various domains, 
         potentially impacting people's lives significantly. Our project aims to develop a 
         predictive machine learning model to assess an individual's likelihood of obtaining a 
         home mortgage loan. We leverage the publicly available dataset from the Home Mortgage 
         Disclosure Act (HMDA), which serves as a valuable resource containing data on the 
         approval and denial of home mortgage loans in the United States, including more than 
         21 different factors such as gender, race, and ethnicity.""")

st.write(""" We built a user-friendly application where individuals can input their 
         information and receive a preliminary assessment of their loan application. 
         The primary goal is to ensure that the model is fair and free from biases, which is 
         especially crucial when users engage with financial institutions like banks to uphold 
         transparency and equity.""")

st.write(""" Furthermore, we enhance our application by providing valuable recommendations to 
         borrowers. These recommendations are designed to help borrowers take actions to improve 
         their chances of securing a home mortgage loan. Our overarching objective is to promote 
         fairness and transparency throughout the process and empower individuals to make 
         well-informed financial decisions.""")

