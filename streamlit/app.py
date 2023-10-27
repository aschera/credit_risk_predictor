#console command: streamlit run app.py

import streamlit as st

st.title("Credit Risk Predictor")

st.header("Project Description")
st.write("Placeholder for description")


#Sidebar
st.sidebar.header("Navigation")
st.sidebar.button("Home")
st.sidebar.button("App")
st.sidebar.button("Interactive Page")