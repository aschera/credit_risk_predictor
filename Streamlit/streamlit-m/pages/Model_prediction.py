import streamlit as st

st.title("Model Prediction")

st.write("Enable users to input data and obtain predictions using the XGBoost model. Display whether a loan application is approved or denied based on user inputs.")

st.write("Test: Web app testing the predictions.")

st.write("Howto: make sure the app runs before clicking the link.")

web_app_link = '<a href="http://127.0.0.1:5000" target="_blank">link</a>'
st.markdown(web_app_link, unsafe_allow_html=True)
