import streamlit as st
import subprocess
import time

st.title("Loan Predictor")

st.write("This app can be used to predict the likeliness of your loan application being accepted or rejected. Note this is not a guaranteed outcome.")
st.write("Input your own data or use our predetermined 'Applicants' to view the results.")
start_button = st.button("Start loan predictor app")

if start_button:
    subprocess.Popen(["python", "../app/app.py"])
    launch_text = st.text("Launching Loan Predictor")
    embedded_app = st.markdown('<iframe src="http://127.0.0.1:5000" width="800" height="600"></iframe>', unsafe_allow_html=True)
    if embedded_app:
        time.sleep(3)
        launch_text.empty()
