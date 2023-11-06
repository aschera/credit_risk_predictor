#console command: streamlit run Home.py

import streamlit as st
import subprocess
import time
subprocess.Popen(["python", "../app/app.py"])

st.title("Credit Risk Predictor")

st.header("Project Description")
st.write("Placeholder for description")
