import streamlit as st
import pandas as pd

# ----------------------------------------------------------------#
# Christinas paths.
c_downloaded_data1= 'C:/Users/asche/OneDrive/Dokumenter/repos/Streamlit/streamlit-m/static/1_downloaded_data.csv';
# Markus paths.
# m_downloaded_data1= 'C:/Users/marku/Documents/Programmering/Credit_Risk_Predictor/credit_risk_predictor/Streamlit/streamlit-m/static/1_downloaded_data.csv';
# ----------------------------------------------------------------#

st.header("1 Raw Data")
st.write("""
         The following dataframe shows the raw data loaded from (https://s3.amazonaws.com/cfpb-hmda-public/prod/three-year-data/2019/2019_public_lar_three_year_csv.zip).
         Using Singh et al. (2022) and Gardineer (2019) connected to our dataset we have reduced the number of columns to 33. 
         """)

@st.cache_data()
def load_data():
    df = pd.read_csv(c_downloaded_data1, low_memory=False)
    return df

df=load_data()
st.dataframe(df, hide_index=True)

st.write(""" 
    Gardineer, Grovetta N. 2019. Home mortgage disclosure act: key data fields for full and partial reports. occ.gov/news-issuances/bulletins/2019/bulletin-2019-12.html

    Singh, Arashdeep. Singh, Jashandeep. Khan, Ariba. Gupta, Amar. 2022. Developing a novel fairl-loan classifier through a multi-sensitive debiasing pipeline: dualfair. mdpi.com/2504-4990/4/1/11/htm
""")