import streamlit as st


st.header("Steps taken during data processing")

st.write("""
    -  1,000,000 rows are downloaded from the dataset: https://s3.amazonaws.com/cfpb-hmda-public/prod/three-year-data/2019/2019_public_lar_three_year_csv.zip with conditions set on what columns should be used.
    -  Rows without any missing data are filtered out as well as any rows where 'action_taken' is not equal to 1 or 3 (accepted or rejected, respectively), these are saved in a csv file which will be used in the next steps of the data processing. 
    -  Next we filter the dataset for the sex of the applicants and co-applicants. We remove anything other than male, female or both (including if there is no co-applicant).
    -  We also filter race and ethnicity so that we include only 'White' or 'Black or African American' for race and 'Hispanic or Latino' & 'Not Hispanic or Latino' for ethnicity. The same as Singh et al. (2022).
    -  The next step taken was encoding any categorical data (ie. debt to income ratio, applicant and co-applicant age, interest rates, combined loan to value ratio, total loan costs, origination charges, loan_term, and total_units)

    """)