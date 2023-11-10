import streamlit as st

col1, col2 = st.columns(2, gap= "large")

with col1:
    st.header("Conclusion")
    st.write(""" 
    - Loan predictions are a multivariate problem where different features can have a different impact on the
    outcome.
              
    - All of our columns have a statisticaly significant impact on the outcome of a loan getting rejected or accepted.
    
    - Financial factors appear to notably influence the loan predictions.

    - Demographic factors show a moderate positive correlation with the outcome.
    
    - Debt-to-income ration is the most significant influence.
    """)

    # Results
    # % predictability
    # kolla notebook 15

with col2:
    st.header("Improvements")
    st.markdown(" 1- Improve the model's performance, particularly in the <span style='color: #ff6e55;'>'Rejected' class</span>.", unsafe_allow_html=True)
    st.write("""
             Despite its commendable performance with the "Accepted" class, we have identified issues related to the classification threshold or data imbalance. 
             These challenges contribute to lower precision, recall, and F1-score for the "Rejected" class. 
        """)
    st.markdown(" 2- Expanding the dataset or synthetically balancing the dataset with oversampling techniques <span style='color: #ff6e55;'>(SMOTE)</span>.", unsafe_allow_html=True)
    st.markdown(" 3- Our model only works if there is an applicant and a co-applicant. It could perhaps be expanded to include  <span style='color: #ff6e55;'>solo applicants</span> as well.", unsafe_allow_html=True)