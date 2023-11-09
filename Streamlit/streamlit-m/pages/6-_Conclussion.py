import streamlit as st

col1, col2 = st.columns(2, gap= "large")

with col1:
    st.header("Conclussion")
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
    st.write("""
        Whilst the model performs well, particularly for the "Accepted" class, 
        there appears to be some issues with the classification threshold or data 
        imbalance leading to lower precision, recall and F1-score for the "Rejected" class.
        
        As such some of the improvements that can be made are:
        
        - Expanding the dataset or synthetically balancing the dataset with oversampling techniques (SMOTE) 
        
        - Our model only works if there is an applicant and a co-applicant. It could perhaps be expanded to include solo applicants as well. 
        """)
