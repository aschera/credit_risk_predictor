import streamlit as st
import pandas as pd
import numpy as np

data = {
    'Feature': [
        'Performance',
        'Regularization',
        'Gradient Boosting',
        'Flexibility',
        'Parallel and Distributed Computing',
        'Feature Importance',
        'Handling Missing Values',
        'Tree Pruning',
        'Speed'
    ],
    'Description': [
        'high performance and accuracy.',
        'built-in L1 (Lasso) and L2 (Ridge) regularization.',
        'gradient boosting algorithm that combines multiple models.',
        'classification and regression tasks with customizable settings.',
        'supports parallel and distributed computing, making it suitable for large datasets and multi-core CPUs.',
        'feature importance scores for feature selection and interpretation.',
        'handles missing values, improving performance in datasets with missing data.',
        'uses tree pruning to prevent overfitting by removing unimportant branches in decision trees.',
        'optimized for speed, making it one of the fastest implementations of gradient boosting.'
    ]
}

df = pd.DataFrame(data)

st.title("Model Prediction")

st.header('Model selection', divider='rainbow')

col1, col2, col3 = st.columns(3)

with col1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg")




st.header('XGBoost (eXtreme Gradient Boosting)', divider='rainbow')



st.write("We selected XgBoost because of the following features.")


st.table(df)





st.subheader('Gridsearch to find hyperparameters', divider='rainbow')
code = '''
# Add early stopping
early_stopping_rounds = 3

xgb_model1 = xgb.XGBClassifier(early_stopping_rounds=early_stopping_rounds)

parameters = {
    'nthread': [4],
    'objective': ['binary:logistic'],
    'learning_rate': [0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.5, 0.7, 0.9],
    'colsample_bytree': [0.7],
    'n_estimators': [8000, 9000, 10000],
    'missing': [-999],
    'seed': [1337],
    'eval_metric': ['logloss'],
    'reg_alpha': [0.0, 0.1, 0.2],  # Test L1 regularization (0.0 means no L1 regularization)
    'reg_lambda': [0.0, 0.1, 0.2]  # Test L2 regularization (0.0 means no L2 regularization)
}

# Use 3-fold cross-validation
clf = GridSearchCV(xgb_model1, parameters, n_jobs=3,
                   cv=KFold(n_splits=3, shuffle=True, random_state=random_state),
                   scoring='neg_log_loss',  
                   verbose=4, refit=True)


# Implement early stopping
eval_set = [(X_val, y_val)]
clf.fit(X_train, y_train, eval_set=eval_set)
'''
st.code(code, language='python')

st.subheader(' Best hyperparameters and final model', divider='rainbow')

code = '''
# Define your hyperparameters
params = {'colsample_bytree': 0.7,
 'eval_metric': 'logloss',
 'learning_rate': 0.2,
 'max_depth': 5,
 'min_child_weight': 1,
 'missing': -999,
 'n_estimators': 8000,
 'nthread': 4,
 'objective': 'binary:logistic',
 'seed': 1337,
 'subsample': 0.9}
'''
st.code(code, language='python')







st.subheader('Test: Web app testing the predictions.', divider='rainbow')

st.write("Enable users to input data and obtain predictions using the XGBoost model. Display whether a loan application is approved or denied based on user inputs.")

st.write("Test: Web app testing the predictions.")

st.write("Howto: make sure the app runs before clicking the link.")

web_app_link = '<a href="http://127.0.0.1:5000" target="_blank">link</a>'
st.markdown(web_app_link, unsafe_allow_html=True)

