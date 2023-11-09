import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import plotly.express as px
import pickle
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import shap
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
from scipy.stats import chi2_contingency


# ----------------------------------------------------------------#
# Christinas paths.
c_style = 'C:/Users/asche/OneDrive/Dokumenter/repos/Streamlit/streamlit-m/static/style.css';
c_dataset1 = 'C:/Users/asche/OneDrive/Dokumenter/repos/Streamlit/streamlit-m/static/10_all_numerical_32bit.csv'
c_dataset2 = 'C:/Users/asche/OneDrive/Dokumenter/repos/Streamlit/streamlit-m/static/final_dataset.csv'
c_model = 'C:/Users/asche/OneDrive/Dokumenter/repos/Streamlit/streamlit-m/static/xgboost_model_not_scaled.pkl';

# Riannas paths.
r_style = '/Users/riannaaalto/Documents/GitHub/Streamlit/streamlit-m/static/style.css';
r_dataset1 = '/Users/riannaaalto/Documents/GitHub/Streamlit/static/10_all_numerical_32bit.csv'
r_dataset2 = '/Users/riannaaalto/Documents/GitHub/Streamlit/static/final_dataset.csv'
r_model = '/Users/riannaaalto/Documents/GitHub/Streamlit/static/xgboost_model_not_scaled.pkl';
# ---------------------------------------------------------------#

#map
dataset1 = c_dataset1
dataset2 = c_dataset2 
model = c_model

# ---------------------------------------------------------------#

# Set the page configuration first
st.set_page_config(layout="wide")

with open(c_style) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ------------------------------------------------------------------------#
#-----------Correlation Heatmap-------------------------------------------#

st.header(" Correlation and Significance Analysis")
st.set_option('deprecation.showPyplotGlobalUse', False)    

df = pd.read_csv(dataset1, low_memory= False)

# Drop the column you want to exclude
column_to_exclude = 'reverse_mortgage'
if column_to_exclude in df:
    df.drop(column_to_exclude, axis=1, inplace=True)

# Convert the data types of the remaining columns as needed
#df = df.astype(dtypes)

# Convert 'action_taken' to a binary variable (0 or 1)
df['action_taken_binary'] = df['action_taken'].apply(lambda x: 1 if x == 3 else 0)

    # Calculate the correlation matrix for the binary 'action_taken' variable
correlation_matrix = df.corr()

# Create a heatmap for the binary 'action_taken' variable
st.markdown(" #### Correlation Heatmap")
st.write("""Positive correlations are represented by warmer colors red, indicating that as 
         one variable increases, 'action_taken' is more likely to increase as well.
        Negative correlations are represented by cooler colors blue, indicating that 
         as one variable increases, 'action_taken' is more likely to decrease.If a cell 
         is close to 1 or -1, it suggests a strong linear relationship.""")


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar=True, linecolor='black')
plt.title("Correlation Matrix (Binary Action Taken)")
st.pyplot()



#-----------------------------------------------------------------------#
#-------------------Chi Square -----------------------------------------#
st.markdown(" #### Chi-squared Test: Categorical Values ")
st.write(""" """)
from scipy.stats import chi2_contingency
# Display a PNG image
st.image('/Users/riannaaalto/Documents/GitHub/Streamlit/static/Chi-square.png', caption='Chi-squared Test', use_column_width=True)

st.write("""Chi-Squared Statistic: The Chi-Squared statistic measures the dependence or 
         independence of two categorical variables. In your case, it assesses the 
         relationship between each categorical column and "Action Taken." 
         If the Chi-Squared statistic is significantly high, it indicates that the 
         categorical column is strongly associated with "Action Taken." This is important 
         because it suggests that the specific category within the column influences the action 
         taken. P-Value: The p-value is used to determine the statistical 
         significance of the Chi-Squared test. If the p-value is very low (typically less 
         than 0.05), it suggests that the relationship between the categorical column and "Action Taken" 
         is statistically significant. In other words, the results are unlikely to be due to 
         random chance.""")


#-------------------T-test or ANOVA-------------------------------------#
st.markdown(" #### T-Test or ANOVA: Categorical Values ")
st.write(""" """)
from scipy.stats import chi2_contingency
# Display a PNG image
st.image('/Users/riannaaalto/Documents/GitHub/Streamlit/static/t-test.png', caption='T-test/ ANOVA Test', use_column_width=True)
st.write("""T-Statistic: The T-statistic measures the difference in means between two groups, 
         in this case, "Action Taken" and the numerical column. A higher T-statistic indicates 
         a larger difference between the means. If the T-statistic is positive, it suggests that 
         the numerical column has a higher mean for certain categories of "Action Taken." If it's
          negative, it means the numerical column has a lower mean for certain categories.""")


#-------------------Data Load-------------------------------------------#
st.set_option('deprecation.showPyplotGlobalUse', False)    
with open(model, 'rb') as model_file:
    xgb_model = pickle.load(model_file)


data = pd.read_csv(dataset2)

# -----------------DATA split-------------------------------------------#


X = data.drop('action_taken', axis=1)
y = data['action_taken']

    # Replace values in the target variable 'y'
y = y.replace(3, 0)  # Replace 3 with 1
    # Replace values in the target variable 'y'
y = y.replace(1, 1)  # Replace 3 with 1
 

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Split X_train and y_train into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



    # Define your hyperparameters
params = {
        'colsample_bytree': 0.7,
        'eval_metric': 'logloss',  # Specify eval_metric during initialization
        'learning_rate': 0.3,
        'max_depth': 3,
        'min_child_weight': 1,
        'missing': -999,
        'n_estimators': 8000,
        'nthread': 4,
        'objective': 'binary:logistic',
        'seed': 1337,
        'subsample': 0.9
        }
    

    # Define and initialize the XGBoost classifier
xgb_model2 = XGBClassifier(**params, early_stopping_rounds=3)

    # Set the validation dataset for early stopping
eval_set2 = [(X_val, y_val)]

    # Train the model and monitor early stopping
xgb_model2.fit(X_train, y_train, eval_set=eval_set2, verbose=True)

    # Make predictions on the test set
y_predict2 = (xgb_model2.predict_proba(X_test)[:, 1] >= 0.59)

    # Define and initialize the XGBoost classifier
xgb_model2 = XGBClassifier(**params, early_stopping_rounds=3)

    # Set the validation dataset for early stopping
eval_set2 = [(X_val, y_val)]

    # Train the model and monitor early stopping
xgb_model2.fit(X_train, y_train, eval_set=eval_set2, verbose=True)

    # Make predictions on the test set
y_predict2 = (xgb_model2.predict_proba(X_test)[:, 1] >= 0.59)

#------------------SHAPS Visualisation--------------------------------

# Load the SHAP explainer
explainer = shap.TreeExplainer(xgb_model2)

    # Extract one row from X_train (change the index as needed)
row_index = 0
sample_row = X_train.iloc[[row_index]]  # Wrap it in double brackets to make it a DataFrame

    # Use the explainer to generate explanations for the sample row
explanation = explainer.shap_values(sample_row)


    # Compute SHAP values for the dataset (X_train is the dataset)
shap_values = explainer(X_train)

    # List of feature indices or names you want to analyze
feature_indices = [6,7,8]  # Replace with the indices or names of the features you want to analyze

    # summarize the effects of all the features
st.markdown(" ### Summary Of the Effects of all the Features")
shap.plots.beeswarm(shap_values)
st.pyplot()
shap.plots.bar(shap_values)
st.pyplot()
    #shap.plots.force(shap_values)
    #st.pyplot()