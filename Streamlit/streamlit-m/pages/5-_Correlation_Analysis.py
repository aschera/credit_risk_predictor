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
import os
# Set relative paths
current_directory = os.path.dirname(os.path.abspath(__file__))

style_path = os.path.join(
    current_directory,
    "..",
    "static",
    "style.css"
)

final_dataset_path = os.path.join(
    current_directory,
    "..",
    "static",
    "final_dataset.csv"
)

raw_dataset_path = os.path.join(
    current_directory,
    "..",
    "static",
    "1_downloaded_data.csv"
)
model_path = os.path.join(
    current_directory,
    "..",
    "static",
    "xgboost_model_not_scaled.pkl"
)
# Riannas paths.
r_style = style_path; 
r_dataset2 = final_dataset_path
r_dataset3 = raw_dataset_path
r_model = model_path
# ---------------------------------------------------------------#

#map
#dataset1 = c_dataset1
#dataset2 = c_dataset2 
#model = c_model

# -------------------------------------------------------------------------#

# Set the page configuration first
#st.set_page_config(layout="wide")

with open(r_style) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)    
st.header(" Correlation and Significance Analysis")


df = pd.read_csv(r_dataset2, low_memory= False)

# ------------------Dataset Statistics------------------------------------#
# Display the summary statistics in Streamlit
st.write("### Summary Statistics:")
st.write(df.describe())
st.write()
# -------------------------------------------------------------------------#
df = pd.read_csv(r_dataset2, low_memory= False)

# Drop the column you want to exclude
column_to_exclude = 'reverse_mortgage'
if column_to_exclude in df:
    df.drop(column_to_exclude, axis=1, inplace=True)

# Convert the data types of the remaining columns as needed
#df = df.astype(dtypes)

# Convert 'action_taken' to a binary variable (0 or 1)
df['action_taken_binary'] = df['action_taken'].apply(lambda x: 1 if x == 3 else 0)

#-----------Correlation Heatmap-------------------------------------------#
    # Calculate the correlation matrix for the binary 'action_taken' variable
correlation_matrix = df.corr()

# Create a heatmap for the binary 'action_taken' variable
st.markdown(" #### Correlation Heatmap")
st.write("""Positive correlations are represented by warmer colors red, indicating that as 
         one variable increases, 'action_taken' is more likely to increase as well.
        Negative correlations are represented by cooler colors blue, indicating that 
         as one variable increases, 'action_taken' is more likely to decrease.If a cell 
         is close to 1 or -1, it suggests a strong linear relationship.""")


plt.figure(figsize=(10, 8))  # Adjust the width and height as needed
sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar=True, linecolor='black')


plt.title("Correlation Matrix (Binary Action Taken)")
st.pyplot()

#-----------------------------------------------------------------------#
#-------------------Chi Square -----------------------------------------#
st.markdown(" #### Chi-squared Test: Categorical Values ")


df = pd.read_csv(r_dataset3, low_memory= False)

new_categorical_columns = df.select_dtypes(include=['object']).columns

from scipy import stats

# Create an empty list to store the results
results_data = []

# Loop through categorical columns
for col in new_categorical_columns:
    if col != 'action_taken':
        crosstab = pd.crosstab(df['action_taken'], df[col])
        chi2, p, _, _ = chi2_contingency(crosstab.values)
        
        # Append results to the list
        results_data.append([col, chi2, p])

# Create a DataFrame from the results
results_df = pd.DataFrame(results_data, columns=['Column', 'Chi-squared', 'P-value'])

# Display the DataFrame as a table
st.table(results_df)



st.write(""" 
         - Chi-Squared Statistic: The Chi-Squared statistic measures the dependence or 
         independence of two categorical variables.  

         - interest rate, total loan costs, origination charges, loan term, 
         property value, total units, debt-to-income ratio, applicant age, and co-applicant 
         age are important factors influencing the action taken on loan applications.""")

#-------------------Data Load-------------------------------------------#
st.set_option('deprecation.showPyplotGlobalUse', False)    
with open(r_model, 'rb') as model_file:
    xgb_model = pickle.load(model_file)


data = pd.read_csv(r_dataset2)

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
    #shap.plots.force(shap_values)
    #st.pyplot()