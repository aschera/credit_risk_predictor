import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plost
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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

# ----------------------------------------------------------------#
# Riannas paths.
dataset1 = pd.read_csv('/Users/riannaaalto/Documents/GitHub/Streamlit/static/10_all_numerical_32bit.csv');
dataset2 = pd.read_csv('/Users/riannaaalto/Documents/GitHub/Streamlit/static/final_dataset.csv')
model = '/Users/riannaaalto/Documents/GitHub/Streamlit/static/xgboost_model_not_scaled.pkl';

# ---------------------------------------------------------------#


# Set the page configuration first
st.set_page_config(layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# =============================================================================
# Creating the streamlit application. #
# =============================================================================

# Creating a navigation menu with three different sections the user can choose. 
nav = st.sidebar.radio("Navigation Menu",["Home", "Data & Modelling", "Mortgage Predictor"])

if nav == "Home":
    st.title("About The Project")
    st.header("Introduction")
    st.write("""Machine learning models are increasingly being applied across 
             various domains, potentially impacting people's lives significantly. 
             Our project aims to develop a predictive machine learning model to assess an 
             individual's likelihood of obtaining a home mortgage loan. We leverage the publicly 
             available dataset from the Home Mortgage Disclosure Act (HMDA), which serves as a valuable 
             resource containing data on the approval and denial of home mortgage loans in the United States, 
             including more than 21 different factors such as gender, race, and ethnicity.
            """)
    
    st.write("""We are building a user-friendly application where individuals can 
             input their information and receive a preliminary assessment of their loan application. The primary goal 
             is to ensure that the model is fair and free from biases, which is especially crucial when users engage 
             with financial institutions like banks to uphold transparency and equity.""")
    
    st.write("""Furthermore, we enhance our application by providing valuable recommendations to borrowers. 
             These recommendations are designed to help borrowers take actions to improve their chances 
             of securing a home mortgage loan. Our overarching objective is to promote fairness and transparency 
             throughout the process and empower individuals to make well-informed financial decisions.""")
    
    st.header("Data")
    st.write("""In our study, we utilized the 2019 The Home Mortgage Disclosure Act (HMDA) Loan Application dataset, which mandated the collection, 
             reporting, and public disclosure of mortgage-related information by various financial institutions. 
             Initially comprising 1,000,000 entries with 97 features, rigorous data cleaning and feature 
             correlation testing reduced our dataset to a refined set of 54,832 entries with 33 features. 
             These features include 13 float64 and 20 int64 data types, providing a more streamlined and relevant dataset for our analysis.
            """)
    st.write("""You can check out the dataset at The Home Mortgage Disclosure Act website:
    [https://ffiec.cfpb.gov/data-publication/three-year-national-loan-level-dataset/2019](https://ffiec.cfpb.gov/data-publication/three-year-national-loan-level-dataset/2019) .""")
    
    st.header("Mortgage Predictor Web Application")
    st.write("""The Mortgage Prediction Web Application is powered by a Flask app, where users enter the 
             required information. It provides a preliminary assessment of their home loan eligibility based 
             on the provided data. The application also offers recommendations to assist borrowers in enhancing their prospects of obtaining a home loan.
            """)

if nav == "Data & Modelling":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Data & Machine Learning Modelling")
    st.write("In this section, we will look at the data and model it using XGBoost.")
    
    # Section to make predictions and display confusion matrix
    st.header("XGBoost Model")
    
    # Display your data
    st.subheader("Raw Data")
    # =============================================================================
    # Loading Data and Modelling it. 
    # =============================================================================

    # Load the model
    with open(model, 'rb') as model_file:
        model = pickle.load(model_file)

    # Load your data (example using a pandas DataFrame)
    data = pd.read_csv(dataset2)

    # Suppress FutureWarnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    X = data.drop('action_taken', axis=1)
    y = data['action_taken']

    # Replace values in the target variable 'y'
    y = y.replace(3, 0)  # Replace 3 with 1
    # Replace values in the target variable 'y'
    y = y.replace(1, 1)  # Replace 3 with 1

    X 

    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Split X_train and y_train into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    X_train.info()
    
    st.header("Result Summary")
    st.write("""

High Precision and Recall: A high precision (0.9727) and recall (0.9991) indicate that the model is making accurate positive predictions and correctly identifying almost all actual positive instances. This suggests that the model generalizes well to unseen data.

High Accuracy: An accuracy of 0.9874 means the model is making correct predictions for a large majority of instances, which is another indicator of good generalization.

High F1 Score: The F1 score, which balances precision and recall, is also high at 0.9857, further indicating the model's strong performance.

""")

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

    # Calculate precision and recall on the test set
    precision2 = precision_score(y_test, y_predict2)
    recall2 = recall_score(y_test, y_predict2)
    

    # Print test set results for the second model
    print("Test set results:")
    print(f"Precision: {precision2:.4f}, Recall: {recall2:.4f}")
    

    # Calculate other evaluation metrics
    accuracy2 = accuracy_score(y_test, y_predict2)
    f12 = f1_score(y_test, y_predict2)

    # Calculate the confusion matrix
    cm2 = confusion_matrix(y_test, y_predict2)

    # Print other evaluation metrics
    print("Test set results:")
    print(f"Precision: {precision2:.4f}")
    print(f"Recall: {recall2:.4f}")
    print(f"Accuracy: {accuracy2:.4f}")
    print(f"F1 Score: {f12:.4f}")
    print("Confusion Matrix:")


    # Calculate the confusion matrix
    cm2 = confusion_matrix(y_test, y_predict2)
    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues", linewidths=0.5, linecolor="black", square=True, cbar=False,
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["True 0", "True 1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - non-scaled dataset")
    plt.show()
    st.header("Confustion Matrix")
    st.pyplot()

    # Make predictions on the test set
    y_prob2 = xgb_model2.predict_proba(X_test)[:, 1]

    # Calculate and print the classification report
    report = classification_report(y_test, (y_prob2 > 0.5).astype(int), zero_division=1)
    #print(report)

    # Calculate AUC-ROC
    oc_auc = roc_auc_score(y_test, y_prob2)
    #print("AUC-ROC:", roc_auc)

    # Calculate Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_prob2, pos_label=1)
    pr_auc = auc(recall, precision)
    #print("Precision-Recall AUC:", pr_auc)

    # Calculate the False Positive Rate (FPR) and True Positive Rate (TPR)
    fpr, tpr, _ = roc_curve(y_test, y_prob2, pos_label=1)

    # Calculate AUC-ROC
    roc_auc = auc(fpr, tpr)

    # Create the ROC curve plot
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    st.header("ROC Curve Plot")
    st.pyplot()

    # Load the SHAP explainer
    explainer = shap.TreeExplainer(xgb_model2)

    # Extract one row from X_train (change the index as needed)
    row_index = 0
    sample_row = X_train.iloc[[row_index]]  # Wrap it in double brackets to make it a DataFrame

    # Use the explainer to generate explanations for the sample row
    explanation = explainer.shap_values(sample_row)

    # The SHAP explanation you provided represents the SHAP values for the features of a specific sample in your dataset. 
    # Each value in the array corresponds to the impact or contribution of a feature to the prediction for that sample.

    print("SHAP Explanation:")
    print(explanation)

    # Compute SHAP values for the dataset (X_train is the dataset)
    shap_values = explainer(X_train)

    # Visualize the first prediction's explanation as a waterfall plot
    shap.plots.waterfall(shap_values[0])
    st.header("First Prediction")
    st.pyplot()

    # List of feature indices or names you want to analyze
    feature_indices = [6,7,8]  # Replace with the indices or names of the features you want to analyze

    # Create scatter plots for the selected features
    for feature_index in feature_indices:
        shap.plots.scatter(shap_values[:, feature_index])
        st.header("Scattered Plots for Selected Features")
        st.pyplot()

    # summarize the effects of all the features
    st.header("Summary Of the Effects of all the Features")
    shap.plots.beeswarm(shap_values)
    st.pyplot()
    shap.plots.bar(shap_values)
    st.pyplot()
    #shap.plots.force(shap_values)
    #st.pyplot()

if nav == "Mortgage Predictor":
    st.title("Mortgage Predictor Application")

    import subprocess
    # Run the Flask app as a separate process
    subprocess.Popen(["python", "app.py"])

    # Provide a link to access the Flask app
    st.write("Access Mortgage Predictor at: [http://127.0.0.1:5000](http://127.0.0.1:5000)")
    st.components.v1.iframe("http://127.0.0.1:5000", width=800, height=600)  # Replace the URL with your Flask app's URL

    # Define the Flask app's URL for prediction
    flask_url = "http://127.0.0.1:5000/predict"  # Replace with the correct URL if hosted elsewhere

    if st.button("Predict"):
        # Send a POST request to the Flask app
        response = requests.post(flask_url, json=data)

        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction_text"]

            if prediction == "APPROVED":
                st.write("Your Loan will likely be:")
                st.subheader("APPROVED!")
                st.subheader("Recommendations")
                st.write("Please contact your lender for further information and assessment.")
                for rec in recommendations:
                    st.write(rec)
            elif prediction == "DENIED":
                st.write("Your Loan will likely be:")
                st.subheader("DENIED")
                st.subheader("Recommendations")
                st.write("Boost your credit score.")
                st.write("Please find a co-applicant if you don't have one.")
                st.write("Pay down high-interest debts.")
                st.write("Apply for a lower loan amount.")
            else:
                st.write("Failed to get a response from the Flask app.")
        else:
            st.write("Failed to get a response from the Flask app.")





    
    
    
    
    
    
