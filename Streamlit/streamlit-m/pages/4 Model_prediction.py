import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc, f1_score,
    precision_recall_curve, average_precision_score, precision_score, recall_score, 
)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import plotly.graph_objects as go
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------#
# Christinas paths.
dataset = pd.read_csv('C:/Users/asche/OneDrive/Dokumenter/repos/Streamlit/streamlit-m/static/final_dataset.csv');
model = 'C:/Users/asche/OneDrive/Dokumenter/repos/Streamlit/streamlit-m/static/xgboost_model_not_scaled.pkl';
gridsearch_model = 'C:/Users/asche/OneDrive/Dokumenter/repos/Streamlit/streamlit-m/static/grid_search_xgboost.pkl';
model_comparison = pd.read_csv('C:/Users/asche/OneDrive/Dokumenter/repos/Streamlit/streamlit-m/static/model_comparison.csv');
# ---------------------------------------------------------------#

#-----------import files-------------------------------------------#
with open(model, 'rb') as model_file:
    xgb_model = pickle.load(model_file)

# Load the grid search object
grid_search = joblib.load(gridsearch_model)

# Access the results
grid_results = pd.DataFrame(grid_search.cv_results_)

# ------------------------------DATA preparation-------------------------------------------#

# SMOTE:
# ................................................................#

# Define the minority classes for each column
minority_classes = {
    'applicant_sex': [2.0, 3.0, 6.0],
    'co_applicant_sex': [6.0, 1.0],
    'applicant_race_1': [3.0],
    'co_applicant_race_1': [3.0],
    'applicant_ethnicity_1': [1.0],
    'co_applicant_ethnicity_1': [1.0],
}

# Create a dictionary to store resampled datasets for each column
resampled_datasets = {}

# Create a new DataFrame to store the resampled data
resampled_df = dataset.copy()

# Iterate through the columns and apply SMOTE to each
for column, minority_class in minority_classes.items():
    # Select the specific column
    selected_column = dataset[column].values.reshape(-1, 1)

    # Define y_min based on the minority class for this column
    y_min = [1 if value in minority_class else 0 for value in dataset[column]]

    # Apply SMOTE to the selected column
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled2, y_resampled2 = smote.fit_resample(selected_column, y_min)

    # Ensure that the resampled data has the same number of rows as the original DataFrame
    if len(X_resampled2) > len(dataset):
        X_resampled2 = X_resampled2[:len(dataset)]
        y_resampled2 = y_resampled2[:len(dataset)]

    # Update the resampled data in the new DataFrame
    resampled_df[column] = X_resampled2.flatten()

# train, val, test:
# ................................................................#
X = resampled_df.drop('action_taken', axis=1)
y = resampled_df['action_taken']

# Transform the labels
y = (y == 1).astype(int)
# Map class label 3 to 0
y[y == 3] = 0

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split X_train and y_train into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------------------------------------#
# ------------------------------XGBoost Model Analysis------------------------------------------------#
st.title("XGBoost Model Analysis")
st.write("The XGBoost model is a powerful ensemble learning algorithm widely used in machine learning. It offers exceptional performance and several features that make it a popular choice in predictive modeling.")

# ------------------------------------------------------------------------------#
#---------------------------Model selection-------------------------------------#
st.header('2 Model selection', divider='rainbow')

# Create a style function to highlight the top two values in each numeric column
def highlight_top_two(s):
    is_max = s == s.max()
    is_second_max = s == s.nlargest(2).iloc[-1]
    styles = []
    for i, (max_val, second_max_val) in enumerate(zip(is_max, is_second_max)):
        if max_val:
            styles.append("background-color: #ff6e55")
        elif second_max_val:
            styles.append("background-color: #fff87f")
        else:
            styles.append("")
    return styles

# Apply the style function to numeric columns
numeric_columns = model_comparison.select_dtypes(include=['float64']).columns
styled_df = model_comparison.style.apply(lambda x: highlight_top_two(x), subset=numeric_columns, axis=0)

# Display the styled DataFrame
st.dataframe(styled_df)

st.write("We chose XGBoost ('xgb_model1' in the table) as our preferred model due to its exceptional performance and a set of impressive metrics, including high accuracy (`0.982788`), perfect precision (`1.000`), high recall (`0.969421`), and a balanced F1 Score (`0.984473`). Beyond its standout scores, XGBoost's utilization of gradient boosting, built-in L1 and L2 regularization, and optimization for speed make it a versatile and high-performing choice for diverse machine learning applications. While Decision Tree exhibits similar metrics, XGBoost's ensemble approach with multiple decision trees enhances overall performance and generalization, making it a more reliable choice for predictive modeling.")

# ----------------------------------------------------------------------------#
#---------------------------Hyperparameter Tuning ----------------------------#
st.header('3 Hyperparameter Tuning', divider='rainbow')

st.write("We employed GridSearchCV to discover the optimal hyperparameters for our XGBoost model. After a thorough evaluation, the best combination of hyperparameters was identified. It included a moderate learning rate of `0.2`, a maximum tree depth of `5`, a minimum child weight of `1`,  `8,000` boosting rounds, a `90%` subsample of data, and other settings to enhance the model's performance in binary classification. These hyperparameters were found to minimize the negative log loss and provide the most effective model configuration.")

# ----------------------------------------------------------------------------#
st.subheader('3.3 Gridsearch results', divider='rainbow')
# ----------------------------------------------------------------------------#
st.write("Grisdearch Results:")
st.dataframe(grid_results)
# ----------------------------------------------------------------------------#

# Extract information from the best estimator
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Convert the best estimator to a dictionary
best_estimator_dict = {
    "best_params": best_params,
    "best_score": best_score,
    "best_estimator": grid_search.best_estimator_
}

# Display the best estimator dictionary
st.write("Best Params, Score and Estimator:")
st.write(best_estimator_dict)
# ----------------------------------------------------------------------------#
# Display the feature importance in this mode..
st.write("Model feature importance:")

import xgboost as xgb
# Access the best estimator
best_estimator = grid_search.best_estimator_


# ----------------------------------------------------------------------------#
# ----------------------------------------------------------------------------#

# Extract hyperparameters and their unique combinations
hyperparameters = list(grid_results.columns[:-1])  # Exclude 'mean_test_score'

# Convert dictionaries to tuples for hashing
grid_results_tuples = grid_results[hyperparameters].apply(tuple, axis=1)

# Extract AUC scores and other relevant information
val_accuracies = grid_results['mean_test_score']
val_std = grid_results['std_test_score']
# Create a range of x-values to represent the time (e.g., iterations)
x_values = range(len(val_accuracies))

# Plot the mean validation accuracies over time
acc = plt.figure(figsize=(10, 6))
plt.plot(x_values, val_accuracies, marker='o', linestyle='-', color='#ff6e55', label='Validation Accuracy')
plt.title('Validation Accuracy Over Time')
plt.xlabel('Iteration')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# Plot the error bars
err = plt.figure(figsize=(10, 6))
plt.errorbar(x_values, val_accuracies, yerr=val_std, fmt='none', ecolor='#4e79a7', label='Error Bars')
plt.title('Validation Error Over Time')
plt.xlabel('Iteration')
plt.ylabel('Validation Error')
plt.legend()
plt.grid(True)
plt.show()

col1, col2= st.columns(2)
with col1:
    st.markdown("__Validation accuracies plot:__ ")
    st.pyplot(acc)
with col2:
    st.markdown("__Validation errors plot:__ ")
    st.pyplot(err)


# ----------------------------------------------------------------------------#

st.subheader('3.4 Hyperparameter Tuning Results', divider='rainbow')

# ----------------------------------------------------------------------------#
# Classification Report
st.subheader("3.5 Classification Report")

# Make predictions on the test set
y_predict = (xgb_model.predict(X_test) >= 0.59)

# Calculate classification report
clf = classification_report(y_test, y_predict, labels=[0, 1], output_dict=True)

# Calculate precision and recall on the test set
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)

# Calculate other evaluation metrics
accuracy = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_predict)

# confusion matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['decline', 'accept'])
fig, ax = plt.subplots()  # Create a Matplotlib figure and axis
disp.plot(ax=ax)  # Plot the confusion matrix on the provided axis

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("__Validation Set Results:__")

    st.markdown("**For Class 0 (declined):**")
    st.write(f"Precision: {clf['0']['precision']:.2f}")
    st.write(f"Recall: {clf['0']['recall']:.2f}")
    st.write(f"F1-score: {clf['0']['f1-score']:.2f}")

    st.markdown("**For Class 1 (accepted):**")
    st.write(f"Precision: {clf['1']['precision']:.3f}")
    st.write(f"Recall: {clf['1']['recall']:.3f}")
    st.write(f"F1-score: {clf['1']['f1-score']:.3f}")
with col2:
    st.markdown("__Test Set Results:__")
    # Display Precision, Recall, Accuracy, and F1 Score
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
with col3:
    st.markdown("__Confusion matrix:__")
    st.pyplot(fig)

# ---------------------------------------------------------------------------#
#---------------------------Prediction Testing-----------------------------------#
st.header('7 Prediction Testing', divider='rainbow')

st.write("Describe the results of testing the XGBoost model on the test data.Highlight any insights gained from these predictions.")


# Assuming X_test is your test set
y_probabilities = xgb_model.predict_proba(X_test)[:, 1]
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming y_test is your true labels (ground truth)
fpr, tpr, thresholds = roc_curve(y_test, y_probabilities)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
roc =plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Assuming xgb_model is your trained XGBoost model
feature_importance = xgb_model.feature_importances_

# Create a DataFrame to store feature names and their importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,  # Assuming X is your feature matrix
    'Importance': feature_importance
})

# Sort the features based on importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the top N important features
top_n = 10  # You can adjust this based on how many top features you want to display
top_features = feature_importance_df.head(top_n)

# Plot feature importances
fig, ax = plt.subplots()
xgb.plot_importance(best_estimator, ax=ax, max_num_features=10)  # Plot the top 10 features

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("__Top 10 Important Features::__")
    top_features

with col2:
    st.markdown("__Feature importances plot:__")
    # Display the feature importances plot in your Streamlit app
    st.pyplot(fig)

with col3:
    st.markdown("__ROC curve:__")
    st.pyplot(roc)
# -------------------------------------------------------------------------------#
#---------------------------Model evaluation-------------------------------------#
st.header('8 Model evaluation', divider='rainbow')

st.write("Enable users to input data and obtain predictions using the XGBoost model. Display whether a loan application is approved or denied based on user inputs.")

st.write("Test: Web app testing the predictions.")

st.write("Howto: make sure the app runs before clicking the link.")

web_app_link = '<a href="http://127.0.0.1:5000" target="_blank">link</a>'
st.markdown(web_app_link, unsafe_allow_html=True)



# -------------------------------------------------------------------------------#
#---------------------------Improvements-------------------------------------#





st.header('Improvements', divider='rainbow')
st.write("The numbers suggest that your model is performing well, especially for Class 1 (accepted), where it has perfect precision. However, there might be an issue with the classification threshold or data imbalance that leads to lower precision, recall, and F1-score for Class 0 (declined).")

st.write("You may want to adjust the classification threshold or explore ways to handle class imbalance in your data if improving performance for Class 0 is important.")