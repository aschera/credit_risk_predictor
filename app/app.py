import logging
import flask
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import pickle
from flask import jsonify
import numpy as np
import shap
import time
import matplotlib
matplotlib.use('Agg')
import os



app = Flask(__name__)

# Define the order of columns
column_order = [
    'loan_type', 'lien_status',
    'open_end_line_of_credit', 'loan_amount', 'combined_loan_to_value_ratio',
    'interest_rate', 'total_loan_costs', 'origination_charges', 'loan_term',
    'negative_amortization', 'interest_only_payment', 'balloon_payment',
    'other_nonamortizing_features', 'property_value', 'occupancy_type',
    'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest',
    'total_units', 'income', 'debt_to_income_ratio', 'applicant_credit_score_type',
    'co_applicant_credit_score_type', 'applicant_ethnicity_1', 'co_applicant_ethnicity_1',
    'applicant_race_1', 'applicant_race_2', 'co_applicant_race_1', 'co_applicant_race_2',
    'applicant_sex', 'co_applicant_sex', 'applicant_age', 'co_applicant_age'
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the pre-trained model
model = pickle.load(open('../model/xgboost_model_not_scaled.pkl', 'rb'))

# Load the SHAP explainer
explainer = shap.TreeExplainer(model)

def shap_plot(data, filename, feature_names):
    shap_values_Model = explainer.shap_values(data)
    
    # Create a SHAP force plot
    p = shap.force_plot(explainer.expected_value, shap_values_Model[0], data[0], matplotlib=True, show=False)
    
    # Generate a unique number, for example, using a timestamp or a random number
    unique_number = str(int(time.time()))  # Using a timestamp as a unique number
    
    image_filename = f'{filename}_{unique_number}.png'  # Append the unique number to the filename
    image_path = f'static/{image_filename}'  # Save the image in the 'static' folder

    # Save the plot as an image
    p.savefig(image_path)

    # Annotate the image with feature names and values using matplotlib
    img = plt.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    annotation_text = ', '.join([f'{feature}: {value}' for feature, value in zip(feature_names, data[0])])
    ax.text(10, 20, annotation_text, fontsize=8, color='black')
    
    plt.savefig(image_path)
    plt.close()

    return image_filename


# Define the order of columns
column_order = [
    'loan_type', 'lien_status',
    'open_end_line_of_credit', 'loan_amount', 'combined_loan_to_value_ratio',
    'interest_rate', 'total_loan_costs', 'origination_charges', 'loan_term',
    'negative_amortization', 'interest_only_payment', 'balloon_payment',
    'other_nonamortizing_features', 'property_value', 'occupancy_type',
    'manufactured_home_secured_property_type', 'manufactured_home_land_property_interest',
    'total_units', 'income', 'debt_to_income_ratio', 'applicant_credit_score_type',
    'co_applicant_credit_score_type', 'applicant_ethnicity_1', 'co_applicant_ethnicity_1',
    'applicant_race_1', 'applicant_race_2', 'co_applicant_race_1', 'co_applicant_race_2',
    'applicant_sex', 'co_applicant_sex', 'applicant_age', 'co_applicant_age'
]
model = pickle.load(open('../model/xgboost_model_not_scaled.pkl', 'rb'))

# Load the SHAP explainer
explainer = shap.TreeExplainer(model)

def shap_force_plot(data, filename, feature_names):
    shap_values_Model = explainer.shap_values(data)
    p = shap.force_plot(explainer.expected_value, shap_values_Model[0], data[0], matplotlib=True, show=False)
    unique_number = str(int(time.time()))
    image_filename = f'{filename}_force_{unique_number}.png'
    image_path = f'static/{image_filename}'  # Save the image in the 'static' folder
    p.savefig(image_path)
    img = plt.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    annotation_text = ', '.join([f'{feature}: {value}' for feature, value in zip(feature_names, data[0])])
    ax.text(10, 20, annotation_text, fontsize=8, color='black')
    plt.savefig(image_path)
    plt.close()
    return image_filename


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    # Initialize an empty list for input features
    init_features = []

    # Populate the list dynamically based on the order of columns
    for column_name in column_order:
        feature_value = request.form.get(column_name)
        if feature_value is not None:
            init_features.append(float(feature_value))

    # Ensure that the list has the same number of values as the column_order
    if len(init_features) != len(column_order):
        return jsonify({"error": f"Input data must contain {len(column_order)} float values"})
    
    # Convert to a 2D array
    init_features_2d = np.array([init_features])  

    # Perform the prediction
    prediction = model.predict([init_features])
    
    # Convert the prediction to a regular Python list
    prediction = prediction.tolist()

    # Set the prediction_text variable
    prediction_text = f"Prediction: {prediction[0]}"

    # compute SHAP values
    explainer = shap.Explainer(model, init_features_2d)
    prediction_shap_values = explainer(init_features_2d)

    # make a plot
    image_name = shap_force_plot(init_features_2d, 'shap-force', column_order)  # Provide a filename (e.g., 'force') as the argument
   
    # Create a SHAP summary plot
    shap_values_Model = explainer.shap_values(init_features_2d)
    shap.summary_plot(shap_values_Model, init_features_2d, feature_names=column_order)

    # Save the plot as an image
    unique_number = str(int(time.time()))
    filename = '_shap_'
    image_filename1 = f'{filename}_summary_{unique_number}.png'
    image_path2 = f'static/{image_filename1}'  # Save the image in the 'static' folder

    plt.savefig(image_path2)

 
    app.logger.info(prediction_shap_values )

        # Extract the required information from the Explanation object
    shap_values = exp.shap_values
    base_values = exp.base_values
    data = exp.data
    feature_names = exp.feature_names

    # Convert the data to a dictionary
    serializable_data = {
        "shap_values": shap_values,
        "base_values": base_values,
        "data": data.tolist(),  # Convert the data array to a list
        "feature_names": feature_names,
    }

    # Now you can return the serializable data
    return jsonify(serializable_data)

    # Return a JSON response with both prediction_text and explanation
    return jsonify({"prediction_text": prediction_text, "image_name": image_name, "image_filename1": image_filename1, "prediction_shap_values": prediction_shap_values[0] })

if __name__ == '__main__':
    app.run(debug=True)
