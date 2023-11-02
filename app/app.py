import json
import logging
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib
from flask import request
import numpy as np
import requests
import json
import logging
import numpy as np
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import pickle

app = Flask(__name__)

# Load the pre-trained model
# model_xgboost = joblib.load('../notebooks/Christinas/EDA_stepwise/xgboost_model.pkl')
model = pickle.load(open('../notebooks/Christinas/EDA_stepwise/xgboost_model.pkl', 'rb')) # Load the trained model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        init_features = []

        for x in request.form.values():
            if x.strip().lower() == 'undefined':
                continue  # Skip 'undefined' values
            try:
                float_value = float(x)
                init_features.append(float_value)
            except ValueError:
                # Handle any other non-numeric values as needed
                pass

        logger.info('init_features: %s', init_features)   

        if init_features:
            final_features = [np.array(init_features)]
            prediction = model.predict(final_features)

            logger.info('prediction: %s', prediction)
            #prediction = 'test'

            return render_template('index.html', prediction_text='Prediction: {}'.format(prediction))
        else:
            # Handle the case where no valid numeric values were found
            return render_template('index.html', prediction_text='Invalid input values')

if __name__ == '__main__':
    app.run(debug=True)