import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import logging


app = Flask(__name__) # Initialize the flask App
model = pickle.load(open('model.pkl', 'rb')) # Load the trained model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # UI rendering the results
    # Retrieve values from a form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features) # Make a prediction

    app.logger.info('Received data: %s', init_features)
    app.logger.info('final_features: %s', final_features)
        
        

    return render_template('index.html', prediction_text='Predicted Species: {}'.format(prediction)) # Render the predicted result


if __name__ == "__main__":
    app.run(debug=True)