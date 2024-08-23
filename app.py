from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('mushroom_classifier')
cols = ['odor', 'gill-size', 'gill-color', 'cap-color','bruises', 'spore-print-color', 'gill-spacing', 'ring-type', 'stalk-surface-above-ring']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input features from the form
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    
    # Create DataFrame from the input features
    data_unseen = pd.DataFrame([final], columns=cols)
    
    # Get predictions from the model
    prediction = predict_model(model, data=data_unseen, round=0)
    
    # Debugging: Print the prediction DataFrame
    print(prediction.head())  # Check the output structure
    
    # Extract the predicted class
    if 'Label' in prediction.columns:
        predicted_class = prediction['Label'][0]  # Get the class label as a string
    elif 'prediction_label' in prediction.columns:
        predicted_class = prediction['prediction_label'][0]  # Alternative name
    else:
        return render_template('index.html', pred='Error: Prediction output format has changed.')

    return render_template('index.html', pred='Predicted Mushroom Class is {}'.format(predicted_class))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    
    # Debugging: Print the prediction output
    print(prediction.head())  # Check the output structure for the API
    
    # Extract the output
    if 'Label' in prediction.columns:
        output = prediction['Label'][0]
    elif 'prediction_label' in prediction.columns:
        output = prediction['prediction_label'][0]
    else:
        return jsonify({'error': 'Error: Prediction output format has changed.'}), 500
    
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)