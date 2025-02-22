from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np
from scipy.stats import norm

app = Flask(__name__)

# Load your pre-trained model
model = joblib.load('rf_model.pkl')
threshold = 0.7032966204148201  # Decision threshold

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract values from the form
        features = [
            float(request.form.get('AppendixDiameter')),
            float(request.form.get('ReboundTenderness')),
            float(request.form.get('CoughingPain')),
            float(request.form.get('FreeFluids')),
            float(request.form.get('MigratoryPain')),
            float(request.form.get('BodyTemp')),
            float(request.form.get('KetonesInUrine')),
            float(request.form.get('Nausea')),
            float(request.form.get('WBCCount')),
            float(request.form.get('NeutrophilPerc')),
            float(request.form.get('CRPEntry')),
            float(request.form.get('Peritonitis'))
        ]
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error in input: {e}")
    
    # Create a DataFrame with proper column names
    columns = ['AppendixDiameter', 'ReboundTenderness', 'CoughingPain', 'FreeFluids', 'MigratoryPain',
               'BodyTemp', 'KetonesInUrine', 'Nausea', 'WBCCount', 'NeutrophilPerc', 'CRPEntry', 'Peritonitis']
    data = pd.DataFrame([features], columns=columns)
    
    # Get the predicted probability
    pred_prob = model.predict_proba(data)[0][1]

    # Compute Confidence Interval (95%)
    n = 344  # Assumed sample size for approximation
    z = norm.ppf(0.975)  # Z-score for 95% CI
    margin_error = z * np.sqrt((pred_prob * (1 - pred_prob)) / n)
    
    lower_bound = max(0, pred_prob - margin_error)  # Ensure within [0,1]
    upper_bound = min(1, pred_prob + margin_error)

    confidence_interval = f"({lower_bound:.3f}, {upper_bound:.3f})"

    # Apply threshold for final prediction
    prediction = "Appendicitis" if pred_prob >= threshold else "No Appendicitis"

    return render_template('index.html', 
                           prediction_text=f"Prediction: {prediction}", 
                           confidence_text=f"95% Confidence Interval: {confidence_interval}")

if __name__ == "__main__":
    app.run(debug=True)
