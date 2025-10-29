from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os
import yaml

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "model.joblib")
model = joblib.load(model_path)

# Load params to get feature information
params_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "params.yaml")
with open(params_path, "r") as f:
    params = yaml.safe_load(f)

# Load training data to get feature names
train_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "prepared_data.csv")
train_data = pd.read_csv(train_data_path)
feature_names = train_data.drop("price", axis=1).columns.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # One-hot encode categorical variables
        df = pd.get_dummies(df, drop_first=True)
        
        # Ensure all expected columns are present
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training data
        df = df[feature_names]
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Return prediction
        return jsonify({'prediction': float(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)