from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import subprocess
import numpy as np
app = Flask(__name__)

# Go to project root before pulling
os.chdir(os.path.dirname(os.path.dirname(__file__)))

subprocess.run(["dvc", "pull", "models/model.joblib"], check=True)

# Then back to app dir for Flask
os.chdir(os.path.join(os.getcwd(), "app"))

# Load model and encoders
model = joblib.load("../models/model.joblib")
encoders = joblib.load("../models/encoders.joblib")

@app.route('/')
def home():
    return "🏠 House Price Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])

        # Apply encoders for categorical columns
        for col, le in encoders.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].map(lambda x: x if x in le.classes_ else "unknown")
                # Add unseen labels handling
                if "unknown" not in le.classes_:
                    le.classes_ = np.append(le.classes_, "unknown")
                input_df[col] = le.transform(input_df[col].astype(str))

        # Convert remaining columns to numeric if possible
        for col in input_df.columns:
            try:
                input_df[col] = input_df[col].astype(float)
            except:
                pass

        # Predict
        pred = model.predict(input_df)[0]
        return jsonify({"predicted_price": float(pred)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
