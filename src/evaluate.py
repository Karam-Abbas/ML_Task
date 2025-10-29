# src/evaluate.py
import pandas as pd, yaml, joblib, os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load params and model
params = yaml.safe_load(open("params.yaml"))
model = joblib.load("models/model.joblib")

# Load test data
test = pd.read_csv("data/test.csv")
target_col = params["training"]["target"]

# Drop same irrelevant columns used in training
drop_cols = [
    "property_id", "location_id", "page_url", "date_added",
    "agency", "agent"
]
test = test.drop(columns=drop_cols, errors="ignore")

# Separate features and target (if available)
X_test = test.drop(columns=[target_col], errors="ignore")
y_test = test[target_col] if target_col in test.columns else None

# Encode categorical columns the same way
for col in X_test.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col].astype(str))

# Predict
pred = model.predict(X_test)

# Evaluate only if true labels exist
if y_test is not None:
    rmse = np.sqrt(mean_squared_error(y_test, pred))
else:
    rmse = None

# Save metrics
with open("metrics.txt", "w") as f:
    if rmse is not None:
        f.write(f"RMSE: {rmse}\n")
    else:
        f.write("Predictions generated (no true labels found)\n")

# Save predictions
os.makedirs("predictions", exist_ok=True)
pd.DataFrame({"Predicted": pred}).to_csv("predictions/output.csv", index=False)
