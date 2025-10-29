import os
import pandas as pd, yaml, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

params = yaml.safe_load(open("params.yaml"))
df = pd.read_csv("data/train.csv")

target_col = params["training"]["target"]

drop_cols = [
    "property_id", "location_id", "page_url", "date_added",
    "agency", "agent"
]
df = df.drop(columns=drop_cols, errors="ignore")

X = df.drop(columns=[target_col])
y = df[target_col]

# Encode categorical columns and save encoders
encoders = {}
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Train model
model = RandomForestRegressor(
    n_estimators=params["model"]["n_estimators"],
    max_depth=params["model"]["max_depth"],
    random_state=params["model"]["random_state"]
)
model.fit(X, y)

# Save model and encoders
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
joblib.dump(encoders, "models/encoders.joblib")
