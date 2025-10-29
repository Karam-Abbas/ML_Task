import os
import pandas as pd, yaml, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

params = yaml.safe_load(open("params.yaml"))
df = pd.read_csv("data/train.csv")

target_col = params["training"]["target"]

# 1️⃣ Drop completely irrelevant or high-cardinality columns
drop_cols = [
    "property_id", "location_id", "page_url", "date_added",
    "agency", "agent"  # optional: often too many unique values
]
df = df.drop(columns=drop_cols, errors="ignore")

# 2️⃣ Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# 3️⃣ Encode categorical columns
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 4️⃣ Train model
model = RandomForestRegressor(
    n_estimators=params["model"]["n_estimators"],
    max_depth=params["model"]["max_depth"],
    random_state=params["model"]["random_state"]
)
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
