# src/prepare.py
import pandas as pd, yaml, sys
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))
df = pd.read_csv(params["data"]["raw_path"])
# minimal cleaning (example): drop NA
df = df.dropna()
train, test = train_test_split(df, test_size=params["data"]["test_size"], random_state=params["model"]["random_state"])
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
