import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle
import os

mlflow.set_experiment("churn-prediction")

# Ensure mlruns exists
os.makedirs("mlruns", exist_ok=True)

df = pd.read_csv("data/processed/train.csv")
X = df.drop("target", axis=1)
y = df["target"]

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)

    acc = model.score(X, y)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    # ✅ Updated
    mlflow.sklearn.log_model(model, name="model")

    os.makedirs("models", exist_ok=True)
    pickle.dump(model, open("models/model.pkl", "wb"))