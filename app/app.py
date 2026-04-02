from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("models/model.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "MLOps Model Running 🚀"}

@app.post("/predict")
def predict(data: list):
    prediction = model.predict([data])
    return {"prediction": prediction.tolist()}