from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="EdTech Student Pass Prediction API")

# Load model
model = joblib.load("model.pkl")

class StudentData(BaseModel):
    math: float
    physics: float
    chemistry: float

@app.get("/")
def read_root():
    return {"message": "EdTech Student Prediction API is running!"}

@app.post("/predict")
def predict_status(data: StudentData):
    features = np.array([[data.math, data.physics, data.chemistry]])
    prediction = model.predict(features)[0]
    return {
        "input": data.dict(),
        "prediction": str(prediction)
    }
