from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="Heart Disease Risk Inference API",
    description="REST API serving clinical predictions for cardiac health diagnostics.",
    version="1.0.0"
)

try:
    model = joblib.load("model/heart_disease_model.pkl")
except Exception as e:
    model = None

class PatientMetrics(BaseModel):
    age: int = Field(..., ge=1, le=130, example=52)
    sex: int = Field(..., ge=0, le=1, example=1)
    cp: int = Field(..., ge=0, le=3, example=0)
    trestbps: int = Field(..., ge=50, le=300, example=125)
    chol: int = Field(..., ge=50, le=700, example=212)
    fbs: int = Field(..., ge=0, le=1, example=0)
    restecg: int = Field(..., ge=0, le=2, example=1)
    thalach: int = Field(..., ge=40, le=250, example=168)
    exang: int = Field(..., ge=0, le=1, example=0)
    oldpeak: float = Field(..., ge=0.0, le=10.0, example=1.0)
    slope: int = Field(..., ge=0, le=2, example=2)
    ca: int = Field(..., ge=0, le=4, example=2)
    thal: int = Field(..., ge=0, le=3, example=3)

class DiagnosisResponse(BaseModel):
    prediction: int
    diagnosis: str
    risk_probability: float

@app.get("/")
def health_check():
    return {"status": "online", "model_loaded": model is not None}

@app.post("/predict", response_model=DiagnosisResponse)
def evaluate_cardiac_health(patient: PatientMetrics):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not accessible.")
    
    try:
        input_df = pd.DataFrame([patient.model_dump()])
        prediction = int(model.predict(input_df)[0])
        prob = float(model.predict_proba(input_df)[0][1])

        diagnosis_label = "Positive - High Risk of Heart Disease" if prediction == 1 else "Negative - Low Risk"

        return DiagnosisResponse(
            prediction=prediction,
            diagnosis=diagnosis_label,
            risk_probability=round(prob, 4)
        )
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))