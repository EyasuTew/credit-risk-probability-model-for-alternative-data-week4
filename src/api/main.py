# Updated src/api/main.py for Task 6
# Loads model from MLflow registry; /predict endpoint with Pydantic validation

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.pyfunc
import mlflow
import joblib
from typing import Dict, Any
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
from src.predict import prob_to_score, get_loan_terms  # From earlier
from pathlib import Path

app = FastAPI(title="Credit Risk Model API")

# Load model from MLflow registry (assume staged as 'Production' or version 1)
MODEL_NAME = "CreditRiskModel"
MODEL_VERSION = "1"  # Or "Production" if staged
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

# Load preprocessor and feature list if needed (for alignment)
preprocessor = joblib.load("models/preprocessor.pkl")
selected_features = joblib.load("models/selected_features.pkl")

@app.get("/")
async def root():
    return {"message": "Credit Risk Model API - Ready for predictions"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CustomerFeatures):
    """Predict risk for customer features."""
    try:
        # Validate and convert to DF
        input_df = pd.DataFrame([features.features])
        
        # Align to selected features (reorder/fill missing)
        input_aligned = input_df.reindex(columns=selected_features, fill_value=0)
        
        # Preprocess if needed (but since model is pipeline, pass directly)
        pred_proba = model.predict_proba(input_aligned)[0][1]  # Assume binary, prob of high_risk=1
        is_high_risk = int(pred_proba > 0.5)  # Threshold
        score = prob_to_score(pred_proba)
        terms = get_loan_terms(score)
        
        return PredictionResponse(
            risk_probability=float(pred_proba),
            is_high_risk=is_high_risk,
            credit_score=score,
            recommended_amount=terms['amount'],
            recommended_duration_months=terms['duration_months']
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)