from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from ..predict import predict_risk

app = FastAPI(title="Credit Risk API")

@app.post("/predict")
async def predict(features: dict):
    """Predict risk for a customer."""
    df = pd.DataFrame([features])
    result = predict_risk(df)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)