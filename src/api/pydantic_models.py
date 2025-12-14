from pydantic import BaseModel
from typing import List

class CustomerFeatures(BaseModel):
    Recency: float
    Frequency: int
    Monetary: float
    fraud_rate: float
    amount_std: float
    avg_amount: float
    unique_products: int
    unique_channels: int

class PredictionResponse(BaseModel):
    risk_probability: float
    credit_score: int
    recommended_amount: int
    recommended_duration_months: int