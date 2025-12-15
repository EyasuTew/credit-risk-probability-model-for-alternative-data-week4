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

# Updated src/api/pydantic_models.py for Task 6
# Assumes dynamic features; for demo, use dict for flexibility (or list specific)

from pydantic import BaseModel
from typing import Dict, Any

class CustomerFeatures(BaseModel):
    """Input features matching model (e.g., Recency, Frequency, etc.)."""
    features: Dict[str, Any]  # Flexible dict for varying selected features

class PredictionResponse(BaseModel):
    """Output: Risk probability and details."""
    risk_probability: float
    is_high_risk: int
    credit_score: int  # Optional: From prob mapping
    recommended_amount: int
    recommended_duration_months: int