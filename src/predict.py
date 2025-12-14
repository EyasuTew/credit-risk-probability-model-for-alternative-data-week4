# import joblib
# import pandas as pd
# import numpy as np
# from pathlib import Path

# def load_models():
#     """Load trained models."""
#     risk_model = joblib.load("models/risk_model.pkl")
#     score_mapper = joblib.load("models/score_mapper.pkl")
#     loan_optimizer = joblib.load("models/loan_optimizer.pkl")
#     return risk_model, score_mapper, loan_optimizer

# def predict_risk(features_df: pd.DataFrame):
#     """Predict for new customer features (DataFrame with selected features)."""
#     risk_model, score_mapper, loan_optimizer = load_models()
    
#     # Assume features are scaled/preprocessed as in training
#     prob = risk_model.predict_proba(features_df)[:, 1][0]
#     score = score_mapper(prob)
#     terms = loan_optimizer(score)
    
#     return {
#         'risk_probability': float(prob),
#         'credit_score': int(score),
#         'recommended_amount': terms['amount'],
#         'recommended_duration_months': terms['duration_months']
#     }

# if __name__ == "__main__":
#     # Example usage
#     sample_features = pd.DataFrame({
#         'Recency': [10], 'Frequency': [5], 'Monetary': [1000],
#         'fraud_rate': [0.01], 'amount_std': [50], 'avg_amount': [200],
#         'unique_products': [3], 'unique_channels': [2]
#     })
#     result = predict_risk(sample_features)
#     print(result)


    # Updated src/predict.py
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Define functions globally for pickling
def prob_to_score(prob: np.ndarray) -> int:
    """Convert risk probability to credit score (300-850 scale)."""
    return np.clip(850 - (550 * prob), 300, 850).astype(int)

def get_loan_terms(score: int) -> Dict[str, int]:
    """Rule-based optimal loan amount and duration based on score."""
    if score >= 700:
        return {'amount': 5000, 'duration_months': 12}
    elif score >= 600:
        return {'amount': 3000, 'duration_months': 6}
    else:
        return {'amount': 1000, 'duration_months': 3}

def load_models():
    """Load trained models and artifacts."""
    Path("models").mkdir(exist_ok=True)  # Ensure dir exists
    
    risk_model = joblib.load("models/risk_model.pkl")
    score_mapper = prob_to_score  # Use global function
    loan_optimizer = get_loan_terms  # Use global function
    feature_list = joblib.load("models/feature_list.pkl") if Path("models/feature_list.pkl").exists() else None
    
    return risk_model, score_mapper, loan_optimizer, feature_list

def predict_risk(features_df: pd.DataFrame):
    """Predict for new customer features (DataFrame with selected features)."""
    risk_model, score_mapper, loan_optimizer, feature_list = load_models()
    
    # Align features to training columns
    if feature_list is not None:
        # Reorder and fill missing with 0 (or handle as needed)
        features_df = features_df.reindex(columns=feature_list, fill_value=0)
    else:
        # Fallback: assume input matches exactly
        pass
    
    # Ensure numeric
    features_df = features_df.select_dtypes(include=[np.number])
    
    # Predict
    prob = risk_model.predict_proba(features_df)[:, 1][0]
    score = score_mapper(prob)
    terms = loan_optimizer(score)
    
    return {
        'risk_probability': float(prob),
        'credit_score': int(score),
        'recommended_amount': terms['amount'],
        'recommended_duration_months': terms['duration_months']
    }

if __name__ == "__main__":
    # Example usage
    sample_features = pd.DataFrame({
        'Recency': [10.0], 'Frequency': [5], 'Monetary': [1000.0],
        'fraud_rate': [0.01], 'amount_std': [50.0], 'avg_amount': [200.0],
        'unique_products': [3], 'unique_channels': [2]
    })
    result = predict_risk(sample_features)
    print(result)