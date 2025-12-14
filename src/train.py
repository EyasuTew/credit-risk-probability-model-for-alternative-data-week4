# import mlflow
# import mlflow.sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score, classification_report
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# import pandas as pd
# import numpy as np
# import joblib
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Start MLflow
# mlflow.set_experiment("credit_risk_model")
# with mlflow.start_run():
    
#     # Load data
#     df = pd.read_csv("data/processed/credit_features.csv")
#     X = df.drop(['CustomerId', 'default_proxy', 'RFM_score', 'R_score', 'F_score', 'M_score'], axis=1)
#     y = df['default_proxy']
    
#     # Select high-correlation features (threshold 0.1)
#     corr = X.corrwith(y).abs()
#     selected_features = corr[corr > 0.1].index.tolist()
#     X = X[selected_features]
#     logger.info(f"Selected features: {selected_features}")
    
#     # Split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
#     # Pipeline
#     pipe = Pipeline([
#         ('scaler', StandardScaler()),
#         ('model', LogisticRegression(class_weight='balanced', random_state=42))
#     ])
    
#     # Train
#     pipe.fit(X_train, y_train)
    
#     # Predict
#     y_pred_proba = pipe.predict_proba(X_test)[:, 1]
#     y_pred = pipe.predict(X_test)
    
#     auc = roc_auc_score(y_test, y_pred_proba)
#     logger.info(f"AUC: {auc}")
#     print(classification_report(y_test, y_pred))
    
#     # Log metrics
#     mlflow.log_metric("auc", auc)
    
#     # Save model
#     mlflow.sklearn.log_model(pipe, "risk_model")
#     joblib.dump(pipe, "models/risk_model.pkl")
    
#     # Credit score function (300-850, inverse to prob)
#     def prob_to_score(prob):
#         return np.clip(850 - (550 * prob), 300, 850).astype(int)
    
#     joblib.dump(prob_to_score, "models/score_mapper.pkl")
#     mlflow.log_param("score_range", "300-850")
    
#     # Loan optimizer (simple rules)
#     def get_loan_terms(score):
#         if score >= 700:
#             return {'amount': 5000, 'duration_months': 12}
#         elif score >= 600:
#             return {'amount': 3000, 'duration_months': 6}
#         else:
#             return {'amount': 1000, 'duration_months': 3}
    
#     joblib.dump(get_loan_terms, "models/loan_optimizer.pkl")
#     mlflow.log_param("loan_rules", "score-based tiers")

# logger.info("Training complete. Models saved.")

# Updated src/train.py
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
# from src.predict import prob_to_score, get_loan_terms  # Import global functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start MLflow
mlflow.set_experiment("credit_risk_model")
with mlflow.start_run():
    
    # Load data
    df = pd.read_csv("data/processed/credit_features.csv")
    X = df.drop(['CustomerId', 'default_proxy', 'RFM_score', 'R_score', 'F_score', 'M_score'], axis=1)
    y = df['default_proxy']
    
    # Select high-correlation features (threshold 0.1)
    corr = X.corrwith(y).abs()
    selected_features = corr[corr > 0.1].index.tolist()
    X = X[selected_features]
    logger.info(f"Selected features: {selected_features}")
    
    # Save feature list for prediction alignment
    Path("models").mkdir(exist_ok=True)
    joblib.dump(selected_features, "models/feature_list.pkl")
    mlflow.log_param("selected_features", selected_features)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced', random_state=42))
    ])
    
    # Train
    pipe.fit(X_train, y_train)
    
    # Predict
    y_pred_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"AUC: {auc}")
    print(classification_report(y_test, y_pred))
    
    # Log metrics
    mlflow.log_metric("auc", auc)
    
    # Save model
    mlflow.sklearn.log_model(pipe, "risk_model")
    joblib.dump(pipe, "models/risk_model.pkl")
    
    # Save functions (but since global, just reference; no need to dump functions anymore)
    mlflow.log_param("score_range", "300-850")
    mlflow.log_param("loan_rules", "score-based tiers")

logger.info("Training complete. Models saved.")