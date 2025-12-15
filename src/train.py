# Updated src/train.py for Task 5: Model Training and Tracking
# Includes multi-model training, hyperparam tuning, MLflow logging, evaluation, and registry

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow experiment
mlflow.set_experiment("credit_risk_model_task5")

# Models and their param grids for tuning
MODELS = {
    'logistic_regression': {
        'model': LogisticRegression(class_weight='balanced', random_state=42),
        'params': {
            'model__C': [0.1, 1, 10],
            'model__solver': ['liblinear', 'lbfgs']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(class_weight='balanced', random_state=42),
        'params': {
            'model__n_estimators': [50, 100],
            'model__max_depth': [3, 5, None]
        }
    }
}

def load_data(processed_path: str):
    """Load engineered features."""
    df = pd.read_csv(processed_path)
    X = df.drop(['CustomerId', 'is_high_risk'], axis=1)  # Assume all else are features
    y = df['is_high_risk']
    return X, y

def train_and_evaluate_model(model_name, model_config, X_train, X_test, y_train, y_test):
    """Train, tune, evaluate, and log model with MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Pipeline with scaler
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model_config['model'])
        ])
        
        # Hyperparam tuning
        grid_search = GridSearchCV(pipe, model_config['params'], cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info(f"{model_name} Metrics: {metrics}")
        print(classification_report(y_test, y_pred))
        
        # Log to MLflow
        mlflow.log_params(grid_search.best_params_)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_param("model_type", model_name)
        
        # Log model
        mlflow.sklearn.log_model(best_model, "model")
        
        # Save locally
        Path("models").mkdir(exist_ok=True)
        joblib.dump(best_model, f"models/{model_name}_best.pkl")
        
        return best_model, metrics

def main():
    # Load data
    processed_path = "data/processed/credit_features_engineered.csv"
    X, y = load_data(processed_path)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    best_metrics = {}
    best_model_name = None
    
    # Train models
    for name, config in MODELS.items():
        model, metrics = train_and_evaluate_model(name, config, X_train, X_test, y_train, y_test)
        if metrics['roc_auc'] > best_metrics.get('roc_auc', 0):
            best_metrics = metrics
            best_model_name = name
    
    # Register best model
    if best_model_name:
        with mlflow.start_run(run_name=f"best_{best_model_name}"):
            mlflow.sklearn.log_model(joblib.load(f"models/{best_model_name}_best.pkl"), "best_model")
            mlflow.log_param("registered_model", best_model_name)
            mlflow.log_metrics(best_metrics)
            # Register in registry
            mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/best_model", "CreditRiskModel")
        logger.info(f"Best model: {best_model_name} with ROC-AUC: {best_metrics['roc_auc']:.4f}")
    
    logger.info("Training complete. View in MLflow UI: mlflow ui")

if __name__ == "__main__":
    main()