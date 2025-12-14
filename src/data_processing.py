import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw transaction data."""
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df)} rows from {file_path}")
    return df

def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse TransactionStartTime to datetime."""
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], utc=True)
    return df

def calculate_rfm(df: pd.DataFrame, customer_id_col: str = 'CustomerId') -> pd.DataFrame:
    """Calculate RFM for customers (purchases only: Amount > 0)."""
    purchases = df[df['Amount'] > 0].copy()
    current_time = purchases['TransactionStartTime'].max()
    
    rfm = purchases.groupby(customer_id_col).agg({
        'TransactionStartTime': [
            ('Frequency', 'count'),
            ('Recency', lambda x: (current_time - x.max()).days)
        ],
        'Amount': [('Monetary', 'sum')]
    }).round(2)
    
    # Flatten multi-index columns
    rfm.columns = rfm.columns.droplevel(0)
    
    rfm = rfm.reset_index()
    logger.info(f"Computed RFM for {len(rfm)} customers")
    return rfm

def rfm_scoring(rfm: pd.DataFrame) -> pd.DataFrame:
    """Score RFM 1-5 (higher better; reverse for Recency)."""
    def score_col(series, reverse=False):
        if reverse:
            # Lower recency is better -> rank descending
            return pd.qcut(series.rank(method='dense', ascending=False), 5, labels=False, duplicates='drop') + 1
        else:
            # Higher frequency/monetary is better -> rank ascending
            return pd.qcut(series.rank(method='dense', ascending=True), 5, labels=False, duplicates='drop') + 1
    
    rfm['R_score'] = score_col(rfm['Recency'], reverse=True).astype(int)
    rfm['F_score'] = score_col(rfm['Frequency']).astype(int)
    rfm['M_score'] = score_col(rfm['Monetary']).astype(int)
    rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
    
    # Proxy: bad (1) if score < 9
    rfm['default_proxy'] = (rfm['RFM_score'] < 9).astype(int)
    return rfm

def engineer_features(df: pd.DataFrame, rfm: pd.DataFrame, customer_id_col: str = 'CustomerId') -> pd.DataFrame:
    """Engineer additional features."""
    # Fraud rate
    fraud_rate = df.groupby(customer_id_col)['FraudResult'].agg(['mean', 'count']).reset_index()
    fraud_rate.columns = [customer_id_col, 'fraud_rate', 'total_trans']
    
    # Volatility
    purchases = df[df['Amount'] > 0]
    volatility = purchases.groupby(customer_id_col)['Amount'].agg(['std', 'mean']).reset_index()
    volatility.columns = [customer_id_col, 'amount_std', 'avg_amount']
    
    # Unique products/channels
    unique = purchases.groupby(customer_id_col).agg({
        'ProductId': 'nunique',
        'ChannelId': 'nunique'
    }).reset_index()
    unique.columns = [customer_id_col, 'unique_products', 'unique_channels']
    
    # Merge all
    features = rfm.merge(fraud_rate, on=customer_id_col, how='left') \
                  .merge(volatility, on=customer_id_col, how='left') \
                  .merge(unique, on=customer_id_col, how='left')
    
    # Fill NaNs (e.g., no fraud)
    features['fraud_rate'] = features['fraud_rate'].fillna(0)
    features['amount_std'] = features['amount_std'].fillna(0)
    features['total_trans'] = features['total_trans'].fillna(1)
    features['unique_products'] = features['unique_products'].fillna(1)
    features['unique_channels'] = features['unique_channels'].fillna(1)
    
    # Correlation check (for selection later) - numeric only
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    if 'default_proxy' in numeric_cols:
        corr_with_target = features[numeric_cols].corr()['default_proxy'].abs().sort_values(ascending=False)
        logger.info(f"Top correlated features:\n{corr_with_target.head()}")
    
    return features

def save_processed(features: pd.DataFrame, output_path: str):
    """Save processed features."""
    features.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    # Adjust path if your file is named 'data.csv'
    raw_path = Path("data/raw/data.csv")  # Changed to match your file name
    processed_path = Path("data/processed/credit_features.csv")
    
    df = load_raw_data(raw_path)
    df = parse_timestamps(df)
    rfm = calculate_rfm(df)
    rfm = rfm_scoring(rfm)
    features = engineer_features(df, rfm)
    save_processed(features, processed_path)