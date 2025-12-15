# Updated src/data_processing.py for Task 3: Feature Engineering
# Integrates sklearn Pipeline for reproducible transformations
# Includes aggregation, time extraction, encoding, imputation, scaling, and custom WoE/IV

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import joblib

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

# Custom WoE Transformer (Manual Implementation)
class WOETransformer(BaseEstimator, TransformerMixin):
    """Apply Weight of Evidence transformation to binned features."""
    def __init__(self, target_col='default_proxy', bins=5):
        self.target_col = target_col
        self.bins = bins
        self.woe_dict = {}
    
    def fit(self, X, y=None):
        df = X.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != self.target_col:
                # Bin the feature
                try:
                    df[f'{col}_bin'] = pd.qcut(df[col], q=self.bins, duplicates='drop', labels=False)
                except:
                    df[f'{col}_bin'] = pd.cut(df[col], bins=self.bins, labels=False)
                # Calculate WoE: ln(%non-events / %events) per bin
                woe_vals = []
                total_events = df[self.target_col].sum()
                total_non_events = len(df) - total_events
                if total_events == 0 or total_non_events == 0:
                    woe_vals = [0] * self.bins
                else:
                    unique_bins = sorted(df[f'{col}_bin'].dropna().unique())
                    for bin_val in unique_bins:
                        bin_mask = df[f'{col}_bin'] == bin_val
                        bin_events = df[bin_mask][self.target_col].sum()
                        bin_total = bin_mask.sum()
                        bin_non_events = bin_total - bin_events
                        if bin_events == 0 or bin_non_events == 0 or total_events == 0 or total_non_events == 0:
                            woe = 0
                        else:
                            pct_events_bin = bin_events / total_events
                            pct_non_events_bin = bin_non_events / total_non_events
                            woe = np.log(pct_non_events_bin / pct_events_bin) if pct_events_bin > 0 and pct_non_events_bin > 0 else 0
                        woe_vals.append(woe)
                self.woe_dict[col] = dict(zip(unique_bins, woe_vals))
                df.drop(f'{col}_bin', axis=1, inplace=True)
        return self
    
    def transform(self, X):
        df = X.copy()
        for col, woe_map in self.woe_dict.items():
            # Bin consistently
            try:
                df[f'{col}_bin'] = pd.qcut(df[col], q=len(woe_map), duplicates='drop', labels=False)
            except:
                df[f'{col}_bin'] = pd.cut(df[col], bins=len(woe_map), labels=False)
            df[f'{col}_woe'] = df[f'{col}_bin'].map(woe_map).fillna(0)
            df.drop(f'{col}_bin', axis=1, inplace=True)
        # Keep WoE columns, drop originals
        woe_cols = [f'{col}_woe' for col in self.woe_dict.keys()]
        df = df[woe_cols]
        return df

# IV Calculator (for feature selection)
def calculate_iv(df, feature, target):
    """Calculate Information Value for a feature."""
    try:
        df_copy = df.copy()
        df_copy['bin'] = pd.qcut(df_copy[feature], q=10, duplicates='drop')
        iv = 0
        total_events = df_copy[target].sum()
        total_non = len(df_copy) - total_events
        if total_events == 0 or total_non == 0:
            return 0
        for bin_val in df_copy['bin'].unique():
            bin_df = df_copy[df_copy['bin'] == bin_val]
            bin_events = bin_df[target].sum()
            bin_total = len(bin_df)
            if bin_total == 0 or bin_events == 0 or bin_total == len(df_copy):
                continue
            pct_events_bin = bin_events / total_events
            pct_non_events_bin = (bin_total - bin_events) / total_non
            if pct_non_events_bin == 0 or pct_events_bin == 0:
                continue
            woe = np.log(pct_non_events_bin / pct_events_bin)
            iv += abs(pct_events_bin - pct_non_events_bin) * woe  # Use abs for selection
        df_copy.drop('bin', axis=1, inplace=True)
        return abs(iv)  # Return absolute for usefulness
    except:
        return 0

def engineer_features(df: pd.DataFrame) -> tuple:
    """Engineer base features (RFM + aggregates)."""
    # Parse timestamps first
    df = parse_timestamps(df)
    
    # RFM and proxy
    rfm = calculate_rfm(df)
    rfm = rfm_scoring(rfm)
    
    # Additional aggregates
    fraud_rate = df.groupby('CustomerId')['FraudResult'].agg(['mean', 'count']).reset_index()
    fraud_rate.columns = ['CustomerId', 'fraud_rate', 'total_trans']
    
    # Volatility and uniques (purchases)
    purchases = df[df['Amount'] > 0]
    volatility = purchases.groupby('CustomerId')['Amount'].agg(['std', 'mean']).reset_index()
    volatility.columns = ['CustomerId', 'amount_std', 'avg_amount']
    
    unique = purchases.groupby('CustomerId').agg({
        'ProductId': 'nunique',
        'ChannelId': 'nunique',
        'ProductCategory': 'nunique'
    }).reset_index()
    unique.columns = ['CustomerId', 'unique_products', 'unique_channels', 'unique_categories']
    
    # Merge all
    features = rfm.merge(fraud_rate, on='CustomerId', how='left') \
                  .merge(volatility, on='CustomerId', how='left') \
                  .merge(unique, on='CustomerId', how='left')
    
    # Fill NaNs
    features['fraud_rate'] = features['fraud_rate'].fillna(0)
    features['amount_std'] = features['amount_std'].fillna(0)
    features['total_trans'] = features['total_trans'].fillna(1)
    features['unique_products'] = features['unique_products'].fillna(1)
    features['unique_channels'] = features['unique_channels'].fillna(1)
    features['unique_categories'] = features['unique_categories'].fillna(1)
    
    # Time features
    time_agg = df.groupby('CustomerId')['TransactionStartTime'].agg([
        ('avg_hour', lambda x: x.dt.hour.mean()),
        ('avg_day', lambda x: x.dt.day.mean()),
        ('unique_months', lambda x: x.dt.month.nunique()),
        ('unique_years', lambda x: x.dt.year.nunique())
    ]).round(2).reset_index()
    
    features = features.merge(time_agg, on='CustomerId', how='left')
    
    # Fill any remaining NaNs from time
    features['avg_hour'] = features['avg_hour'].fillna(12)  # Noon default
    features['avg_day'] = features['avg_day'].fillna(15)  # Mid-month
    features['unique_months'] = features['unique_months'].fillna(1)
    features['unique_years'] = features['unique_years'].fillna(1)
    
    # Correlation/IV check
    numeric_cols = features.select_dtypes(include=[np.number]).columns.drop('default_proxy')
    corr_with_target = features[numeric_cols].corrwith(features['default_proxy']).abs().sort_values(ascending=False)
    logger.info(f"Top correlated features:\n{corr_with_target.head()}")
    
    # IV for selection
    selected_features = []
    for feat in numeric_cols:
        iv = calculate_iv(features, feat, 'default_proxy')
        logger.info(f"IV for {feat}: {iv}")
        if iv > 0.01:  # Lowered threshold for selection
            selected_features.append(feat)
    
    # Fallback if no features selected
    if not selected_features:
        logger.warning("No features with IV > 0.01; using all numeric features.")
        selected_features = list(numeric_cols)
    
    return features, selected_features

def apply_transformations(features: pd.DataFrame, selected_features: list, target_col: str = 'default_proxy'):
    """Apply imputation, scaling, and WoE via pipeline."""
    # Prepare columns
    X = features[selected_features]
    y = features[target_col]
    
    # Ensure X is numeric
    X = X.select_dtypes(include=[np.number])
    
    if len(X.columns) == 0:
        raise ValueError("No numeric features available after selection.")
    
    # Column Transformer for num (all selected are num)
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    X_transformed = preprocessor.fit_transform(X)
    X_transformed_df = pd.DataFrame(X_transformed, columns=X.columns, index=features.index)
    
    # WoE on original features for binning
    woe_transformer = WOETransformer(target_col=target_col, bins=5)
    woe_df = woe_transformer.fit_transform(features[X.columns.tolist() + [target_col]])
    
    # Combine: scaled + WoE
    final_X = pd.concat([X_transformed_df, woe_df], axis=1)
    
    final_df = final_X.copy()
    final_df['CustomerId'] = features['CustomerId']
    final_df[target_col] = y
    
    return final_df, preprocessor

def save_processed(final_df: pd.DataFrame, preprocessor, output_path: str):
    """Save processed features and artifacts."""
    Path("models").mkdir(exist_ok=True)
    final_df.to_csv(output_path, index=False)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    logger.info(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    raw_path = Path("data/raw/data.csv")
    processed_path = Path("data/processed/credit_features_engineered.csv")
    
    df = load_raw_data(raw_path)
    features, selected = engineer_features(df)
    final_df, preprocessor = apply_transformations(features, selected)
    save_processed(final_df, preprocessor, processed_path)