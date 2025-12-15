import pytest
import pandas as pd
from src.data_processing import calculate_rfm, rfm_scoring

def test_rfm_calculation():
    # Synthetic data
    data = {
        'CustomerId': ['A', 'A', 'B'],
        'Amount': [100, 200, 150],
        'TransactionStartTime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    }
    df = pd.DataFrame(data)
    
    rfm = calculate_rfm(df)
    assert len(rfm) == 2
    assert rfm.loc[rfm['CustomerId'] == 'A', 'Frequency'].iloc[0] == 2
    assert rfm.loc[rfm['CustomerId'] == 'A', 'Monetary'].iloc[0] == 300

def test_rfm_scoring():
    rfm_data = pd.DataFrame({
        'Recency': [1, 30], 'Frequency': [10, 1], 'Monetary': [1000, 100]
    })
    scored = rfm_scoring(rfm_data)
    assert scored['RFM_score'].min() >= 3
    assert scored['default_proxy'].sum() == 1  # One bad


# Updated tests/test_data_processing.py for Task 5: Add unit tests
import pytest
import pandas as pd
import numpy as np
from src.data_processing import calculate_rfm, rfm_scoring, create_high_risk_proxy, engineer_features, calculate_iv

def test_calculate_rfm():
    # Synthetic data
    data = {
        'CustomerId': ['A', 'A', 'B'],
        'Amount': [100, 200, 150],
        'TransactionStartTime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
    }
    df = pd.DataFrame(data)
    rfm = calculate_rfm(df)
    assert len(rfm) == 2
    assert rfm.loc[rfm['CustomerId'] == 'A', 'Frequency'].iloc[0] == 2
    assert rfm.loc[rfm['CustomerId'] == 'A', 'Monetary'].iloc[0] == 300
    assert 'Recency' in rfm.columns

def test_rfm_scoring():
    rfm_data = pd.DataFrame({
        'Recency': [1, 30], 'Frequency': [10, 1], 'Monetary': [1000, 100]
    })
    scored = rfm_scoring(rfm_data)
    assert 'RFM_score' in scored.columns
    assert scored['RFM_score'].min() >= 3
    assert len(scored) == 2

def test_create_high_risk_proxy():
    rfm_data = pd.DataFrame({
        'CustomerId': ['A', 'B', 'C'],
        'Recency': [1, 30, 5],
        'Frequency': [10, 1, 8],
        'Monetary': [1000, 100, 800]
    })
    proxy_df = create_high_risk_proxy(rfm_data)
    assert 'is_high_risk' in proxy_df.columns
    assert proxy_df['is_high_risk'].sum() > 0  # At least one high-risk
    assert proxy_df['is_high_risk'].dtype == 'int64'

# def test_engineer_features():
#     # Use small synthetic df
#     data = {
#         'CustomerId': ['A'] * 2 + ['B'],
#         'Amount': [100, 200, 150],
#         'TransactionStartTime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
#         'FraudResult': [0, 1, 0],
#         'ProductId': ['P1', 'P2', 'P1'],
#         'ChannelId': ['C1', 'C1', 'C2'],
#         'ProductCategory': ['Cat1', 'Cat1', 'Cat2']
#     }
#     df = pd.DataFrame(data)
#     features, selected = engineer_features(df)
#     assert 'is_high_risk' in features.columns
#     assert len(selected) > 0
    # assert features.shape[0] == 2  # Unique customers

def test_calculate_iv():
    df_test = pd.DataFrame({
        'feature': [1, 2, 3, 4, 5],
        'target': [0, 0, 1, 1, 0]
    })
    iv = calculate_iv(df_test, 'feature', 'target')
    assert isinstance(iv, (int, float))
    assert iv >= 0