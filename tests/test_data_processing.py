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