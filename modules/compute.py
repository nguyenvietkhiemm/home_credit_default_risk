import numpy as np
import pandas as pd

def interest_rate(df: pd.DataFrame, n_iter: int = 500, init_rate: float = 0.9, tolerance: float = 1e-6) -> pd.DataFrame:
    _df = df.copy()
    
    mask = (_df['AMT_CREDIT'] > 0) & (_df['CNT_PAYMENT'] > 0) & (_df['AMT_ANNUITY'] > 0)
    df_valid = _df[mask].copy()

    annuity_credit_ratio = df_valid['AMT_ANNUITY'] / df_valid['AMT_CREDIT']
    cnt = df_valid['CNT_PAYMENT'].values
    rate = np.full_like(cnt, init_rate, dtype=np.float64)

    for _ in range(n_iter):
        rate_new = annuity_credit_ratio.values * ((1 + rate)**cnt - 1) / ((1 + rate)**cnt)
        if np.all(np.abs(rate_new - rate) < tolerance):
            break
        rate = rate_new

    _df.loc[mask, 'interest_rate'] = rate

    return _df