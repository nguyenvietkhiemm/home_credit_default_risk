import numpy as np
import pandas as pd
import modules.cpp as cpp

def interest_rate(df: pd.DataFrame, n_iter: int = 500, init_rate: float = 0.9, tolerance: float = 1e-6) -> pd.DataFrame:
    _df = df.copy()
    
    N = len(_df)
    amt_annuity = _df["AMT_ANNUITY"].values.astype(np.float64)
    amt_credit = _df["AMT_CREDIT"].values.astype(np.float64)
    cnts = _df["CNT_PAYMENT"].values.astype(np.int32)
    rates = np.zeros(N, dtype=np.float64)

    cpp.interest_rate(amt_annuity, amt_credit, cnts, rates, N)
    
    _df = pd.concat([_df, pd.DataFrame({'interest_rate': rates}, index=_df.index)], axis=1)

    return _df