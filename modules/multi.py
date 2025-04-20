# tính diff và pctchange

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def multi(c, df):
    df_grouped = df.groupby('SK_ID_CURR')[c]

    ret_diff = df_grouped.diff()

    prev = df_grouped.shift(1)
    ret_pctchng = (df[c] - prev) / prev.replace(0, np.nan)

    ret_diff = ret_diff.rename(f'{c}_diff')
    ret_pctchng = ret_pctchng.rename(f'{c}_pctchange')

    return pd.concat([ret_diff, ret_pctchng], axis=1)

def diff_pctchange(cols, _df):
    df_list = []
    for col in cols:
        df = multi(col, _df)  # diff pctchange
        df_list.append(df)
    df = pd.concat(df_list, axis=1)
    
    return df