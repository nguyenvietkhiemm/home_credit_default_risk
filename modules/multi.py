import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def multi(c): # diff và pctchange
    df_grouped = prev.groupby('SK_ID_CURR')[c]

    ret_diff = df_grouped.diff() 
    ret_pctchng = df_grouped.pct_change()  
    
    ret_diff = ret_diff.rename(f'{c}_diff')
    ret_pctchng = ret_pctchng.rename(f'{c}_pctchange')

    ret = pd.concat([ret_diff, ret_pctchng], axis=1)
    
    return ret
