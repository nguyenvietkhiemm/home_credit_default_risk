import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import gc
import io
import os
from itertools import combinations
from IPython.display import display
pd.set_option('display.max_columns', 99)
pd.set_option('display.max_rows', 200)
pd.reset_option('display.float_format')
pd.set_option('display.max_colwidth', None)  
from sitecustomize import ROOT # lib này được khởi tạo ban đầu dự án
import helpers.view as view
import helpers.EDA as EDA
import modules.multi as multi
importlib.reload(view)
importlib.reload(EDA)
importlib.reload(multi)
def cache_clear():
    for var in list(globals()):  
        if var not in _keep_vars and not var.startswith("_"):  
            del globals()[var]  
    gc.collect()
_keep_vars = set(globals().keys())  # lưu biến gốc
pos_balance = pd.read_pickle(ROOT + "/data/pkl/pos_cash_balance.p")
pos_balance.loc[((pos_balance["NAME_CONTRACT_STATUS"]=='Active') & (pos_balance["CNT_INSTALMENT_FUTURE"]==0)), 'NAME_CONTRACT_STATUS'] = 'Completed' ####### sửa
pos_balance.loc[(["NAME_CONTRACT_STATUS"]=='Completed') & (pos_balance["CNT_INSTALMENT_FUTURE"]!=0), "CNT_INSTALMENT_FUTURE"] = "Active" ####### sửa
pos_balance_0 = pos_balance[pos_balance['CNT_INSTALMENT_FUTURE']==0].copy()
pos_balance_1 = pos_balance[pos_balance['CNT_INSTALMENT_FUTURE']>0].copy()
pos_balance_0.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'], ascending=[True, False], inplace=True)
pos_balance_0.drop_duplicates(['SK_ID_PREV', "NAME_CONTRACT_STATUS"], keep='last', inplace=True)
pos_balance = pd.concat([pos_balance_0, pos_balance_1], ignore_index=True)
pos_balance['CNT_INSTALMENT-s-CNT_INSTALMENT_FUTURE'] = pos_balance['CNT_INSTALMENT'] - pos_balance['CNT_INSTALMENT_FUTURE']
pos_balance['CNT_INSTALMENT_FUTURE-d-CNT_INSTALMENT'] = pos_balance['CNT_INSTALMENT_FUTURE'] / pos_balance['CNT_INSTALMENT']
pos_balance['SK_DPD-s-SK_DPD_DEF'] = pos_balance['SK_DPD'] - pos_balance['SK_DPD_DEF']
pos_balance.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'], inplace=True)
pos_balance.reset_index(drop=True, inplace=True)
cols = ['CNT_INSTALMENT_FUTURE', 'SK_DPD', 'SK_DPD_DEF']
df_list = []
for col in cols:
    df = multi.multi(col, pos_balance)
    df_list.append(df)
df = pd.concat(df_list, axis=1)
pos_balance = pd.concat([pos_balance, df], axis=1)
pos_balance.replace(np.inf, np.nan, inplace=True) 
pos_balance.replace(-np.inf, np.nan, inplace=True)
_keep_vars.update(["pos_balance"])
cache_clear()
pos_balance.to_pickle(ROOT + "/data/processed/f401_pos_balance.p")
pos_balance
cache_clear()
