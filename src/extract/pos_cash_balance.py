import pandas as pd
import numpy as np
from config import ROOT, use_cols, prev_use_cols  # lib này được khởi tạo ban đầu dự án
import modules.utils as utils
import modules.multi as multi
from helpers.cache_clear import cache_clear
get_pickle = utils.get_pickle
_keep_vars = set(globals().keys())  # lưu biến gốc

def pos_cash_extract(test_run = False):
    if test_run:
        print("extract pos cash")
        for path in utils.get_pickle_paths(name="pos_cash"):
            print(path)
        return
    
    pos_balance = get_pickle("pos_cash")
    
    pos_balance.loc[((pos_balance["NAME_CONTRACT_STATUS"] == 'Active') & (pos_balance["CNT_INSTALMENT_FUTURE"] == 0)), 'NAME_CONTRACT_STATUS'] = 'Completed'  ####### sửa
    pos_balance.loc[(["NAME_CONTRACT_STATUS"] == 'Completed') & (pos_balance["CNT_INSTALMENT_FUTURE"] != 0), "CNT_INSTALMENT_FUTURE"] = "Active"  ####### sửa
    pos_balance_0 = pos_balance[pos_balance['CNT_INSTALMENT_FUTURE'] == 0].copy()
    pos_balance_1 = pos_balance[pos_balance['CNT_INSTALMENT_FUTURE'] > 0].copy()
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
    cache_clear(globals(), _keep_vars)
    utils.to_pickles(pos_balance, "pos_cash")
    
    print("extract pos cash")
    cache_clear(globals())
