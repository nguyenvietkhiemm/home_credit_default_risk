import pandas as pd
import numpy as np
import gc

from config import ROOT, use_cols, prev_use_cols, bureau_money_cols, bureau_day_cols  # lib này được khởi tạo ban đầu dự án
import modules.utils as utils
import modules.multi as multi
from helpers.cache_clear import cache_clear

get_pickle = utils.get_pickle
_keep_vars = set(globals().keys())  # lưu biến gốc

def bureau_extract(test_run=False):
    if test_run:
        print("extract bureau")
        for path in utils.get_pickle_paths(name="bureau"):
            print(path)
        return
    
    bureau = get_pickle("bureau")
    
    bureau["DAYS_CREDIT_ENDDATE-s-DAYS_CREDIT"] = bureau["DAYS_CREDIT_ENDDATE"] - bureau["DAYS_CREDIT"]
    bureau["DAYS_ENDDATE_FACT-s-DAYS_CREDIT"] = bureau["DAYS_ENDDATE_FACT"] - bureau["DAYS_CREDIT"]
    bureau["DAYS_CREDIT_UPDATE-s-DAYS_CREDIT"] = bureau["DAYS_CREDIT_UPDATE"] - bureau["DAYS_CREDIT"]
    bureau["DAYS_ENDDATE_FACT-s-DAYS_CREDIT_ENDDATE"] = bureau["DAYS_ENDDATE_FACT"] - bureau["DAYS_CREDIT_ENDDATE"]
    bureau["DAYS_CREDIT_UPDATE-s-DAYS_CREDIT_ENDDATE"] = bureau["DAYS_CREDIT_UPDATE"] - bureau["DAYS_CREDIT_ENDDATE"]
    bureau["DAYS_CREDIT_UPDATE-s-DAYS_ENDDATE_FACT"] = bureau["DAYS_CREDIT_UPDATE"] - bureau["DAYS_ENDDATE_FACT"]
    bureau['AMT_CREDIT_SUM-s-AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    bureau['AMT_CREDIT_SUM_DEBT-d-AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM']
    bureau['AMT_CREDIT_MAX_OVERDUE-d-AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_MAX_OVERDUE'] / bureau['AMT_CREDIT_SUM']
    bureau['AMT_CREDIT_SUM_OVERDUE-d-AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM_OVERDUE'] / bureau['AMT_CREDIT_SUM']
    bureau['AMT_CREDIT_SUM_DEBT-d-AMT_CREDIT_SUM_LIMIT'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM_LIMIT']
    bureau['AMT_CREDIT_SUM-s-AMT_CREDIT_SUM_DEBT-d-AMT_CREDIT_SUM_LIMIT'] = bureau['AMT_CREDIT_SUM-s-AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM_LIMIT']
    bureau['AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT'] = bureau['AMT_CREDIT_SUM_DEBT'] + bureau['AMT_CREDIT_SUM_LIMIT']  # số tiền còn phải trả + hạn mức
    bureau["AMT_CREDIT_SUM-d-AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT"] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT']
    bureau["AMT_ANNUITY-d-AMT_CREDIT_SUM"] = bureau["AMT_ANNUITY"] / bureau["AMT_CREDIT_SUM"]
    bureau["AMT_ANNUITY-d-AMT_CREDIT_MAX_OVERDUE"] = bureau["AMT_ANNUITY"] / bureau["AMT_CREDIT_MAX_OVERDUE"]
    bureau["AMT_ANNUITY-d-AMT_CREDIT_SUM_OVERDUE"] = bureau["AMT_ANNUITY"] / bureau["AMT_CREDIT_SUM_OVERDUE"]

    trte = utils.get_trte()
    bureau = pd.merge(bureau, trte, on='SK_ID_CURR', how='left')
    from config import app_money_cols, app_day_cols
    for c1 in bureau_money_cols:
        for c2 in app_money_cols:
            bureau[f'{c1}-d-{c2}'] = bureau[c1] / bureau[c2]
    for c1 in bureau_day_cols:
        for c2 in app_day_cols:
            bureau[f'{c1}-s-{c2}'] = bureau[c1] - bureau[c2]
            bureau[f'{c1}-d-{c2}'] = bureau[c1] / bureau[c2]
    del trte
    gc.collect()
    bureau.drop(app_money_cols + app_day_cols, axis=1, inplace=True)
    cols = [
        'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'DAYS_CREDIT_UPDATE', 'AMT_ANNUITY',
        'AMT_CREDIT_SUM-s-AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_DEBT-d-AMT_CREDIT_SUM', 'AMT_CREDIT_MAX_OVERDUE-d-AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_OVERDUE-d-AMT_CREDIT_SUM',
        'AMT_CREDIT_SUM_DEBT-d-AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM-s-AMT_CREDIT_SUM_DEBT-d-AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT',
        'AMT_CREDIT_SUM-d-AMT_CREDIT_SUM_DEBT-p-AMT_CREDIT_SUM_LIMIT', 'AMT_ANNUITY-d-AMT_CREDIT_SUM', 'AMT_ANNUITY-d-AMT_CREDIT_MAX_OVERDUE', 'AMT_ANNUITY-d-AMT_CREDIT_SUM_OVERDUE',
        'AMT_CREDIT_SUM-d-app_AMT_INCOME_TOTAL', 'AMT_CREDIT_SUM-d-app_AMT_CREDIT', 'AMT_CREDIT_SUM-d-app_AMT_ANNUITY', 'AMT_CREDIT_SUM-d-app_AMT_GOODS_PRICE',
        'AMT_CREDIT_SUM_DEBT-d-app_AMT_INCOME_TOTAL', 'AMT_CREDIT_SUM_DEBT-d-app_AMT_CREDIT', 'AMT_CREDIT_SUM_DEBT-d-app_AMT_ANNUITY', 'AMT_CREDIT_SUM_DEBT-d-app_AMT_GOODS_PRICE',
        'AMT_CREDIT_SUM_LIMIT-d-app_AMT_INCOME_TOTAL', 'AMT_CREDIT_SUM_LIMIT-d-app_AMT_CREDIT', 'AMT_CREDIT_SUM_LIMIT-d-app_AMT_ANNUITY', 'AMT_CREDIT_SUM_LIMIT-d-app_AMT_GOODS_PRICE',
        'AMT_CREDIT_SUM_OVERDUE-d-app_AMT_INCOME_TOTAL', 'AMT_CREDIT_SUM_OVERDUE-d-app_AMT_CREDIT', 'AMT_CREDIT_SUM_OVERDUE-d-app_AMT_ANNUITY', 'AMT_CREDIT_SUM_OVERDUE-d-app_AMT_GOODS_PRICE',
        'AMT_ANNUITY-d-app_AMT_INCOME_TOTAL', 'AMT_ANNUITY-d-app_AMT_CREDIT', 'AMT_ANNUITY-d-app_AMT_ANNUITY', 'AMT_ANNUITY-d-app_AMT_GOODS_PRICE'
    ]
    bureau.sort_values(['SK_ID_CURR', 'DAYS_CREDIT'], inplace=True)
    bureau.reset_index(drop=True, inplace=True)
    bureau.replace(np.inf, np.nan, inplace=True)
    bureau.replace(-np.inf, np.nan, inplace=True)
    df_list = []
    for col in cols:
        df = multi.multi(col, bureau)  # diff pctchange
        df_list.append(df)
    df = pd.concat(df_list, axis=1)
    del df_list
    gc.collect()
    bureau = pd.concat([bureau, df], axis=1)
    utils.to_pickles(bureau, "bureau")
    
    print("extract bureau")
    cache_clear(globals())
