import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import gc
import io
import os
from IPython.display import display
pd.set_option('display.max_columns', 99)
pd.set_option('display.max_rows', 200)
pd.reset_option('display.float_format')
pd.set_option('display.max_colwidth', None)
from sitecustomize import ROOT  # lib này được khởi tạo ban đầu dự án
import helpers.view as view
import helpers.EDA as EDA
import config.config as config
import modules.utils as utils
import modules.multi as multi
from helpers.cache_clear import cache_clear
importlib.reload(view)
importlib.reload(EDA)
importlib.reload(utils)
importlib.reload(config)
importlib.reload(multi)
use_cols = config.use_cols
prev_use_cols = config.prev_use_cols
app_day_cols = config.app_day_cols
_keep_vars = set(globals().keys())  # lưu biến gốc
bureau = pd.read_pickle(ROOT + "/data/pkl/bureau.p")
bureau_money_cols = ["AMT_CREDIT_MAX_OVERDUE", 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', "AMT_ANNUITY"]
bureau_day_cols = ['DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', "DAYS_CREDIT_UPDATE"]
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
train = pd.read_pickle(ROOT + "/data/pkl/application_train.p")[use_cols]
test = pd.read_pickle(ROOT + "/data/pkl/application_test.p")[use_cols]
trte = utils.get_trte(train, test)
del train, test
gc.collect()
bureau = pd.merge(bureau, trte, on='SK_ID_CURR', how='left')
from helpers.config import app_money_cols, app_day_cols
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
bureau.to_pickle(ROOT + "/data/processed/f501_bureau.p")
cache_clear(globals())
