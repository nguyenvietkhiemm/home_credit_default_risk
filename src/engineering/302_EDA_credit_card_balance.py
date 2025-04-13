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
import helpers.config as config
import modules.utils as utils
import modules.multi as multi
importlib.reload(view)
importlib.reload(EDA)
importlib.reload(utils)
importlib.reload(config)
importlib.reload(multi)
use_cols = config.use_cols
prev_use_cols = config.prev_use_cols
app_day_cols = config.app_day_cols
def cache_clear():
    for var in list(globals()):
        if var not in _keep_vars and not var.startswith("_"):
            del globals()[var]
    gc.collect()
_keep_vars = set(globals().keys())  # lưu biến gốc
credit_balance = pd.read_pickle(ROOT + "/data/pkl/credit_card_balance.p")
credit_balance["AMT_BALANCE-d-AMT_CREDIT_LIMIT_ACTUAL"] = credit_balance["AMT_BALANCE"] / credit_balance["AMT_CREDIT_LIMIT_ACTUAL"]
credit_balance['AMT_DRAWINGS_CURRENT-d-AMT_CREDIT_LIMIT_ACTUAL'] = credit_balance['AMT_DRAWINGS_CURRENT'] / credit_balance['AMT_CREDIT_LIMIT_ACTUAL']
credit_balance['AMT_TOTAL_RECEIVABLE-d-AMT_BALANCE'] = credit_balance['AMT_TOTAL_RECEIVABLE'] / credit_balance['AMT_BALANCE']
credit_balance['AMT_RECIVABLE-d-AMT_BALANCE'] = credit_balance['AMT_RECIVABLE'] / credit_balance['AMT_BALANCE']
credit_balance['AMT_RECEIVABLE_PRINCIPAL-d-AMT_BALANCE'] = credit_balance['AMT_RECEIVABLE_PRINCIPAL'] / credit_balance['AMT_BALANCE']
credit_balance['AMT_INST_MIN_REGULARITY-d-AMT_BALANCE'] = credit_balance['AMT_INST_MIN_REGULARITY'] / credit_balance['AMT_BALANCE']
credit_balance['AMT_BALANCE-d-AMT_DRAWINGS_CURRENT'] = credit_balance['AMT_BALANCE'] / credit_balance['AMT_DRAWINGS_CURRENT']
credit_balance['AMT_DRAWINGS_ATM_CURRENT-d-AMT_DRAWINGS_CURRENT'] = credit_balance['AMT_DRAWINGS_ATM_CURRENT'] / credit_balance['AMT_DRAWINGS_CURRENT']
credit_balance['AMT_DRAWINGS_OTHER_CURRENT-d-AMT_DRAWINGS_CURRENT'] = credit_balance['AMT_DRAWINGS_OTHER_CURRENT'] / credit_balance['AMT_DRAWINGS_CURRENT']
credit_balance['AMT_DRAWINGS_POS_CURRENT-d-AMT_DRAWINGS_CURRENT'] = credit_balance['AMT_DRAWINGS_POS_CURRENT'] / credit_balance['AMT_DRAWINGS_CURRENT']
credit_balance['AMT_RECEIVABLE_PRINCIPAL-d-AMT_TOTAL_RECEIVABLE'] = credit_balance['AMT_RECEIVABLE_PRINCIPAL'] / credit_balance['AMT_TOTAL_RECEIVABLE']
credit_balance['AMT_RECIVABLE-d-AMT_TOTAL_RECEIVABLE'] = credit_balance['AMT_RECIVABLE'] / credit_balance['AMT_TOTAL_RECEIVABLE']
credit_balance['AMT_TOTAL_RECEIVABLE-s-AMT_RECIVABLE'] = credit_balance['AMT_TOTAL_RECEIVABLE'] - credit_balance['AMT_RECIVABLE']
credit_balance['AMT_RECIVABLE-s-AMT_RECEIVABLE_PRINCIPAL'] = credit_balance['AMT_TOTAL_RECEIVABLE'] - credit_balance['AMT_RECEIVABLE_PRINCIPAL']
credit_balance["AMT_PAYMENT_TOTAL_CURRENT-d-AMT_PAYMENT_CURRENT"] = credit_balance["AMT_PAYMENT_TOTAL_CURRENT"] / credit_balance["AMT_PAYMENT_CURRENT"]
credit_balance["AMT_PAYMENT_TOTAL_CURRENT-s-AMT_PAYMENT_CURRENT"] = credit_balance["AMT_PAYMENT_TOTAL_CURRENT"] - credit_balance["AMT_PAYMENT_CURRENT"]
credit_balance['SK_DPD-s-SK_DPD_DEF'] = credit_balance['SK_DPD'] - credit_balance['SK_DPD_DEF']
credit_balance['SK_DPD-s-SK_DPD_DEF_over0'] = (credit_balance['SK_DPD-s-SK_DPD_DEF'] > 0) * 1
credit_balance['SK_DPD-s-SK_DPD_DEF_over5'] = (credit_balance['SK_DPD-s-SK_DPD_DEF'] > 5) * 1
credit_balance['SK_DPD-s-SK_DPD_DEF_over10'] = (credit_balance['SK_DPD-s-SK_DPD_DEF'] > 10) * 1
credit_balance['SK_DPD-s-SK_DPD_DEF_over15'] = (credit_balance['SK_DPD-s-SK_DPD_DEF'] > 15) * 1
credit_balance['SK_DPD-s-SK_DPD_DEF_over20'] = (credit_balance['SK_DPD-s-SK_DPD_DEF'] > 20) * 1
credit_balance['SK_DPD-s-SK_DPD_DEF_over25'] = (credit_balance['SK_DPD-s-SK_DPD_DEF'] > 25) * 1
test = pd.read_pickle(ROOT + "/data/pkl/application_test.p")[use_cols]
train = pd.read_pickle(ROOT + "/data/pkl/application_train.p")[use_cols]
trte = utils.get_trte(train, test)
trte[app_day_cols] = trte[app_day_cols] / 30  # get month
del train, test
gc.collect()
credit_balance = pd.merge(credit_balance, trte, on="SK_ID_CURR", how="left")
del trte
gc.collect()
credit_balance['AMT_BALANCE-d-app_AMT_INCOME_TOTAL'] = credit_balance['AMT_BALANCE'] / credit_balance['app_AMT_INCOME_TOTAL']
credit_balance['AMT_BALANCE-d-app_AMT_CREDIT'] = credit_balance['AMT_BALANCE'] / credit_balance['app_AMT_CREDIT']
credit_balance['AMT_BALANCE-d-app_AMT_ANNUITY'] = credit_balance['AMT_BALANCE'] / credit_balance['app_AMT_ANNUITY']
credit_balance['AMT_BALANCE-d-app_AMT_GOODS_PRICE'] = credit_balance['AMT_BALANCE'] / credit_balance['app_AMT_GOODS_PRICE']
credit_balance['AMT_DRAWINGS_CURRENT-d-app_AMT_INCOME_TOTAL'] = credit_balance['AMT_DRAWINGS_CURRENT'] / credit_balance['app_AMT_INCOME_TOTAL']
credit_balance['AMT_DRAWINGS_CURRENT-d-app_AMT_CREDIT'] = credit_balance['AMT_DRAWINGS_CURRENT'] / credit_balance['app_AMT_CREDIT']
credit_balance['AMT_DRAWINGS_CURRENT-d-app_AMT_ANNUITY'] = credit_balance['AMT_DRAWINGS_CURRENT'] / credit_balance['app_AMT_ANNUITY']
credit_balance['AMT_DRAWINGS_CURRENT-d-app_AMT_GOODS_PRICE'] = credit_balance['AMT_DRAWINGS_CURRENT'] / credit_balance['app_AMT_GOODS_PRICE']
for c in app_day_cols:
    print(f'MONTHS_BALANCE-s-{c}')
    credit_balance[f'MONTHS_BALANCE-s-{c}'] = credit_balance['MONTHS_BALANCE'] - credit_balance[c]
drop_cols = [
    'app_AMT_INCOME_TOTAL', 'app_AMT_CREDIT', 'app_AMT_ANNUITY', 'app_AMT_GOODS_PRICE', 'app_DAYS_BIRTH', 'app_DAYS_EMPLOYED', 'app_DAYS_REGISTRATION', 'app_DAYS_ID_PUBLISH',
    'app_DAYS_LAST_PHONE_CHANGE'
]
credit_balance.drop(drop_cols, inplace=True, axis=1)
cols = [
    'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT', 'AMT_INST_MIN_REGULARITY',
    'AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
    'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM', 'SK_DPD', 'SK_DPD_DEF', 'AMT_BALANCE-d-AMT_CREDIT_LIMIT_ACTUAL',
    'AMT_DRAWINGS_CURRENT-d-AMT_CREDIT_LIMIT_ACTUAL', 'AMT_TOTAL_RECEIVABLE-d-AMT_BALANCE', 'AMT_RECIVABLE-d-AMT_BALANCE', 'AMT_RECEIVABLE_PRINCIPAL-d-AMT_BALANCE',
    'AMT_INST_MIN_REGULARITY-d-AMT_BALANCE', 'AMT_BALANCE-d-AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_ATM_CURRENT-d-AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT-d-AMT_DRAWINGS_CURRENT',
    'AMT_DRAWINGS_POS_CURRENT-d-AMT_DRAWINGS_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL-d-AMT_TOTAL_RECEIVABLE', 'AMT_RECIVABLE-d-AMT_TOTAL_RECEIVABLE', 'AMT_TOTAL_RECEIVABLE-s-AMT_RECIVABLE',
    'AMT_RECIVABLE-s-AMT_RECEIVABLE_PRINCIPAL', 'AMT_PAYMENT_TOTAL_CURRENT-d-AMT_PAYMENT_CURRENT', 'AMT_PAYMENT_TOTAL_CURRENT-s-AMT_PAYMENT_CURRENT', 'AMT_BALANCE-d-app_AMT_INCOME_TOTAL',
    'AMT_BALANCE-d-app_AMT_CREDIT', 'AMT_BALANCE-d-app_AMT_ANNUITY', 'AMT_BALANCE-d-app_AMT_GOODS_PRICE', 'AMT_DRAWINGS_CURRENT-d-app_AMT_INCOME_TOTAL', 'AMT_DRAWINGS_CURRENT-d-app_AMT_CREDIT',
    'AMT_DRAWINGS_CURRENT-d-app_AMT_ANNUITY', 'AMT_DRAWINGS_CURRENT-d-app_AMT_GOODS_PRICE'
]
credit_balance.sort_values(['SK_ID_PREV', 'MONTHS_BALANCE'], inplace=True)
credit_balance.reset_index(drop=True, inplace=True)
df_list = []
for col in cols:
    df = multi.multi(col, credit_balance)  # diff pctchange
    df_list.append(df)
df = pd.concat(df_list, axis=1)
del df_list
gc.collect()
batch_size = 1000000
n_rows = credit_balance.shape[0]
for i in range(0, n_rows, batch_size):
    credit_batch = credit_balance.iloc[i:i + batch_size]
    df_batch = df.iloc[i:i + batch_size]
    combined = pd.concat([credit_batch, df_batch], axis=1)
    combined.replace(np.inf, np.nan, inplace=True)
    combined.replace(-np.inf, np.nan, inplace=True)
    combined.to_pickle(ROOT + f'/data/processed/f301_credit_balance_batch_{i//batch_size + 1}.p')
    del combined, credit_batch, df_batch
    gc.collect()
tmp = pd.read_pickle(ROOT + f'/data/processed/f301_credit_balance_batch_1.p')  # test
cache_clear()
