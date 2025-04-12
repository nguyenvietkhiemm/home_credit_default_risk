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
def cache_clear():
    for var in list(globals()):  
        if var not in _keep_vars and not var.startswith("_"):  
            del globals()[var]  
    gc.collect()
_keep_vars = set(globals().keys())  # lưu biến gốc
prev = pd.read_csv(ROOT + "/data/csv/previous_application.csv")
train = pd.read_pickle(ROOT + "/data/pkl/application_train.p")
test = pd.read_pickle(ROOT + "/data/pkl/application_test.p")
obj_features = [c for c in prev.columns if (prev[c].dtype=="O") | (prev[c].nunique() <= 7)]
con_features = [c for c in prev.columns if c not in obj_features]
prev_money_cols = [ 'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE']
prev_rate_cols = ['RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED']
prev_other_cols = ['HOUR_APPR_PROCESS_START', 'SELLERPLACE_AREA']
prev_cnt_cols = ['CNT_PAYMENT']
prev_day_cols = ['DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION'] 
train_id = train["SK_ID_CURR"].unique()
prev.loc[prev['SK_ID_CURR'].isin(train_id), 'data'] = 1
prev.loc[~prev['SK_ID_CURR'].isin(train_id), 'data'] = 0
prev_train = prev[prev["data"] == 1]
prev_test = prev[prev["data"] == 0]
_keep_vars.update(["prev", "train", "test", "obj_features", "con_features"])
cache_clear()
prev['AMT_APPLICATION'] = prev['AMT_APPLICATION'].replace(0, np.nan)
prev['AMT_CREDIT'] = prev['AMT_CREDIT'].replace(0, np.nan)
prev['CNT_PAYMENT'] = prev['CNT_PAYMENT'].replace(0, np.nan)
prev['AMT_DOWN_PAYMENT'] = prev['AMT_DOWN_PAYMENT'].replace(np.nan, 0)
prev['RATE_DOWN_PAYMENT'] = prev['RATE_DOWN_PAYMENT'].replace(np.nan, 0)
prev['FLAG_LAST_APPL_PER_CONTRACT'] = (prev['FLAG_LAST_APPL_PER_CONTRACT']=='Y')*1 # biến nhị phân
for c in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 
            'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
    prev.loc[prev[c]==365243, c] = np.nan
prev["DAYS_FIRST_DRAWING-s-DAYS_DECISIONS"] = prev["DAYS_FIRST_DRAWING"] - prev["DAYS_DECISION"]
prev["DAYS_FIRST_DUE-s-DAYS_DECISIONS"] = prev["DAYS_FIRST_DUE"] - prev["DAYS_DECISION"]
prev["DAYS_LAST_DUE_1ST_VERSION-s-DAYS_DECISIONS"] = prev["DAYS_LAST_DUE_1ST_VERSION"] - prev["DAYS_DECISION"]
prev["DAYS_LAST_DUE-s-DAYS_DECISIONS"] = prev["DAYS_LAST_DUE"] - prev["DAYS_DECISION"]
prev["DAYS_TERMINATION-s-DAYS_DECISIONS"] = prev["DAYS_TERMINATION"] - prev["DAYS_DECISION"]
prev['DAYS_FIRST_DUE-s-DAYS_FIRST_DRAWING'] = prev['DAYS_FIRST_DUE'] - prev['DAYS_FIRST_DRAWING']
prev['DAYS_LAST_DUE_1ST_VERSION-s-DAYS_FIRST_DRAWING'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_FIRST_DRAWING']
prev['DAYS_LAST_DUE-s-DAYS_FIRST_DRAWING'] = prev['DAYS_LAST_DUE'] - prev['DAYS_FIRST_DRAWING']
prev['DAYS_TERMINATION-s-DAYS_FIRST_DRAWING'] = prev['DAYS_TERMINATION'] - prev['DAYS_FIRST_DRAWING']
prev['DAYS_LAST_DUE_1ST_VERSION-s-DAYS_FIRST_DUE'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_FIRST_DUE']
prev['DAYS_LAST_DUE-s-DAYS_FIRST_DUE'] = prev['DAYS_LAST_DUE'] - prev['DAYS_FIRST_DUE']
prev['DAYS_TERMINATION-s-DAYS_FIRST_DUE'] = prev['DAYS_TERMINATION'] - prev['DAYS_FIRST_DUE']
prev['DAYS_LAST_DUE-s-DAYS_LAST_DUE_1ST_VERSION'] = prev['DAYS_LAST_DUE'] - prev['DAYS_LAST_DUE_1ST_VERSION']
prev['DAYS_TERMINATION-s-DAYS_LAST_DUE_1ST_VERSION'] = prev['DAYS_TERMINATION'] - prev['DAYS_LAST_DUE_1ST_VERSION']
prev['DAYS_TERMINATION-s-DAYS_LAST_DUE'] = prev['DAYS_TERMINATION'] - prev['DAYS_LAST_DUE']
prev['total_debt'] = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
prev['AMT_GOODS_PRICE-d-total_debt'] = prev['AMT_GOODS_PRICE'] / prev['total_debt']
prev['AMT_CREDIT-d-total_debt'] = prev['AMT_CREDIT'] / prev['total_debt'] # so sánh số tiền phải trả thực tế so với số tiền vay đã đượcf giải ngân
prev["AMT_CREDIT-d-AMT_ANNUITY"] = prev["AMT_CREDIT"] / prev["AMT_ANNUITY"] # how many month
prev["AMT_GOODS_PRICE-d-AMT_ANNUITY"] = prev["AMT_GOODS_PRICE"] / prev["AMT_ANNUITY"]
prev["AMT_CREDIT-d-AMT_APPLICATION"] = prev["AMT_CREDIT"] / prev["AMT_APPLICATION"]
prev['AMT_GOODS_PRICE-d-AMT_CREDIT'] = prev['AMT_GOODS_PRICE'] / prev['AMT_CREDIT']
prev['AMT_DOWN_PAYMENT-d-AMT_GOODS_PRICE'] = prev["AMT_DOWN_PAYMENT"] / prev["AMT_GOODS_PRICE"]
use_cols = ['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE'] # helpers.config
trte = pd.concat([train, test])[use_cols] # modules.utils.get_trte()
rename_di = {
    'AMT_INCOME_TOTAL':       'app_AMT_INCOME_TOTAL', 
    'AMT_CREDIT':             'app_AMT_CREDIT', 
    'AMT_ANNUITY':            'app_AMT_ANNUITY',
    'AMT_GOODS_PRICE':        'app_AMT_GOODS_PRICE',
    'DAYS_BIRTH':             'app_DAYS_BIRTH', 
    'DAYS_EMPLOYED':          'app_DAYS_EMPLOYED', 
    'DAYS_REGISTRATION':      'app_DAYS_REGISTRATION', 
    'DAYS_ID_PUBLISH':        'app_DAYS_ID_PUBLISH', 
    'DAYS_LAST_PHONE_CHANGE': 'app_DAYS_LAST_PHONE_CHANGE',
    }
trte.rename(columns=rename_di, inplace=True)
app_money_cols = ['app_AMT_INCOME_TOTAL', 'app_AMT_CREDIT', 'app_AMT_ANNUITY', 'app_AMT_GOODS_PRICE']
app_day_cols = ['app_DAYS_BIRTH', 'app_DAYS_EMPLOYED', 'app_DAYS_REGISTRATION', 'app_DAYS_ID_PUBLISH', 'app_DAYS_LAST_PHONE_CHANGE'] 
prev_money_cols = [ 'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE']
prev_rate_cols = ['RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED']
prev_other_cols = ['HOUR_APPR_PROCESS_START', 'SELLERPLACE_AREA']
prev_cnt_cols = ['CNT_PAYMENT']
prev_day_cols = ['DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
prev = prev.merge(trte, on = "SK_ID_CURR", how="left")
prev['AMT_ANNUITY-d-app_AMT_INCOME_TOTAL']     = prev['AMT_ANNUITY']     / prev['app_AMT_INCOME_TOTAL'] # AMT_INCOME_TOTAL là thu nhập MONTHLY. đây là thu nhập tự xưng và có sai số
prev['AMT_APPLICATION-d-app_AMT_INCOME_TOTAL'] = prev['AMT_APPLICATION'] / prev['app_AMT_INCOME_TOTAL']
prev['AMT_CREDIT-d-app_AMT_INCOME_TOTAL']      = prev['AMT_CREDIT']      / prev['app_AMT_INCOME_TOTAL']
prev['AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL'] = prev['AMT_GOODS_PRICE'] / prev['app_AMT_INCOME_TOTAL']
prev['AMT_ANNUITY-s-app_AMT_INCOME_TOTAL']     = prev['AMT_ANNUITY']     - prev['app_AMT_INCOME_TOTAL']
prev['AMT_APPLICATION-s-app_AMT_INCOME_TOTAL'] = prev['AMT_APPLICATION'] - prev['app_AMT_INCOME_TOTAL']
prev['AMT_CREDIT-s-app_AMT_INCOME_TOTAL']      = prev['AMT_CREDIT']      - prev['app_AMT_INCOME_TOTAL']
prev['AMT_GOODS_PRICE-s-app_AMT_INCOME_TOTAL'] = prev['AMT_GOODS_PRICE'] - prev['app_AMT_INCOME_TOTAL']
prev['AMT_ANNUITY-d-app_AMT_CREDIT']     = prev['AMT_ANNUITY']     / prev['app_AMT_CREDIT']
prev['AMT_APPLICATION-d-app_AMT_CREDIT'] = prev['AMT_APPLICATION'] / prev['app_AMT_CREDIT']
prev['AMT_CREDIT-d-app_AMT_CREDIT']      = prev['AMT_CREDIT']      / prev['app_AMT_CREDIT']
prev['AMT_GOODS_PRICE-d-app_AMT_CREDIT'] = prev['AMT_GOODS_PRICE'] / prev['app_AMT_CREDIT']
prev['AMT_ANNUITY-s-app_AMT_CREDIT']     = prev['AMT_ANNUITY']     - prev['app_AMT_CREDIT']
prev['AMT_APPLICATION-s-app_AMT_CREDIT'] = prev['AMT_APPLICATION'] - prev['app_AMT_CREDIT']
prev['AMT_CREDIT-s-app_AMT_CREDIT']      = prev['AMT_CREDIT']      - prev['app_AMT_CREDIT']
prev['AMT_GOODS_PRICE-s-app_AMT_CREDIT'] = prev['AMT_GOODS_PRICE'] - prev['app_AMT_CREDIT']
prev['AMT_ANNUITY-d-app_AMT_ANNUITY']     = prev['AMT_ANNUITY']     / prev['app_AMT_ANNUITY']
prev['AMT_APPLICATION-d-app_AMT_ANNUITY'] = prev['AMT_APPLICATION'] / prev['app_AMT_ANNUITY']
prev['AMT_CREDIT-d-app_AMT_ANNUITY']      = prev['AMT_CREDIT']      / prev['app_AMT_ANNUITY']
prev['AMT_GOODS_PRICE-d-app_AMT_ANNUITY'] = prev['AMT_GOODS_PRICE'] / prev['app_AMT_ANNUITY']
prev['AMT_ANNUITY-s-app_AMT_ANNUITY']     = prev['AMT_ANNUITY']     - prev['app_AMT_ANNUITY']
prev['AMT_APPLICATION-s-app_AMT_ANNUITY'] = prev['AMT_APPLICATION'] - prev['app_AMT_ANNUITY']
prev['AMT_CREDIT-s-app_AMT_ANNUITY']      = prev['AMT_CREDIT']      - prev['app_AMT_ANNUITY']
prev['AMT_GOODS_PRICE-s-app_AMT_ANNUITY'] = prev['AMT_GOODS_PRICE'] - prev['app_AMT_ANNUITY']
prev['AMT_ANNUITY-d-app_AMT_GOODS_PRICE']     = prev['AMT_ANNUITY']     / prev['app_AMT_GOODS_PRICE']
prev['AMT_APPLICATION-d-app_AMT_GOODS_PRICE'] = prev['AMT_APPLICATION'] / prev['app_AMT_GOODS_PRICE']
prev['AMT_CREDIT-d-app_AMT_GOODS_PRICE']      = prev['AMT_CREDIT']      / prev['app_AMT_GOODS_PRICE']
prev['AMT_GOODS_PRICE-d-app_AMT_GOODS_PRICE'] = prev['AMT_GOODS_PRICE'] / prev['app_AMT_GOODS_PRICE']
prev['AMT_ANNUITY-s-app_AMT_GOODS_PRICE']     = prev['AMT_ANNUITY']     - prev['app_AMT_GOODS_PRICE']
prev['AMT_APPLICATION-s-app_AMT_GOODS_PRICE'] = prev['AMT_APPLICATION'] - prev['app_AMT_GOODS_PRICE']
prev['AMT_CREDIT-s-app_AMT_GOODS_PRICE']      = prev['AMT_CREDIT']      - prev['app_AMT_GOODS_PRICE']
prev['AMT_GOODS_PRICE-s-app_AMT_GOODS_PRICE'] = prev['AMT_GOODS_PRICE'] - prev['app_AMT_GOODS_PRICE']
prev['AMT_ANNUITY-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL']     = prev['AMT_ANNUITY-s-app_AMT_CREDIT'] / prev['app_AMT_INCOME_TOTAL']
prev['AMT_APPLICATION-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL'] = prev['AMT_APPLICATION-s-app_AMT_CREDIT'] / prev['app_AMT_INCOME_TOTAL']
prev['AMT_CREDIT-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL']      = prev['AMT_CREDIT-s-app_AMT_CREDIT'] / prev['app_AMT_INCOME_TOTAL']
prev['AMT_GOODS_PRICE-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL'] = prev['AMT_GOODS_PRICE-s-app_AMT_CREDIT'] / prev['app_AMT_INCOME_TOTAL']
prev['AMT_ANNUITY-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL']     = prev['AMT_ANNUITY-s-app_AMT_ANNUITY'] / prev['app_AMT_INCOME_TOTAL']
prev['AMT_APPLICATION-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL'] = prev['AMT_APPLICATION-s-app_AMT_ANNUITY'] / prev['app_AMT_INCOME_TOTAL']
prev['AMT_CREDIT-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL']      = prev['AMT_CREDIT-s-app_AMT_ANNUITY'] / prev['app_AMT_INCOME_TOTAL']
prev['AMT_GOODS_PRICE-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL'] = prev['AMT_GOODS_PRICE-s-app_AMT_ANNUITY'] / prev['app_AMT_INCOME_TOTAL']
prev['AMT_ANNUITY-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL']     = prev['AMT_ANNUITY-s-app_AMT_GOODS_PRICE'] / prev['app_AMT_INCOME_TOTAL']
prev['AMT_APPLICATION-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL'] = prev['AMT_APPLICATION-s-app_AMT_GOODS_PRICE'] / prev['app_AMT_INCOME_TOTAL']
prev['AMT_CREDIT-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL']      = prev['AMT_CREDIT-s-app_AMT_GOODS_PRICE']  / prev['app_AMT_INCOME_TOTAL']
prev['AMT_GOODS_PRICE-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL'] = prev['AMT_GOODS_PRICE-s-app_AMT_GOODS_PRICE'] / prev['app_AMT_INCOME_TOTAL']
f_name='interest_rate'; init_rate=0.9; n_iter=500 # hội tụ hàm số tìm lãi suất
prev['AMT_ANNUITY_d_AMT_CREDIT_temp'] = prev.AMT_ANNUITY / prev.AMT_CREDIT   
prev[f_name] = prev['AMT_ANNUITY_d_AMT_CREDIT_temp']*((1 + init_rate)**prev.CNT_PAYMENT - 1)/((1 + init_rate)**prev.CNT_PAYMENT)
for i in range(n_iter):
    prev[f_name] = prev['AMT_ANNUITY_d_AMT_CREDIT_temp']*((1 + prev[f_name])**prev.CNT_PAYMENT - 1)/((1 + prev[f_name])**prev.CNT_PAYMENT) 
prev.drop(['AMT_ANNUITY_d_AMT_CREDIT_temp'], axis=1, inplace=True)
prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], inplace=True)
prev.reset_index(drop=True, inplace=True)
cols = [ 'total_debt',
'AMT_GOODS_PRICE-d-total_debt',
'AMT_CREDIT-d-total_debt',
'AMT_CREDIT-d-AMT_ANNUITY',
'AMT_GOODS_PRICE-d-AMT_ANNUITY',
'AMT_CREDIT-d-AMT_APPLICATION',
'AMT_GOODS_PRICE-d-AMT_CREDIT',
'AMT_DOWN_PAYMENT-d-AMT_GOODS_PRICE',
'AMT_ANNUITY-d-app_AMT_INCOME_TOTAL',
'AMT_APPLICATION-d-app_AMT_INCOME_TOTAL',
'AMT_CREDIT-d-app_AMT_INCOME_TOTAL',
'AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL',
'AMT_ANNUITY-s-app_AMT_INCOME_TOTAL',
'AMT_APPLICATION-s-app_AMT_INCOME_TOTAL',
'AMT_CREDIT-s-app_AMT_INCOME_TOTAL',
'AMT_GOODS_PRICE-s-app_AMT_INCOME_TOTAL',
'AMT_ANNUITY-d-app_AMT_CREDIT',
'AMT_APPLICATION-d-app_AMT_CREDIT',
'AMT_CREDIT-d-app_AMT_CREDIT',
'AMT_GOODS_PRICE-d-app_AMT_CREDIT',
'AMT_ANNUITY-s-app_AMT_CREDIT',
'AMT_APPLICATION-s-app_AMT_CREDIT',
'AMT_CREDIT-s-app_AMT_CREDIT',
'AMT_GOODS_PRICE-s-app_AMT_CREDIT',
'AMT_ANNUITY-d-app_AMT_ANNUITY',
'AMT_APPLICATION-d-app_AMT_ANNUITY',
'AMT_CREDIT-d-app_AMT_ANNUITY',
'AMT_GOODS_PRICE-d-app_AMT_ANNUITY',
'AMT_ANNUITY-s-app_AMT_ANNUITY',
'AMT_APPLICATION-s-app_AMT_ANNUITY',
'AMT_CREDIT-s-app_AMT_ANNUITY',
'AMT_GOODS_PRICE-s-app_AMT_ANNUITY',
'AMT_ANNUITY-d-app_AMT_GOODS_PRICE',
'AMT_APPLICATION-d-app_AMT_GOODS_PRICE',
'AMT_CREDIT-d-app_AMT_GOODS_PRICE',
'AMT_GOODS_PRICE-d-app_AMT_GOODS_PRICE',
'AMT_ANNUITY-s-app_AMT_GOODS_PRICE',
'AMT_APPLICATION-s-app_AMT_GOODS_PRICE',
'AMT_CREDIT-s-app_AMT_GOODS_PRICE',
'AMT_GOODS_PRICE-s-app_AMT_GOODS_PRICE',
'AMT_ANNUITY-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL',
'AMT_APPLICATION-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL',
'AMT_CREDIT-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL',
'AMT_GOODS_PRICE-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL',
'AMT_ANNUITY-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL',
'AMT_APPLICATION-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL',
'AMT_CREDIT-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL',
'AMT_GOODS_PRICE-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL',
'AMT_ANNUITY-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL',
'AMT_APPLICATION-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL',
'AMT_CREDIT-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL',
'AMT_GOODS_PRICE-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL',
'interest_rate']
df_list = []
for col in cols:
    df = multi.multi(col) # diff pctchange
    df_list.append(df)
df = pd.concat(df_list, axis=1)
prev = pd.concat([prev, df], axis=1)
for c1 in prev_day_cols:
    for c2 in app_day_cols:
        prev[f'{c1}-s-{c2}'] = prev[c1] - prev[c2]
        prev[f'{c1}-d-{c2}'] = prev[c1] / prev[c2]
_keep_vars.update(["prev"])
del prev_tmp
cache_clear()
prev['cnt_paid'] = prev.apply(lambda x: min( np.ceil((x['DAYS_FIRST_DUE']/-30) + 1), x['CNT_PAYMENT'] ), axis=1)
prev['cnt_paid_ratio'] = prev['cnt_paid'] / prev['CNT_PAYMENT']
prev['cnt_unpaid'] = prev['CNT_PAYMENT'] - prev['cnt_paid']
prev['amt_paid'] = prev['AMT_ANNUITY'] * prev['cnt_paid'] # thực tế đã trả
prev['amt_unpaid'] = prev['total_debt'] - prev['amt_paid'] # chưa trả hết
prev['active'] = (prev['cnt_unpaid']>0)*1 # chưa trả xong
prev['completed'] = (prev['cnt_unpaid']==0)*1 # đã complete
prev_tmp = pd.DataFrame()
rem_max = prev['cnt_unpaid'].max() # 79 
prev['cnt_unpaid_tmp'] = prev['cnt_unpaid']
for i in range(int( rem_max )):
    c = f'future_payment_{i+1}m'
    prev_tmp[c] = prev['cnt_unpaid_tmp'].map(lambda x: min(x, 1)) * prev['AMT_ANNUITY']
    prev_tmp.loc[prev_tmp[c]==0, c] = np.nan
    prev['cnt_unpaid_tmp'] = prev['cnt_unpaid_tmp'].map(lambda x: max(x - 1, 0))
del prev['cnt_unpaid_tmp']
prev = pd.concat([prev, prev_tmp], axis=1)
batch_size = 500000 # chia batch
n = len(prev)
for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    batch = prev.iloc[start:end]
    rem_max = int(batch['cnt_paid'].max())
    cnt_paid_matrix = np.maximum(batch['cnt_paid'].values[:, None] - np.arange(rem_max), 0)
    cnt_paid_matrix = (cnt_paid_matrix > 0).astype(np.float32) * batch['AMT_ANNUITY'].values[:, None]
    cnt_paid_matrix[cnt_paid_matrix == 0] = np.nan
    cols = [f'past_payment_{i+1}m' for i in range(rem_max)]
    batch_tmp = pd.DataFrame(cnt_paid_matrix, columns=cols, index=batch.index)
    batch = pd.concat([batch, batch_tmp], axis=1)
    batch.replace(np.inf, np.nan, inplace=True)
    batch.replace(-np.inf, np.nan, inplace=True)
    batch.to_pickle(ROOT + f'/data/processed/f101_prev_batch_{start // batch_size + 1}.p')
    del batch
    gc.collect()
tmp = pd.read_pickle(ROOT + "/data/processed/prev_batch_1.p")
cache_clear()
