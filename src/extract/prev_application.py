import pandas as pd
import numpy as np
import os

from config import ROOT, rename_di, use_cols # lib này được khởi tạo ban đầu dự án
import modules.utils as utils
import modules.multi as multi

from helpers.cache_clear import cache_clear
get_pickle = utils.get_pickle
_keep_vars = set(globals().keys())  # lưu biến gốc

def prev_extract(test_run = False):
    if test_run:
        print("extract prev")
        for path in utils.get_pickle_paths(name="prev"):
            print(path)
        return
    
    prev = utils.get_pickle("prev")
    
    prev['AMT_APPLICATION'] = prev['AMT_APPLICATION'].replace(0, np.nan)
    prev['AMT_CREDIT'] = prev['AMT_CREDIT'].replace(0, np.nan)
    prev['CNT_PAYMENT'] = prev['CNT_PAYMENT'].replace(0, np.nan)
    prev['AMT_DOWN_PAYMENT'] = prev['AMT_DOWN_PAYMENT'].replace(np.nan, 0)
    prev['RATE_DOWN_PAYMENT'] = prev['RATE_DOWN_PAYMENT'].replace(np.nan, 0)
    prev['FLAG_LAST_APPL_PER_CONTRACT'] = (prev['FLAG_LAST_APPL_PER_CONTRACT'] == 'Y') * 1  # biến nhị phân
    for c in ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']:
        prev.loc[prev[c] == 365243, c] = np.nan
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
    prev['AMT_CREDIT-d-total_debt'] = prev['AMT_CREDIT'] / prev['total_debt']  # so sánh số tiền phải trả thực tế so với số tiền vay đã đượcf giải ngân
    prev["AMT_CREDIT-d-AMT_ANNUITY"] = prev["AMT_CREDIT"] / prev["AMT_ANNUITY"]  # how many month
    prev["AMT_GOODS_PRICE-d-AMT_ANNUITY"] = prev["AMT_GOODS_PRICE"] / prev["AMT_ANNUITY"]
    prev["AMT_CREDIT-d-AMT_APPLICATION"] = prev["AMT_CREDIT"] / prev["AMT_APPLICATION"]
    prev['AMT_GOODS_PRICE-d-AMT_CREDIT'] = prev['AMT_GOODS_PRICE'] / prev['AMT_CREDIT']
    prev['AMT_DOWN_PAYMENT-d-AMT_GOODS_PRICE'] = prev["AMT_DOWN_PAYMENT"] / prev["AMT_GOODS_PRICE"]
    
    
    # app
    trte = utils.get_trte()  # modules.utils.get_trte()
    app_day_cols = ['app_DAYS_BIRTH', 'app_DAYS_EMPLOYED', 'app_DAYS_REGISTRATION', 'app_DAYS_ID_PUBLISH', 'app_DAYS_LAST_PHONE_CHANGE']
    prev_day_cols = ['DAYS_DECISION', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    prev = prev.merge(trte, on="SK_ID_CURR", how="left")
    prev['AMT_ANNUITY-d-app_AMT_INCOME_TOTAL'] = prev['AMT_ANNUITY'] / prev['app_AMT_INCOME_TOTAL']  # AMT_INCOME_TOTAL là thu nhập MONTHLY. đây là thu nhập tự xưng và có sai số
    prev['AMT_APPLICATION-d-app_AMT_INCOME_TOTAL'] = prev['AMT_APPLICATION'] / prev['app_AMT_INCOME_TOTAL']
    prev['AMT_CREDIT-d-app_AMT_INCOME_TOTAL'] = prev['AMT_CREDIT'] / prev['app_AMT_INCOME_TOTAL']
    prev['AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL'] = prev['AMT_GOODS_PRICE'] / prev['app_AMT_INCOME_TOTAL']
    prev['AMT_ANNUITY-s-app_AMT_INCOME_TOTAL'] = prev['AMT_ANNUITY'] - prev['app_AMT_INCOME_TOTAL']
    prev['AMT_APPLICATION-s-app_AMT_INCOME_TOTAL'] = prev['AMT_APPLICATION'] - prev['app_AMT_INCOME_TOTAL']
    prev['AMT_CREDIT-s-app_AMT_INCOME_TOTAL'] = prev['AMT_CREDIT'] - prev['app_AMT_INCOME_TOTAL']
    prev['AMT_GOODS_PRICE-s-app_AMT_INCOME_TOTAL'] = prev['AMT_GOODS_PRICE'] - prev['app_AMT_INCOME_TOTAL']
    prev['AMT_ANNUITY-d-app_AMT_CREDIT'] = prev['AMT_ANNUITY'] / prev['app_AMT_CREDIT']
    prev['AMT_APPLICATION-d-app_AMT_CREDIT'] = prev['AMT_APPLICATION'] / prev['app_AMT_CREDIT']
    prev['AMT_CREDIT-d-app_AMT_CREDIT'] = prev['AMT_CREDIT'] / prev['app_AMT_CREDIT']
    prev['AMT_GOODS_PRICE-d-app_AMT_CREDIT'] = prev['AMT_GOODS_PRICE'] / prev['app_AMT_CREDIT']
    prev['AMT_ANNUITY-s-app_AMT_CREDIT'] = prev['AMT_ANNUITY'] - prev['app_AMT_CREDIT']
    prev['AMT_APPLICATION-s-app_AMT_CREDIT'] = prev['AMT_APPLICATION'] - prev['app_AMT_CREDIT']
    prev['AMT_CREDIT-s-app_AMT_CREDIT'] = prev['AMT_CREDIT'] - prev['app_AMT_CREDIT']
    prev['AMT_GOODS_PRICE-s-app_AMT_CREDIT'] = prev['AMT_GOODS_PRICE'] - prev['app_AMT_CREDIT']
    prev['AMT_ANNUITY-d-app_AMT_ANNUITY'] = prev['AMT_ANNUITY'] / prev['app_AMT_ANNUITY']
    prev['AMT_APPLICATION-d-app_AMT_ANNUITY'] = prev['AMT_APPLICATION'] / prev['app_AMT_ANNUITY']
    prev['AMT_CREDIT-d-app_AMT_ANNUITY'] = prev['AMT_CREDIT'] / prev['app_AMT_ANNUITY']
    prev['AMT_GOODS_PRICE-d-app_AMT_ANNUITY'] = prev['AMT_GOODS_PRICE'] / prev['app_AMT_ANNUITY']
    prev['AMT_ANNUITY-s-app_AMT_ANNUITY'] = prev['AMT_ANNUITY'] - prev['app_AMT_ANNUITY']
    prev['AMT_APPLICATION-s-app_AMT_ANNUITY'] = prev['AMT_APPLICATION'] - prev['app_AMT_ANNUITY']
    prev['AMT_CREDIT-s-app_AMT_ANNUITY'] = prev['AMT_CREDIT'] - prev['app_AMT_ANNUITY']
    prev['AMT_GOODS_PRICE-s-app_AMT_ANNUITY'] = prev['AMT_GOODS_PRICE'] - prev['app_AMT_ANNUITY']
    prev['AMT_ANNUITY-d-app_AMT_GOODS_PRICE'] = prev['AMT_ANNUITY'] / prev['app_AMT_GOODS_PRICE']
    prev['AMT_APPLICATION-d-app_AMT_GOODS_PRICE'] = prev['AMT_APPLICATION'] / prev['app_AMT_GOODS_PRICE']
    prev['AMT_CREDIT-d-app_AMT_GOODS_PRICE'] = prev['AMT_CREDIT'] / prev['app_AMT_GOODS_PRICE']
    prev['AMT_GOODS_PRICE-d-app_AMT_GOODS_PRICE'] = prev['AMT_GOODS_PRICE'] / prev['app_AMT_GOODS_PRICE']
    prev['AMT_ANNUITY-s-app_AMT_GOODS_PRICE'] = prev['AMT_ANNUITY'] - prev['app_AMT_GOODS_PRICE']
    prev['AMT_APPLICATION-s-app_AMT_GOODS_PRICE'] = prev['AMT_APPLICATION'] - prev['app_AMT_GOODS_PRICE']
    prev['AMT_CREDIT-s-app_AMT_GOODS_PRICE'] = prev['AMT_CREDIT'] - prev['app_AMT_GOODS_PRICE']
    prev['AMT_GOODS_PRICE-s-app_AMT_GOODS_PRICE'] = prev['AMT_GOODS_PRICE'] - prev['app_AMT_GOODS_PRICE']
    prev['AMT_ANNUITY-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL'] = prev['AMT_ANNUITY-s-app_AMT_CREDIT'] / prev['app_AMT_INCOME_TOTAL']
    prev['AMT_APPLICATION-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL'] = prev['AMT_APPLICATION-s-app_AMT_CREDIT'] / prev['app_AMT_INCOME_TOTAL']
    prev['AMT_CREDIT-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL'] = prev['AMT_CREDIT-s-app_AMT_CREDIT'] / prev['app_AMT_INCOME_TOTAL']
    prev['AMT_GOODS_PRICE-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL'] = prev['AMT_GOODS_PRICE-s-app_AMT_CREDIT'] / prev['app_AMT_INCOME_TOTAL']
    prev['AMT_ANNUITY-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL'] = prev['AMT_ANNUITY-s-app_AMT_ANNUITY'] / prev['app_AMT_INCOME_TOTAL']
    prev['AMT_APPLICATION-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL'] = prev['AMT_APPLICATION-s-app_AMT_ANNUITY'] / prev['app_AMT_INCOME_TOTAL']
    prev['AMT_CREDIT-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL'] = prev['AMT_CREDIT-s-app_AMT_ANNUITY'] / prev['app_AMT_INCOME_TOTAL']
    prev['AMT_GOODS_PRICE-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL'] = prev['AMT_GOODS_PRICE-s-app_AMT_ANNUITY'] / prev['app_AMT_INCOME_TOTAL']
    prev['AMT_ANNUITY-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL'] = prev['AMT_ANNUITY-s-app_AMT_GOODS_PRICE'] / prev['app_AMT_INCOME_TOTAL']
    prev['AMT_APPLICATION-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL'] = prev['AMT_APPLICATION-s-app_AMT_GOODS_PRICE'] / prev['app_AMT_INCOME_TOTAL']
    prev['AMT_CREDIT-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL'] = prev['AMT_CREDIT-s-app_AMT_GOODS_PRICE'] / prev['app_AMT_INCOME_TOTAL']
    prev['AMT_GOODS_PRICE-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL'] = prev['AMT_GOODS_PRICE-s-app_AMT_GOODS_PRICE'] / prev['app_AMT_INCOME_TOTAL']
    f_name = 'interest_rate'
    init_rate = 0.9
    n_iter = 500  # hội tụ hàm số tìm lãi suất
    prev['AMT_ANNUITY_d_AMT_CREDIT_temp'] = prev.AMT_ANNUITY / prev.AMT_CREDIT
    prev[f_name] = prev['AMT_ANNUITY_d_AMT_CREDIT_temp'] * ((1 + init_rate)**prev.CNT_PAYMENT - 1) / ((1 + init_rate)**prev.CNT_PAYMENT)
    for i in range(n_iter):
        prev[f_name] = prev['AMT_ANNUITY_d_AMT_CREDIT_temp'] * ((1 + prev[f_name])**prev.CNT_PAYMENT - 1) / ((1 + prev[f_name])**prev.CNT_PAYMENT)
    prev.drop(['AMT_ANNUITY_d_AMT_CREDIT_temp'], axis=1, inplace=True)
    prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], inplace=True)
    prev.reset_index(drop=True, inplace=True)
    cols = [
        'total_debt', 'AMT_GOODS_PRICE-d-total_debt', 'AMT_CREDIT-d-total_debt', 'AMT_CREDIT-d-AMT_ANNUITY', 'AMT_GOODS_PRICE-d-AMT_ANNUITY', 'AMT_CREDIT-d-AMT_APPLICATION',
        'AMT_GOODS_PRICE-d-AMT_CREDIT', 'AMT_DOWN_PAYMENT-d-AMT_GOODS_PRICE', 'AMT_ANNUITY-d-app_AMT_INCOME_TOTAL', 'AMT_APPLICATION-d-app_AMT_INCOME_TOTAL', 'AMT_CREDIT-d-app_AMT_INCOME_TOTAL',
        'AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL', 'AMT_ANNUITY-s-app_AMT_INCOME_TOTAL', 'AMT_APPLICATION-s-app_AMT_INCOME_TOTAL', 'AMT_CREDIT-s-app_AMT_INCOME_TOTAL',
        'AMT_GOODS_PRICE-s-app_AMT_INCOME_TOTAL', 'AMT_ANNUITY-d-app_AMT_CREDIT', 'AMT_APPLICATION-d-app_AMT_CREDIT', 'AMT_CREDIT-d-app_AMT_CREDIT', 'AMT_GOODS_PRICE-d-app_AMT_CREDIT',
        'AMT_ANNUITY-s-app_AMT_CREDIT', 'AMT_APPLICATION-s-app_AMT_CREDIT', 'AMT_CREDIT-s-app_AMT_CREDIT', 'AMT_GOODS_PRICE-s-app_AMT_CREDIT', 'AMT_ANNUITY-d-app_AMT_ANNUITY',
        'AMT_APPLICATION-d-app_AMT_ANNUITY', 'AMT_CREDIT-d-app_AMT_ANNUITY', 'AMT_GOODS_PRICE-d-app_AMT_ANNUITY', 'AMT_ANNUITY-s-app_AMT_ANNUITY', 'AMT_APPLICATION-s-app_AMT_ANNUITY',
        'AMT_CREDIT-s-app_AMT_ANNUITY', 'AMT_GOODS_PRICE-s-app_AMT_ANNUITY', 'AMT_ANNUITY-d-app_AMT_GOODS_PRICE', 'AMT_APPLICATION-d-app_AMT_GOODS_PRICE', 'AMT_CREDIT-d-app_AMT_GOODS_PRICE',
        'AMT_GOODS_PRICE-d-app_AMT_GOODS_PRICE', 'AMT_ANNUITY-s-app_AMT_GOODS_PRICE', 'AMT_APPLICATION-s-app_AMT_GOODS_PRICE', 'AMT_CREDIT-s-app_AMT_GOODS_PRICE', 'AMT_GOODS_PRICE-s-app_AMT_GOODS_PRICE',
        'AMT_ANNUITY-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL', 'AMT_APPLICATION-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL', 'AMT_CREDIT-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL',
        'AMT_GOODS_PRICE-s-app_AMT_CREDIT-d-app_AMT_INCOME_TOTAL', 'AMT_ANNUITY-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL', 'AMT_APPLICATION-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL',
        'AMT_CREDIT-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE-s-app_AMT_ANNUITY-d-app_AMT_INCOME_TOTAL', 'AMT_ANNUITY-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL',
        'AMT_APPLICATION-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL', 'AMT_CREDIT-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE-m-app_AMT_GOODS_PRICE-d-app_AMT_INCOME_TOTAL',
        'interest_rate'
    ]
    df_list = []
    for col in cols:
        df = multi.multi(col, prev)  # diff pctchange
        df_list.append(df)
    df = pd.concat(df_list, axis=1)
    prev = pd.concat([prev, df], axis=1)
    for c1 in prev_day_cols:
        for c2 in app_day_cols:
            prev[f'{c1}-s-{c2}'] = prev[c1] - prev[c2]
            prev[f'{c1}-d-{c2}'] = prev[c1] / prev[c2]
    _keep_vars.update(["prev"])
    cache_clear(globals(), _keep_vars)
    prev["DAYS_FIRST_DUE"].min() / 30
    prev['cnt_paid'] = prev.apply(lambda x: min(np.ceil((x['DAYS_FIRST_DUE'] / -30) + 1), x['CNT_PAYMENT']), axis=1)
    prev['cnt_paid_ratio'] = prev['cnt_paid'] / prev['CNT_PAYMENT']
    prev['cnt_unpaid'] = prev['CNT_PAYMENT'] - prev['cnt_paid']
    prev['amt_paid'] = prev['AMT_ANNUITY'] * prev['cnt_paid']  # thực tế đã trả
    prev['amt_unpaid'] = prev['total_debt'] - prev['amt_paid']  # chưa trả hết
    prev['active'] = (prev['cnt_unpaid'] > 0) * 1  # chưa trả xong
    prev['completed'] = (prev['cnt_unpaid'] == 0) * 1  # đã complete
    rem_max_unpaid = int(prev['cnt_unpaid'].max())
    rem_max_paid = int(prev['cnt_paid'].max())
    utils.to_pickles(prev, "prev")
    from config import processed_paths
    processed_paths["prev"]
    
    def process_file(path):
        prev = pd.read_pickle(path)
        # print(path)
        cnt_unpaid = prev['cnt_unpaid'].values
        cnt_paid = prev['cnt_paid'].values
        amt_annuity = prev['AMT_ANNUITY'].values
        future_mask = np.arange(rem_max_unpaid) < cnt_unpaid[:, None]
        future_amt = np.where(future_mask, amt_annuity[:, None], np.nan)
        future_cols = [f'future_payment_{i+1}m' for i in range(rem_max_unpaid)]
        past_mask = np.arange(rem_max_paid) < cnt_paid[:, None]
        past_amt = np.where(past_mask, amt_annuity[:, None], np.nan)
        past_cols = [f'past_payment_{i+1}m' for i in range(rem_max_paid)]
        df_future = pd.DataFrame(future_amt, columns=future_cols, index=prev.index)
        df_past = pd.DataFrame(past_amt, columns=past_cols, index=prev.index)
        prev = pd.concat([prev, df_past, df_future], axis=1)

        return prev

    for path in utils.get_pickle_paths(name="prev"):
        process_file(path).to_pickle(path)

    print("extract prev")
    
    cache_clear(globals())
