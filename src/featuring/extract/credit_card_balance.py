import pandas as pd
import numpy as np
import gc
import os

from config import ROOT, use_cols, prev_use_cols, app_day_cols, app_money_cols  # lib này được khởi tạo ban đầu dự án
import modules.utils as utils
import modules.multi as multi
from helpers.cache_clear import cache_clear

get_pickle = utils.get_pickle
_keep_vars = set(globals().keys())  # lưu biến gốc


def credit_balance_extract(test_run=False):
    if test_run:
        print("extract credit balance")
        for path in utils.get_pickle_paths(name="credit_card"):
            print(path)
        return

    credit_balance = get_pickle("credit_card")
    _keep_vars.update(["credit_balance"])

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

    trte = utils.get_trte()
    trte[app_day_cols] = trte[app_day_cols] / 30  # get month
    credit_balance = pd.merge(credit_balance, trte, on="SK_ID_CURR", how="left")
    
    cache_clear(globals(), _keep_vars)
    
    credit_balance['AMT_BALANCE-d-app_AMT_INCOME_TOTAL'] = credit_balance['AMT_BALANCE'] / credit_balance['app_AMT_INCOME_TOTAL']
    credit_balance['AMT_BALANCE-d-app_AMT_CREDIT'] = credit_balance['AMT_BALANCE'] / credit_balance['app_AMT_CREDIT']
    credit_balance['AMT_BALANCE-d-app_AMT_ANNUITY'] = credit_balance['AMT_BALANCE'] / credit_balance['app_AMT_ANNUITY']
    credit_balance['AMT_BALANCE-d-app_AMT_GOODS_PRICE'] = credit_balance['AMT_BALANCE'] / credit_balance['app_AMT_GOODS_PRICE']
    credit_balance['AMT_DRAWINGS_CURRENT-d-app_AMT_INCOME_TOTAL'] = credit_balance['AMT_DRAWINGS_CURRENT'] / credit_balance['app_AMT_INCOME_TOTAL']
    credit_balance['AMT_DRAWINGS_CURRENT-d-app_AMT_CREDIT'] = credit_balance['AMT_DRAWINGS_CURRENT'] / credit_balance['app_AMT_CREDIT']
    credit_balance['AMT_DRAWINGS_CURRENT-d-app_AMT_ANNUITY'] = credit_balance['AMT_DRAWINGS_CURRENT'] / credit_balance['app_AMT_ANNUITY']
    credit_balance['AMT_DRAWINGS_CURRENT-d-app_AMT_GOODS_PRICE'] = credit_balance['AMT_DRAWINGS_CURRENT'] / credit_balance['app_AMT_GOODS_PRICE']
    
    for c in app_day_cols:
        credit_balance[f'MONTHS_BALANCE-s-{c}'] = credit_balance['MONTHS_BALANCE'] - credit_balance[c]
        
    credit_balance.drop(app_day_cols + app_money_cols, inplace=True, axis=1)
    
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
    
    utils.to_pickles(credit_balance, "credit_card")

    for path in utils.get_pickle_paths(name="credit_card"):
        credit_balance = pd.read_pickle(path)
        diff_pctchange = multi.diff_pctchange(cols, credit_balance)
        
        merged = credit_balance.join(diff_pctchange)
        
        merged.replace(np.inf, np.nan, inplace=True)
        merged.replace(-np.inf, np.nan, inplace=True)
        
        merged.to_pickle(path)
    cache_clear(globals())
