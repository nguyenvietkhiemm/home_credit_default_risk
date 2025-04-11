import pandas as pd
import numpy as np
from helpers.config import use_cols, app_money_cols, app_day_cols

def get_trte(train, test):
    trte = pd.concat([train, test])[use_cols]
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
    
    return trte