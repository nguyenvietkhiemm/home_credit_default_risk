import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import gc
import io
import os
from itertools import combinations

import logging
from multiprocessing import Pool, Value
from multiprocessing_logging import install_mp_handler

from IPython.display import display

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

pd.reset_option('display.float_format')
pd.set_option('display.max_colwidth', None)

from config import ROOT, prev_num_aggregations  # lib này được khởi tạo ban đầu dự án

import helpers.view as view
import helpers.EDA as EDA
import modules.utils as utils

importlib.reload(view)
importlib.reload(EDA)
importlib.reload(utils)

from helpers.cache_clear import cache_clear

get_pickle = utils.get_pickle
get_pickles = utils.get_pickles

KEY = "SK_ID_CURR"

col_binary = [
            # 'NAME_CONTRACT_TYPE', 
            'NAME_CONTRACT_STATUS', 
              'CODE_REJECT_REASON',
            #   'NAME_YIELD_GROUP', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 
            #   'NAME_PRODUCT_TYPE', 'NAME_SELLER_INDUSTRY', 'CHANNEL_TYPE',
            #   'NAME_PAYMENT_TYPE'
            ]

prev = utils.get_pickles("prev", cols=[KEY, "DAYS_DECISION"]+col_binary)

prev.sort_values(['SK_ID_CURR', 'DAYS_DECISION'], inplace=True) # top latest

col_binary_di = {}

for c in col_binary:
    col_binary_di[c] = list(prev[c].unique())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

install_mp_handler()

def to_decimal(x):
    if len(x) == 0:
        return -1
    return float(str(x[0]) + '.' + ''.join(map(str, x[1:])))

counter = Value('i', 0) 

def multi(args):
    df, group_id = args
    is_app = (df['NAME_CONTRACT_STATUS'] == 'Approved')
    is_ref = (df['NAME_CONTRACT_STATUS'] == 'Refused')
    is_appref = is_app | is_ref

    di = {}
    for c in col_binary:
        for v in col_binary_di[c]:
            arr = (df[c] == v).astype(int).values
            
            arr_app = arr[is_app.values]
            arr_ref = arr[is_ref.values]
            arr_appref = arr[is_appref.values]
            
            di[f'{c}-{v}'] = to_decimal(arr)
            di[f'{c}-{v}_app'] = to_decimal(arr_app)
            di[f'{c}-{v}_ref'] = to_decimal(arr_ref)
            di[f'{c}-{v}_appref'] = to_decimal(arr_appref)

            with counter.get_lock():  # lock để an toàn
                counter.value += 1
                logger.info(f"group_id {group_id}, count {counter.value}")
    return pd.Series(di)

if __name__ == '__main__':
    NTHREAD = 8
    grouped = list(prev.groupby(KEY))
    ids = [(group, key) for key, group in grouped] 

    pool = Pool(NTHREAD)
    callback = pool.map(multi, ids)

    pool.close()
    pool.join()

    base = pd.concat(callback, axis=1).T
    base.reset_index(inplace=True)
    # base = pd.DataFrame()
    base.to_pickle(ROOT + "/data/feature/base.p")