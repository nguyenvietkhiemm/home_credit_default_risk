import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from config import ROOT, avg_cols, mode_cols, medi_cols, req_bureau_cols, imputation_paths  # lib này được khởi tạo ban đầu dự án
import modules.utils as utils
import modules.cpp as cpp
import time
from helpers.cache_clear import cache_clear

get_pickle = utils.get_pickle
get_pickles = utils.get_pickles

_keep_vars = set(globals().keys())  # lưu biến gốc

def calc_possible_rates(trte):
    N = len(trte)
    amt_annuity = trte["AMT_ANNUITY"].values.astype(np.float64)
    amt_credit = trte["AMT_CREDIT"].values.astype(np.float64)
    
    possible_cnts = range(6, 84, 6)
    rate_matrix = []

    for cnt in possible_cnts:
        cnts = np.full(N, cnt, dtype=np.int32)
        rates = np.zeros(N, dtype=np.float64)
        cpp.interest_rate(amt_annuity, amt_credit, cnts, rates, N)
        rate_matrix.append(rates)
    
    rate_matrix = np.array(rate_matrix)
    
    return pd.DataFrame({
        'interest_rate_min': np.nanmin(rate_matrix, axis=0),
        'interest_rate_max': np.nanmax(rate_matrix, axis=0),
        'interest_rate_median': np.nanmedian(rate_matrix, axis=0),
        'interest_rate_std': np.nanstd(rate_matrix, axis=0)
    }, index=trte.index)

def pred_interest_rate():
    start_time = time.time()

    train = get_pickle("train")
    test = get_pickle("test")
    trte = pd.concat([train, test])
    
    # prev = get_pickles("prev", ['SK_ID_CURR', "SK_ID_PREV", 'NAME_CONTRACT_TYPE', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', "DAYS_DECISION", "interest_rate", "CNT_PAYMENT", "NAME_TYPE_SUITE", "WEEKDAY_APPR_PROCESS_START",	"HOUR_APPR_PROCESS_START"])

    interest_rate_features = calc_possible_rates(trte)
    
    interest_rate_features.reset_index(drop=True, inplace=True)
    trte.reset_index(drop=True, inplace=True)
    
    trte = pd.concat([trte, interest_rate_features], axis=1)
    
    train = trte[~trte["TARGET"].isnull()]
    test = trte[trte["TARGET"].isnull()]
    
    utils.to_pickle(train[["SK_ID_CURR", "interest_rate_min", "interest_rate_max", "interest_rate_median", "interest_rate_std"]], name="train", dir="imputation", file_name="interest_rate") 
    utils.to_pickle(test[["SK_ID_CURR", "interest_rate_min", "interest_rate_max", "interest_rate_median", "interest_rate_std"]], name="test", dir="imputation", file_name="interest_rate") 
    
    end_time = time.time()

    print(f"interest_rate time: {end_time - start_time}s")
    
    # only python: 136s với 1 cnt
    # python + njit: 76,67s với 1 cnt
    # c++: 67s với 1 cnt
    # c++ tối ưu vector và ngắt với ngưỡng delta hội tụ: 1.3s ?? và ~4s cho possible_cnt