import pandas as pd
import numpy as np
import lightgbm as lgb
import multiprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

from config import ROOT, avg_cols, mode_cols, medi_cols, req_bureau_cols, imputation_paths  # lib này được khởi tạo ban đầu dự án

import modules.utils as utils
import modules.cpp as cpp
import modules.encode as encode
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


predict_interest_rate_cols = ['NAME_CONTRACT_TYPE', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', "NAME_TYPE_SUITE", "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START", "SK_ID_BUREAU"]

bureau = utils.get_pickle("bureau")
bureau = bureau.groupby("SK_ID_CURR")[["SK_ID_BUREAU"]].max().reset_index()

def train_model():
    prev = get_pickles("prev", ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', "interest_rate", "CNT_PAYMENT", "NAME_TYPE_SUITE", "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START"])
    prev=prev.dropna(subset=['AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'CNT_PAYMENT'])
    
    prev = pd.merge(prev, bureau, on="SK_ID_CURR",how="left")
    
    X = prev[predict_interest_rate_cols]
    y = prev["interest_rate"]
        
    X = encode.label_encode(X, categorical_features=['NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE', 'WEEKDAY_APPR_PROCESS_START'])
    
    for col in prev.select_dtypes(include=['object']).columns:
        prev[col] = prev[col].astype('category')
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_val, y_val)
    
    SEED = 71
    param = {
        'objective': 'regression',       
        'metric': 'rmse',                
        'learning_rate': 0.05,
        'max_depth': -1,
        'num_leaves': 255,
        'max_bin': 255,
        'colsample_bytree': 0.5,
        'subsample': 0.5,
        'nthread': multiprocessing.cpu_count(),
        'bagging_freq': 1,
        'seed': SEED
    }
    
    model = lgb.train(param, dtrain, valid_sets=[dval],
                        num_boost_round=5000,
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"single LightGBM model RMSE: {rmse:.4f}")
    joblib.dump(model, ROOT + '/models/rmse_0.0076_interest_rate_model.pkl') # tạm

def pred_interest_rate(train_model=False):
    train = get_pickle("train")
    test = get_pickle("test")
    
    train['data'] = 1
    test['data'] = 0
    
    trte = pd.concat([train, test])

    interest_rate_features = calc_possible_rates(trte)
    
    interest_rate_features.reset_index(drop=True, inplace=True)
    trte.reset_index(drop=True, inplace=True)
    
    if train_model:
        train_model()
    
    model = joblib.load(ROOT + '/models/rmse_0.0076_interest_rate_model.pkl')
        
    trte = pd.merge(trte, bureau, on="SK_ID_CURR", how="left")
    
    data = trte[["data", "SK_ID_CURR"]]
    trte = trte[predict_interest_rate_cols]
    
    trte = encode.label_encode(trte, categorical_features=['NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE', 'WEEKDAY_APPR_PROCESS_START'])
    
    for col in trte.select_dtypes(include=['object']).columns:
        trte[col] = trte[col].astype('category')
    
    pred = model.predict(trte)
    
    trte = pd.concat([trte, interest_rate_features], axis=1)
    trte["pred_interest_rate"] = pred
    trte[["data", "SK_ID_CURR"]] = data
    
    utils.to_pickle(trte[trte["data"]==1][["SK_ID_CURR", "pred_interest_rate", "interest_rate_min", "interest_rate_max", "interest_rate_median", "interest_rate_std"]], name="train", dir="imputation", file_name="interest_rate") 
    utils.to_pickle(trte[trte["data"]==0][["SK_ID_CURR", "pred_interest_rate", "interest_rate_min", "interest_rate_max", "interest_rate_median", "interest_rate_std"]], name="test", dir="imputation", file_name="interest_rate") 