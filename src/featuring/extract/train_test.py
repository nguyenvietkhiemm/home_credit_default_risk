import pandas as pd
import numpy as np
from itertools import combinations
import warnings


from config import ROOT, avg_cols, mode_cols, medi_cols, req_bureau_cols  # lib này được khởi tạo ban đầu dự án
import modules.utils as utils

from helpers.cache_clear import cache_clear

_keep_vars = set(globals().keys())  # lưu biến gốc
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def preprocessing(trte):
    trte.loc[trte['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan  # 1000 năm. thay bằng null
    trte['CODE_GENDER'] = 1 - (trte['CODE_GENDER'] == 'F') * 1  # 4 'XNA' thay bằng 'M' => 1
    trte['FLAG_OWN_CAR'] = (trte['FLAG_OWN_CAR'] == 'Y') * 1
    trte['FLAG_OWN_REALTY'] = (trte['FLAG_OWN_REALTY'] == 'Y') * 1
    trte['EMERGENCYSTATE_MODE'] = (trte['EMERGENCYSTATE_MODE'] == 'Yes') * 1

    return trte


def flag_docs_extract(trte):    
    docs = [_f for _f in trte.columns if 'FLAG_DOC' in _f]  # lấy FLAG DOCUMENT
    live = [_f for _f in trte.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f)]  # lấy FLAG trừ FLAG DOCUMENT
    trte['alldocs_kurt'] = trte[docs].kurtosis(axis=1)
    trte['alldocs_skew'] = trte[docs].skew(axis=1)
    trte['alldocs_mean'] = trte[docs].mean(axis=1)
    trte['alldocs_sum'] = trte[docs].sum(axis=1)
    trte['alldocs_std'] = trte[docs].std(axis=1)
    trte['alllives_sum'] = trte[live].sum(axis=1)

    return trte


def organization_type_extract(trte):    
    inc_by_org = trte[['AMT_INCOME_TOTAL',
                       'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']  # trung vị tổng thu nhập group by ORGANIZATION_TYPE (ORGANIZATION_TYPE quá nhiều giá trị)
    trte['AMT_INCOME_TOTAL_by_ORGANIZATION_TYPE'] = trte['ORGANIZATION_TYPE'].map(inc_by_org)

    return trte


def train_test_extract(test_run=False):
    if test_run:
        print("extract train test")
        for path in utils.get_pickle_paths(name="train"):
            print(path)
        return

    train = utils.get_pickle("train")
    test = utils.get_pickle("test")

    train["data"] = 1
    test["data"] = 0
    trte = pd.concat([train, test], ignore_index=True)
    trte = preprocessing(trte)

    trte = flag_docs_extract(trte)
    trte = organization_type_extract(trte)

    trte['AMT_CREDIT-d-AMT_INCOME_TOTAL'] = trte['AMT_CREDIT'] / trte['AMT_INCOME_TOTAL']
    trte['AMT_ANNUITY-d-AMT_INCOME_TOTAL'] = trte['AMT_ANNUITY'] / trte['AMT_INCOME_TOTAL']
    trte['AMT_GOODS_PRICE-d-AMT_INCOME_TOTAL'] = trte['AMT_GOODS_PRICE'] / trte['AMT_INCOME_TOTAL']
    trte['AMT_CREDIT-d-AMT_ANNUITY'] = trte['AMT_CREDIT'] / trte['AMT_ANNUITY']  # số tháng trả hết lý thuyết dành cho tín dụng
    trte['AMT_ANNUITY-d-AMT_CREDIT'] = trte['AMT_ANNUITY'] / trte['AMT_CREDIT']  # payment rate tạm thời
    trte['AMT_GOODS_PRICE-d-AMT_ANNUITY'] = trte['AMT_GOODS_PRICE'] / trte['AMT_ANNUITY']  # số tháng trả hết lý thuyết dành cho vay tiêu dùng
    trte['AMT_GOODS_PRICE-d-AMT_CREDIT'] = trte['AMT_GOODS_PRICE'] / trte['AMT_CREDIT']
    trte['AMT_GOODS_PRICE-s-AMT_CREDIT'] = trte['AMT_GOODS_PRICE'] - trte['AMT_CREDIT']  # chênh lệch giữa số tiền nhận được và số tiền của món hàng
    trte['AMT_GOODS_PRICE-s-AMT_CREDIT-d-AMT_INCOME_TOTAL'] = trte['AMT_GOODS_PRICE-s-AMT_CREDIT'] / trte['AMT_INCOME_TOTAL']

    trte["age"] = trte['DAYS_BIRTH'] / -365
    trte['age_finish_payment'] = (trte['DAYS_BIRTH'].abs() + (trte['AMT_CREDIT-d-AMT_ANNUITY'] * 30)) / 365
    trte['DAYS_EMPLOYED-s-DAYS_BIRTH'] = trte['DAYS_EMPLOYED'] - trte['DAYS_BIRTH']
    trte['DAYS_REGISTRATION-s-DAYS_BIRTH'] = trte['DAYS_REGISTRATION'] - trte['DAYS_BIRTH']
    trte['DAYS_ID_PUBLISH-s-DAYS_BIRTH'] = trte['DAYS_ID_PUBLISH'] - trte['DAYS_BIRTH']
    trte['DAYS_LAST_PHONE_CHANGE-s-DAYS_BIRTH'] = trte['DAYS_LAST_PHONE_CHANGE'] - trte['DAYS_BIRTH']
    trte['DAYS_REGISTRATION-s-DAYS_EMPLOYED'] = trte['DAYS_REGISTRATION'] - trte['DAYS_EMPLOYED']
    trte['DAYS_ID_PUBLISH-s-DAYS_EMPLOYED'] = trte['DAYS_ID_PUBLISH'] - trte['DAYS_EMPLOYED']
    trte['DAYS_LAST_PHONE_CHANGE-s-DAYS_EMPLOYED'] = trte['DAYS_LAST_PHONE_CHANGE'] - trte['DAYS_EMPLOYED']
    trte['DAYS_ID_PUBLISH-s-DAYS_REGISTRATION'] = trte['DAYS_ID_PUBLISH'] - trte['DAYS_REGISTRATION']
    trte['DAYS_LAST_PHONE_CHANGE-s-DAYS_REGISTRATION'] = trte['DAYS_LAST_PHONE_CHANGE'] - trte['DAYS_REGISTRATION']
    trte['DAYS_LAST_PHONE_CHANGE-s-DAYS_ID_PUBLISH'] = trte['DAYS_LAST_PHONE_CHANGE'] - trte['DAYS_ID_PUBLISH']

    cols = [
        'DAYS_EMPLOYED-s-DAYS_BIRTH', 'DAYS_REGISTRATION-s-DAYS_BIRTH', 'DAYS_ID_PUBLISH-s-DAYS_BIRTH', 'DAYS_LAST_PHONE_CHANGE-s-DAYS_BIRTH', 'DAYS_REGISTRATION-s-DAYS_EMPLOYED',
        'DAYS_ID_PUBLISH-s-DAYS_EMPLOYED', 'DAYS_LAST_PHONE_CHANGE-s-DAYS_EMPLOYED', 'DAYS_ID_PUBLISH-s-DAYS_REGISTRATION', 'DAYS_LAST_PHONE_CHANGE-s-DAYS_REGISTRATION',
        'DAYS_LAST_PHONE_CHANGE-s-DAYS_ID_PUBLISH'
    ]
    cols_comb = list(combinations(cols, 2))  # tổ hợp chập 2
    for i, j in cols_comb:
        trte[f'{i}-d-{j}'] = trte[i] / trte[j]

    trte['DAYS_EMPLOYED-d-DAYS_BIRTH'] = trte['DAYS_EMPLOYED'] / trte['DAYS_BIRTH']
    trte['DAYS_REGISTRATION-d-DAYS_BIRTH'] = trte['DAYS_REGISTRATION'] / trte['DAYS_BIRTH']
    trte['DAYS_ID_PUBLISH-d-DAYS_BIRTH'] = trte['DAYS_ID_PUBLISH'] / trte['DAYS_BIRTH']
    trte['DAYS_LAST_PHONE_CHANGE-d-DAYS_BIRTH'] = trte['DAYS_LAST_PHONE_CHANGE'] / trte['DAYS_BIRTH']
    trte['DAYS_REGISTRATION-d-DAYS_EMPLOYED'] = trte['DAYS_REGISTRATION'] / trte['DAYS_EMPLOYED']
    trte['DAYS_ID_PUBLISH-d-DAYS_EMPLOYED'] = trte['DAYS_ID_PUBLISH'] / trte['DAYS_EMPLOYED']
    trte['DAYS_LAST_PHONE_CHANGE-d-DAYS_EMPLOYED'] = trte['DAYS_LAST_PHONE_CHANGE'] / trte['DAYS_EMPLOYED']
    trte['DAYS_ID_PUBLISH-d-DAYS_REGISTRATION'] = trte['DAYS_ID_PUBLISH'] / trte['DAYS_REGISTRATION']
    trte['DAYS_LAST_PHONE_CHANGE-d-DAYS_REGISTRATION'] = trte['DAYS_LAST_PHONE_CHANGE'] / trte['DAYS_REGISTRATION']
    trte['DAYS_LAST_PHONE_CHANGE-d-DAYS_ID_PUBLISH'] = trte['DAYS_LAST_PHONE_CHANGE'] / trte['DAYS_ID_PUBLISH']
    trte['OWN_CAR_AGE-d-DAYS_BIRTH'] = (trte['OWN_CAR_AGE'] * (-365)) / trte['DAYS_BIRTH']
    trte['OWN_CAR_AGE-s-DAYS_BIRTH'] = trte['DAYS_BIRTH'] + (trte['OWN_CAR_AGE'] * 365)
    trte['OWN_CAR_AGE-d-DAYS_EMPLOYED'] = trte['OWN_CAR_AGE'] / trte['DAYS_EMPLOYED']
    trte['OWN_CAR_AGE-s-DAYS_EMPLOYED'] = trte['DAYS_EMPLOYED'] + (trte['OWN_CAR_AGE'] * 365)
    trte['cnt_adults'] = trte['CNT_FAM_MEMBERS'] - trte['CNT_CHILDREN']
    trte['CNT_CHILDREN-d-CNT_FAM_MEMBERS'] = trte['CNT_CHILDREN'] / trte['CNT_FAM_MEMBERS']
    trte['AMT_INCOME_TOTAL-d-CNT_CHILDREN'] = trte['AMT_INCOME_TOTAL'] / trte['CNT_CHILDREN']
    trte['AMT_CREDIT-d-CNT_CHILDREN'] = trte['AMT_CREDIT'] / trte['CNT_CHILDREN']
    trte['AMT_ANNUITY-d-CNT_CHILDREN'] = trte['AMT_ANNUITY'] / trte['CNT_CHILDREN']
    trte['AMT_GOODS_PRICE-d-CNT_CHILDREN'] = trte['AMT_GOODS_PRICE'] / trte['CNT_CHILDREN']
    trte['AMT_INCOME_TOTAL-d-cnt_adults'] = trte['AMT_INCOME_TOTAL'] / trte['cnt_adults']
    trte['AMT_CREDIT-d-cnt_adults'] = trte['AMT_CREDIT'] / trte['cnt_adults']
    trte['AMT_ANNUITY-d-cnt_adults'] = trte['AMT_ANNUITY'] / trte['cnt_adults']
    trte['AMT_GOODS_PRICE-d-cnt_adults'] = trte['AMT_GOODS_PRICE'] / trte['cnt_adults']
    trte['AMT_INCOME_TOTAL-d-CNT_FAM_MEMBERS'] = trte['AMT_INCOME_TOTAL'] / trte['CNT_FAM_MEMBERS']
    trte['AMT_CREDIT-d-CNT_FAM_MEMBERS'] = trte['AMT_CREDIT'] / trte['CNT_FAM_MEMBERS']
    trte['AMT_ANNUITY-d-CNT_FAM_MEMBERS'] = trte['AMT_ANNUITY'] / trte['CNT_FAM_MEMBERS']
    trte['AMT_GOODS_PRICE-d-CNT_FAM_MEMBERS'] = trte['AMT_GOODS_PRICE'] / trte['CNT_FAM_MEMBERS']

    trte['EXT_SOURCES_prod'] = trte['EXT_SOURCE_1'] * trte['EXT_SOURCE_2'] * trte['EXT_SOURCE_3']
    trte['EXT_SOURCES_sum'] = trte[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)
    trte['EXT_SOURCES_mean'] = trte[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    trte['EXT_SOURCES_std'] = trte[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    trte['EXT_SOURCES_sum'] = trte['EXT_SOURCES_sum'].fillna(trte['EXT_SOURCES_sum'].mean())
    trte['EXT_SOURCES_mean'] = trte['EXT_SOURCES_mean'].fillna(trte['EXT_SOURCES_mean'].mean())
    trte['EXT_SOURCES_std'] = trte['EXT_SOURCES_std'].fillna(trte['EXT_SOURCES_std'].mean())  # fill bằng mean. có nên fill bằng kNN hay lightgbm không?
    trte['EXT_SOURCES_1-2-3'] = trte['EXT_SOURCE_1'] - trte['EXT_SOURCE_2'] - trte['EXT_SOURCE_3']
    trte['EXT_SOURCES_2-1-3'] = trte['EXT_SOURCE_2'] - trte['EXT_SOURCE_1'] - trte['EXT_SOURCE_3']
    trte['EXT_SOURCES_1-2'] = trte['EXT_SOURCE_1'] - trte['EXT_SOURCE_2']
    trte['EXT_SOURCES_2-3'] = trte['EXT_SOURCE_2'] - trte['EXT_SOURCE_3']
    trte['EXT_SOURCES_1-3'] = trte['EXT_SOURCE_1'] - trte['EXT_SOURCE_3']

    trte['maxwell_feature'] = (trte['EXT_SOURCE_1'] * trte['EXT_SOURCE_3'])**(1 / 2)  # tham khảo từ meta feature

    trte['building_score_avg_mean'] = trte[avg_cols].mean(1)
    trte['building_score_avg_std'] = trte[avg_cols].std(1)
    trte['building_score_avg_sum'] = trte[avg_cols].sum(1)
    trte['building_score_mode_mean'] = trte[mode_cols].mean(1)
    trte['building_score_mode_std'] = trte[mode_cols].std(1)
    trte['building_score_mode_sum'] = trte[mode_cols].sum(1)
    trte['building_score_medi_mean'] = trte[medi_cols].mean(1)
    trte['building_score_medi_std'] = trte[medi_cols].std(1)
    trte['building_score_medi_sum'] = trte[medi_cols].sum(1)

    trte["AMT_REQ_CREDIT_BUREAU_sum"] = trte[req_bureau_cols].sum(1)  # tổng số request

    trte["DEF_30_CNT_SOCIAL_CIRCLE-d-OBS_30_CNT_SOCIAL_CIRCLE"] = trte["DEF_30_CNT_SOCIAL_CIRCLE"] / trte["OBS_30_CNT_SOCIAL_CIRCLE"]
    trte["DEF_60_CNT_SOCIAL_CIRCLE-d-OBS_60_CNT_SOCIAL_CIRCLE"] = trte["DEF_60_CNT_SOCIAL_CIRCLE"] / trte["OBS_60_CNT_SOCIAL_CIRCLE"]
    
    trte.replace(np.inf, np.nan, inplace=True)
    trte.replace(-np.inf, np.nan, inplace=True)
    
    utils.to_pickles(trte[trte["data"] == 1].drop(["data"], axis=1), "train", size=100000)
    utils.to_pickles(trte[trte["data"] == 0].drop(["data"], axis=1), "test", size=100000)

    cache_clear(globals())
