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
from sitecustomize import ROOT # lib này được khởi tạo ban đầu dự án
import helpers.view as view
import helpers.EDA as EDA
import helpers.config as config
import modules.utils as utils
importlib.reload(view)
importlib.reload(EDA)
importlib.reload(utils)
importlib.reload(config)
use_cols = config.use_cols
prev_use_cols = config.prev_use_cols
def cache_clear():
    for var in list(globals()):  
        if var not in _keep_vars and not var.startswith("_"):  
            del globals()[var]  
    gc.collect()
_keep_vars = set(globals().keys())  # lưu biến gốc
installments = pd.read_pickle(ROOT + "/data/pkl/installments_payments.p")
prev = pd.read_pickle(ROOT + "/data/pkl/previous_application.p")[prev_use_cols]
train = pd.read_pickle(ROOT + "/data/pkl/application_train.p")[use_cols]
test = pd.read_pickle(ROOT + "/data/pkl/application_test.p")[use_cols]
installments.sort_values(["SK_ID_PREV", "DAYS_ENTRY_PAYMENT"], ascending=[True, True], inplace=True)
installments.reset_index(drop=True, inplace=True)
installments["index"] = installments.index
prev["exist"] = 1
merged_df = pd.merge(
    installments[installments["SK_ID_PREV"].isin(installments[(installments["AMT_INSTALMENT"]==0) | (installments["AMT_INSTALMENT"].isnull())]["SK_ID_PREV"].unique())], 
    prev,
    on="SK_ID_PREV", 
    how="left"
)
median_instalment = merged_df.groupby("SK_ID_PREV")["AMT_INSTALMENT"].transform("median")
payment_sum = merged_df.groupby(["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"])["AMT_PAYMENT"].sum() # vì có nhiều lần trả cho một INSTALMENT nên lấy sum
median_payment = merged_df["SK_ID_PREV"].map(payment_sum.groupby("SK_ID_PREV").median()) 
median_annuity = pd.Series(
    np.where(
        (median_instalment == 0) | median_instalment.isna(),   # annuity = null => lấy trung vị instalment => instalment = null lấy trung vị payment
        median_payment,  
        median_instalment  
    ),
    index=merged_df.index
)
merged_df["AMT_ANNUITY"] = merged_df["AMT_ANNUITY"].fillna(median_annuity)
mask = (merged_df["NUM_INSTALMENT_VERSION"] != 0) & (merged_df["AMT_PAYMENT"] > 0) # chỉ lấy những bản ghi có PAYMENT
merged_df.loc[mask, "AMT_INSTALMENT"] = merged_df.loc[mask, "AMT_INSTALMENT"].replace(0, np.nan)
merged_df["AMT_INSTALMENT"] = merged_df["AMT_INSTALMENT"].fillna(merged_df["AMT_ANNUITY"])
merged_df = merged_df.rename(columns={"AMT_INSTALMENT": "filled_AMT_INSTALMENT"}) # đổi tên cột và index thành cột
merged_df[merged_df["NUM_INSTALMENT_VERSION"]!=0][["index", "SK_ID_PREV", "filled_AMT_INSTALMENT"]]
installments = installments.merge(
    merged_df[merged_df["NUM_INSTALMENT_VERSION"]!=0][["index", "SK_ID_PREV", "filled_AMT_INSTALMENT"]],
    on=["index", "SK_ID_PREV"],  
    how="left"
)
installments
installments.loc[~installments["filled_AMT_INSTALMENT"].isnull(), "AMT_INSTALMENT"] = installments["filled_AMT_INSTALMENT"]
installments = installments.drop(["filled_AMT_INSTALMENT"], axis=1)
installments["days_delayed_payment"] = installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
installments["AMT_PAYMENT-s-AMT_INSTALMENT"] = installments["AMT_PAYMENT"] - installments["AMT_INSTALMENT"] # chênh lệch giữa số tiền trả thực tế và số tiền phải trả theo quy định
installments["AMT_PAYMENT-d-AMT_INSTALMENT"] = installments["AMT_PAYMENT"] / installments["AMT_INSTALMENT"] # tỉ lệ giữa số tiền trả thực tế và số tiền phải trả theo quy định
installments["days_weighted_delayed_payment"] = installments["days_delayed_payment"] * installments["AMT_PAYMENT-d-AMT_INSTALMENT"] # delay trả góp * tỉ lệ trả đủ (trọng số)
installments["days_weighted_delay_tsw3"] = installments['days_weighted_delayed_payment'] * (1 + (installments['DAYS_ENTRY_PAYMENT'] * 0.0003)) # time series weight decay = 0.0003
installments['DPD'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
installments['DBD'] = installments['DAYS_INSTALMENT'] - installments['DAYS_ENTRY_PAYMENT']
installments['DPD'] = installments['DPD'].apply(lambda x: x if x > 0 else 0)
installments['DBD'] = installments['DBD'].apply(lambda x: x if x > 0 else 0)
installments['month'] = (installments['DAYS_ENTRY_PAYMENT']/30).map(np.floor)
prev['CNT_PAYMENT'].replace(0, np.nan, inplace=True)
installments = installments.merge(prev[["SK_ID_PREV", "CNT_PAYMENT", "AMT_ANNUITY"]], on="SK_ID_PREV", how='left')
installments["NUM_INSTALMENT_ratio"] = installments["NUM_INSTALMENT_NUMBER"] / installments["CNT_PAYMENT"] 
installments['AMT_PAYMENT-d-AMT_ANNUITY'] = installments['AMT_PAYMENT'] / installments['AMT_ANNUITY']
trte = utils.get_trte(train, test)
drop_cols = list(set(list(trte.columns) + ["CNT_PAYMENT", "AMT_ANNUITY", "index"]))
for col in ["SK_ID_PREV", "SK_ID_CURR"]:
    if col in drop_cols:
        drop_cols.remove(col)
installments = installments.merge(trte, on="SK_ID_CURR", how='left')
installments['DAYS_ENTRY_PAYMENT-s-app_DAYS_BIRTH']             = installments['DAYS_ENTRY_PAYMENT'] - installments['app_DAYS_BIRTH']
installments['DAYS_ENTRY_PAYMENT-s-app_DAYS_EMPLOYED']          = installments['DAYS_ENTRY_PAYMENT'] - installments['app_DAYS_EMPLOYED']
installments['DAYS_ENTRY_PAYMENT-s-app_DAYS_REGISTRATION']      = installments['DAYS_ENTRY_PAYMENT'] - installments['app_DAYS_REGISTRATION']
installments['DAYS_ENTRY_PAYMENT-s-app_DAYS_ID_PUBLISH']        = installments['DAYS_ENTRY_PAYMENT'] - installments['app_DAYS_ID_PUBLISH']
installments['DAYS_ENTRY_PAYMENT-s-app_DAYS_LAST_PHONE_CHANGE'] = installments['DAYS_ENTRY_PAYMENT'] - installments['app_DAYS_LAST_PHONE_CHANGE']
installments['AMT_PAYMENT-d-app_AMT_INCOME_TOTAL'] = installments['AMT_PAYMENT'] / installments['app_AMT_INCOME_TOTAL']
installments['AMT_PAYMENT-d-app_AMT_CREDIT']      = installments['AMT_PAYMENT'] / installments['app_AMT_CREDIT']
installments['AMT_PAYMENT-d-app_AMT_ANNUITY']     = installments['AMT_PAYMENT'] / installments['app_AMT_ANNUITY']
installments['AMT_PAYMENT-d-app_AMT_GOODS_PRICE'] = installments['AMT_PAYMENT'] / installments['app_AMT_GOODS_PRICE']
installments.replace(np.inf, np.nan, inplace=True)
installments.replace(-np.inf, np.nan, inplace=True)
installments.drop(columns=drop_cols, inplace=True)
installments
installments.to_pickle(ROOT + "/data/processed/f201_installments_payments.p")
installments[installments['days_delayed_payment']>0].to_pickle(ROOT + "/data/processed/f201_installments_payments_delay.p")
installments[installments['days_delayed_payment']<=0].to_pickle(ROOT + "/data/processed/f201_installments_payments_notdelay.p")
cache_clear()
