import pandas as pd
import numpy as np

from config import ROOT, use_cols, prev_use_cols  # lib này được khởi tạo ban đầu dự án
import modules.utils as utils
from helpers.cache_clear import cache_clear

get_pickle = utils.get_pickle
_keep_vars = set(globals().keys())  # lưu biến gốc

def installments_payments_extract(test_run=False):
    if test_run:
        print("extract installments payments")
        for path in utils.get_pickle_paths(name="installments"):
            print(path)
        return
    
    installments = get_pickle("installments")
    
    prev = get_pickle("prev")[prev_use_cols]
    installments.sort_values(["SK_ID_PREV", "DAYS_ENTRY_PAYMENT"], ascending=[True, True], inplace=True)
    installments.reset_index(drop=True, inplace=True)
    installments["index"] = installments.index
    prev["exist"] = 1
    merged_df = pd.merge(
        installments[installments["SK_ID_PREV"].isin(installments[(installments["AMT_INSTALMENT"] == 0) | (installments["AMT_INSTALMENT"].isnull())]["SK_ID_PREV"].unique())],
        prev,
        on="SK_ID_PREV",
        how="left"
    )
    median_instalment = merged_df.groupby("SK_ID_PREV")["AMT_INSTALMENT"].transform("median")
    payment_sum = merged_df.groupby(["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"])["AMT_PAYMENT"].sum()  # vì có nhiều lần trả cho một INSTALMENT nên lấy sum
    median_payment = merged_df["SK_ID_PREV"].map(payment_sum.groupby("SK_ID_PREV").median())
    median_annuity = pd.Series(
        np.where(
            (median_instalment == 0) | median_instalment.isna(),  # annuity = null => lấy trung vị instalment => instalment = null lấy trung vị payment
            median_payment,
            median_instalment
        ),
        index=merged_df.index
    )
    merged_df["AMT_ANNUITY"] = merged_df["AMT_ANNUITY"].fillna(median_annuity)
    mask = (merged_df["NUM_INSTALMENT_VERSION"] != 0) & (merged_df["AMT_PAYMENT"] > 0)  # chỉ lấy những bản ghi có PAYMENT
    merged_df.loc[mask, "AMT_INSTALMENT"] = merged_df.loc[mask, "AMT_INSTALMENT"].replace(0, np.nan)
    merged_df["AMT_INSTALMENT"] = merged_df["AMT_INSTALMENT"].fillna(merged_df["AMT_ANNUITY"])
    merged_df = merged_df.rename(columns={"AMT_INSTALMENT": "filled_AMT_INSTALMENT"})  # đổi tên cột và index thành cột
    installments = installments.merge(merged_df[merged_df["NUM_INSTALMENT_VERSION"] != 0][["index", "SK_ID_PREV", "filled_AMT_INSTALMENT"]], on=["index", "SK_ID_PREV"], how="left")
    installments.loc[~installments["filled_AMT_INSTALMENT"].isnull(), "AMT_INSTALMENT"] = installments["filled_AMT_INSTALMENT"]
    installments = installments.drop(["filled_AMT_INSTALMENT"], axis=1)
    installments["days_delayed_payment"] = installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    installments["AMT_PAYMENT-s-AMT_INSTALMENT"] = installments["AMT_PAYMENT"] - installments["AMT_INSTALMENT"]  # chênh lệch giữa số tiền trả thực tế và số tiền phải trả theo quy định
    installments["AMT_PAYMENT-d-AMT_INSTALMENT"] = installments["AMT_PAYMENT"] / installments["AMT_INSTALMENT"]  # tỉ lệ giữa số tiền trả thực tế và số tiền phải trả theo quy định
    installments["days_weighted_delayed_payment"] = installments["days_delayed_payment"] * installments["AMT_PAYMENT-d-AMT_INSTALMENT"]  # delay trả góp * tỉ lệ trả đủ (trọng số)
    installments["days_weighted_delay_tsw3"] = installments['days_weighted_delayed_payment'] * (1 + (installments['DAYS_ENTRY_PAYMENT'] * 0.0003))  # time series weight decay = 0.0003
    installments['DPD'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    installments['DBD'] = installments['DAYS_INSTALMENT'] - installments['DAYS_ENTRY_PAYMENT']
    installments['DPD'] = installments['DPD'].apply(lambda x: x if x > 0 else 0)
    installments['DBD'] = installments['DBD'].apply(lambda x: x if x > 0 else 0)
    installments['month'] = (installments['DAYS_ENTRY_PAYMENT'] / 30).map(np.floor)
    prev['CNT_PAYMENT'].replace(0, np.nan, inplace=True)
    installments = installments.merge(prev[["SK_ID_PREV", "CNT_PAYMENT", "AMT_ANNUITY"]], on="SK_ID_PREV", how='left')
    installments["NUM_INSTALMENT_ratio"] = installments["NUM_INSTALMENT_NUMBER"] / installments["CNT_PAYMENT"]
    installments['AMT_PAYMENT-d-AMT_ANNUITY'] = installments['AMT_PAYMENT'] / installments['AMT_ANNUITY']
    _keep_vars.update(["installments"])
    cache_clear(globals(), _keep_vars)

    trte = utils.get_trte()
    drop_cols = list(set(list(trte.columns) + ["CNT_PAYMENT", "AMT_ANNUITY", "index"]))
    for col in ["SK_ID_PREV", "SK_ID_CURR"]:
        if col in drop_cols:
            drop_cols.remove(col)
    installments = installments.merge(trte, on="SK_ID_CURR", how='left')
    installments['DAYS_ENTRY_PAYMENT-s-app_DAYS_BIRTH'] = installments['DAYS_ENTRY_PAYMENT'] - installments['app_DAYS_BIRTH']
    installments['DAYS_ENTRY_PAYMENT-s-app_DAYS_EMPLOYED'] = installments['DAYS_ENTRY_PAYMENT'] - installments['app_DAYS_EMPLOYED']
    installments['DAYS_ENTRY_PAYMENT-s-app_DAYS_REGISTRATION'] = installments['DAYS_ENTRY_PAYMENT'] - installments['app_DAYS_REGISTRATION']
    installments['DAYS_ENTRY_PAYMENT-s-app_DAYS_ID_PUBLISH'] = installments['DAYS_ENTRY_PAYMENT'] - installments['app_DAYS_ID_PUBLISH']
    installments['DAYS_ENTRY_PAYMENT-s-app_DAYS_LAST_PHONE_CHANGE'] = installments['DAYS_ENTRY_PAYMENT'] - installments['app_DAYS_LAST_PHONE_CHANGE']
    installments['AMT_PAYMENT-d-app_AMT_INCOME_TOTAL'] = installments['AMT_PAYMENT'] / installments['app_AMT_INCOME_TOTAL']
    installments['AMT_PAYMENT-d-app_AMT_CREDIT'] = installments['AMT_PAYMENT'] / installments['app_AMT_CREDIT']
    installments['AMT_PAYMENT-d-app_AMT_ANNUITY'] = installments['AMT_PAYMENT'] / installments['app_AMT_ANNUITY']
    installments['AMT_PAYMENT-d-app_AMT_GOODS_PRICE'] = installments['AMT_PAYMENT'] / installments['app_AMT_GOODS_PRICE']
    installments.replace(np.inf, np.nan, inplace=True)
    installments.replace(-np.inf, np.nan, inplace=True)
    installments.drop(columns=drop_cols, inplace=True)
    utils.to_pickles(installments, "installments")
    
    print("extract installments payments")
    cache_clear(globals())
