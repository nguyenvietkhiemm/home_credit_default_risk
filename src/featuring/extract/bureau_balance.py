import pandas as pd

from config import ROOT, use_cols, prev_use_cols  # lib này được khởi tạo ban đầu dự án
import modules.utils as utils
from helpers.cache_clear import cache_clear

get_pickle = utils.get_pickle
_keep_vars = set(globals().keys())  # lưu biến gốc


def bureau_balance_extract(test_run=False):
    if test_run:
        print("extract bureau balance")
        for path in utils.get_pickle_paths(name="bureau_balance"):
            print(path)
        return

    bureau_balance = get_pickle("bureau_balance")
    bureau = get_pickle("bureau")[["SK_ID_BUREAU", "SK_ID_CURR"]]
    
    bureau_balance = pd.merge(bureau_balance, bureau, on="SK_ID_BUREAU", how="left")

    bureau_balance = pd.get_dummies(bureau_balance, columns=['STATUS'])  # one hot encode
    
    bureau_balance.sort_values(['SK_ID_BUREAU', 'MONTHS_BALANCE'], inplace=True)
    bureau_balance.reset_index(drop=True, inplace=True)
    
    utils.to_pickles(bureau_balance, "bureau_balance")

    cache_clear(globals())
