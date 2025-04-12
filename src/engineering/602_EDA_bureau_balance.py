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
import modules.multi as multi
importlib.reload(view)
importlib.reload(EDA)
importlib.reload(utils)
importlib.reload(config)
importlib.reload(multi)
use_cols = config.use_cols
prev_use_cols = config.prev_use_cols
app_day_cols = config.app_day_cols
def cache_clear():
    for var in list(globals()):  
        if var not in _keep_vars and not var.startswith("_"):  
            del globals()[var]  
    gc.collect()
_keep_vars = set(globals().keys())  # lưu biến gốc
bureau_balance = pd.read_pickle(ROOT + "/data/pkl/bureau_balance.p")
bureau_balance.sort_values(['SK_ID_BUREAU', 'MONTHS_BALANCE'], inplace=True)
bureau_balance = pd.get_dummies(bureau_balance, columns=['STATUS']) # one hot encode
bureau_balance.reset_index(drop=True, inplace=True)
bureau_balance
bureau_balance.to_pickle(ROOT + "/data/processed/f601_bureau_balance.p")
cache_clear()
