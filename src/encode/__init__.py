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

from config import ROOT  # lib này được khởi tạo ban đầu dự án

import helpers.view as view
import helpers.EDA as EDA
import modules.utils as utils
import modules.encode as encode
import modules.multi as multi
from config import use_cols, app_categorical_features

importlib.reload(view)
importlib.reload(EDA)
importlib.reload(utils)
importlib.reload(multi)

get_pickle = utils.get_pickle
get_pickles = utils.get_pickles

_keep_vars = set(globals().keys())  # lưu biến gốc


from train_test import label_encode

train = get_pickles("train") # chỉ gọi đúng 1 lần train test trong này
test = get_pickles("test")
label_encode(train, test)