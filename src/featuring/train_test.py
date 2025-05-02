from config import ROOT  # lib này được khởi tạo ban đầu dự án
import pandas as pd

import modules.utils as utils
from config import app_categorical_features

from extract.train_test import train_test_extract
from encode.train_test import label_encode, target_encode
from imputation.interest_rate import pred_interest_rate

# meta feature

def partitioning(train, test):
    PREF = "f001_"
    
    utils.to_feature(train.drop(app_categorical_features + ["SK_ID_CURR", "TARGET"], axis=1).add_prefix(PREF), name="train")
    utils.to_feature(test.drop(app_categorical_features + ["SK_ID_CURR", "TARGET"], axis=1).add_prefix(PREF), name="test") 
    
def encode(train, test):
    PREF = "f002_"
    _train, _test = label_encode(train, test)
    utils.to_feature(_train.drop(["SK_ID_CURR"], axis=1).add_prefix(PREF), name="train")
    utils.to_feature(_test.drop(["SK_ID_CURR"], axis=1).add_prefix(PREF), name="test") 
    
    PREF = "f003_"
    _train, _test = target_encode(train, test)
    utils.to_feature(_train.drop(["SK_ID_CURR"], axis=1).add_prefix(PREF), name="train")
    utils.to_feature(_test.drop(["SK_ID_CURR"], axis=1).add_prefix(PREF), name="test") 
    
def train_test_featuring(train, test):
    print("=====extract train test=====")
    train_test_extract()
    print("=====[DONE extract train test]=====")
    
    # print("=====predict interest rate train test=====")
    # pred_interest_rate(train_model=False)
    # print("=====[DONE predict interest rate train test]=====")
    
    print("=====partitioning train test=====")
    partitioning(train, test)
    print("=====[DONE partitioning train test]=====")
    
    # # imputation EXT_SOURCE sau
    
    print("=====encode train test=====")
    encode(train, test)
    print("=====[DONE encode train test]=====")