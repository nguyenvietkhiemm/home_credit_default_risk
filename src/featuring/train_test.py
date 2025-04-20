from config import ROOT  # lib này được khởi tạo ban đầu dự án
import pandas as pd

import modules.utils as utils
import modules.encode as encode
from config import app_categorical_features

from encode.train_test import label_encode, target_encode

def partitioning(train, test):
    PREF = "f001_"
    
    train["data"]=1
    test["data"]=0
    utils.to_feature(train.drop(app_categorical_features + ["data"], axis=1).add_prefix(PREF), name="train")
    utils.to_feature(test.drop(app_categorical_features + ["data"], axis=1).add_prefix(PREF), name="test") 
    
    print("partitioning train test")
    
def encode(train, test):
    PREF = "f002_"
    _train, _test = label_encode(train, test)
    utils.to_feature(_train.add_prefix(PREF), name="train")
    utils.to_feature(_test.add_prefix(PREF), name="test") 
    
    PREF = "f003_"
    _train, _test = target_encode(train, test)
    utils.to_feature(_train.add_prefix(PREF), name="train")
    utils.to_feature(_test.add_prefix(PREF), name="test") 