from config import ROOT  # lib này được khởi tạo ban đầu dự án
import pandas as pd

import modules.utils as utils
import modules.encode as encode
from config import app_categorical_features

def label_encode(train, test):
    train["data"]=1
    test["data"]=0
    
    trte = pd.concat([train, test])[["SK_ID_CURR", "data"] + app_categorical_features]
    
    trte = encode.label_encode(trte, app_categorical_features)
    
    train = trte[trte["data"]==1].drop(["data"], axis=1)
    test = trte[trte["data"]==0].drop(["data"], axis=1)
    
    return train, test

def target_encode(train, test):
    train = train[["SK_ID_CURR", "TARGET"] + app_categorical_features]
    test = test[["SK_ID_CURR"] + app_categorical_features]
    
    train, test = encode.target_encode(train, test, app_categorical_features)
    
    train = train.drop(["TARGET"], axis=1)

    return train, test