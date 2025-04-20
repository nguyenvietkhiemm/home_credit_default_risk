from config import ROOT  # lib này được khởi tạo ban đầu dự án
import pandas as pd

import modules.utils as utils
import modules.encode as encode
from config import app_categorical_features

def label_encode(train, test):

    train["data"]=1
    test["data"]=0
    trte = pd.concat([train, test])[app_categorical_features + ["data"]]
    trte = encode.label_encode(trte, app_categorical_features)
    train = trte[trte["data"]==1]
    test = trte[trte["data"]==0]
    
    return train.drop(["data"], axis=1), test.drop(["data"], axis=1)

def target_encode(train, test):

    train["data"]=1
    test["data"]=0
    trte = pd.concat([train, test])[app_categorical_features + ["data"]]
    trte = encode.label_encode(trte, app_categorical_features)
    train = trte[trte["data"]==1]
    test = trte[trte["data"]==0]
    
    return train.drop(["data"], axis=1), test.drop(["data"], axis=1)