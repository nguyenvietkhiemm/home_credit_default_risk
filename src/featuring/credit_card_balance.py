from config import ROOT  # lib này được khởi tạo ban đầu dự án
import pandas as pd

import modules.utils as utils
from config import app_categorical_features

from extract.credit_card_balance import credit_balance_extract

# meta feature
    
def credit_balance_featuring():
    print("=====extract train test=====")
    credit_balance_extract()
    print("=====[DONE extract train test]=====")
    
    # print("=====predict interest rate train test=====")
    # pred_interest_rate(train_model=False)
    # print("=====[DONE predict interest rate train test]=====")
    
    # print("=====partitioning train test=====")
    # partitioning(train, test)
    # print("=====[DONE partitioning train test]=====")
    
    # # imputation EXT_SOURCE sau
    
    # print("=====encode train test=====")
    # encode(train, test)
    # print("=====[DONE encode train test]=====")