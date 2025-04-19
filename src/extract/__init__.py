
from config import ROOT  # lib này được khởi tạo ban đầu dự án

import modules.utils as utils

get_pickle = utils.get_pickle
get_pickles = utils.get_pickles

from train_test import train_test_extract
from prev_application import prev_extract
from installments_payments import installments_payments_extract
from credit_card_balance import credit_balance_extract
from bureau import bureau_extract
from bureau_balance import bureau_balance_extract

test_run=True

def extract():
    
    train_test_extract(test_run)
    
    prev_extract(test_run)
    
    installments_payments_extract(test_run)
    
    credit_balance_extract(test_run)
    
    bureau_extract(test_run)
    
    bureau_balance_extract(test_run)