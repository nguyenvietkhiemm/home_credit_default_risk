from train_test import train_test_featuring
from prev_application import prev_featuring
from pos_cash_balance import pos_cash_featuring
from installments_payments import installments_payments_featuring
from credit_card_balance import credit_balance_featuring
from bureau import bureau_featuring
from bureau_balance import bureau_balance_featuring

import time
import modules.utils as utils

def featuring():
    print("=====featuring=====")
    
    # print("==train_test_featuring==")
    # t0 = time.time()
    # train = utils.get_pickles("train")
    # test = utils.get_pickles("test")
    # train_test_featuring(train, test)
    # print(f"train_test_featuring {time.time() - t0:.2f}s")
    
    # print("==prev_featuring==")
    # t0 = time.time()
    # prev_featuring()
    # print(f"prev_featuring {time.time() - t0:.2f}s")
    
    # print("==pos_cash_featuring==")
    # t0 = time.time()
    # pos_cash_featuring()
    # print(f"pos_cash_featuring {time.time() - t0:.2f}s")
    
    # print("==installments_featuring==")
    # t0 = time.time()
    # installments_payments_featuring()
    # print(f"installments_featuring {time.time() - t0:.2f}s")
    
    # print("==credit_balance_featuring==")
    # t0 = time.time()
    # credit_balance_featuring()
    # print(f"credit_balance_featuring {time.time() - t0:.2f}s")
    
    # print("==bureau_featuring==")
    # t0 = time.time()
    # bureau_featuring()
    # print(f"bureau_featuring {time.time() - t0:.2f}s")
    
    # print("==bureau_balance_featuring==")
    # t0 = time.time()
    # bureau_balance_featuring()
    # print(f"bureau_balance_featuring {time.time() - t0:.2f}s")
    
    # encode()
    
    # imputation()
    
    # aggregation()
    
    print("=====[DONE featuring]=====")
featuring() # test