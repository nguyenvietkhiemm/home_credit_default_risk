import ctypes
import numpy as np
from config import ROOT

lib = ctypes.CDLL(ROOT + "/modules/cpp/interest_rate.dll")
lib.interest_rate.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_double,  # tol
    ctypes.c_int      # max_iter
]
lib.interest_rate.restype = None

def interest_rate(amt_annuity, amt_credit, cnt_payment, rates, N, tol=1e-7, max_iter=500): # truyền vào là các vector, các mảng. c++ sẽ xử lý và gán vào địa chỉ ô nhớ 
    return lib.interest_rate(amt_annuity, amt_credit, cnt_payment, rates, N, tol, max_iter)


# only python: 136s với 1 cnt
# python + njit: 76,67s với 1 cnt
# c++: 67s với 1 cnt
# c++ tối ưu vector và ngắt với ngưỡng delta hội tụ: 1.3s ?? và ~4s cho possible_cnt