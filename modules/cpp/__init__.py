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

def interest_rate(amt_annuity, amt_credit, cnt_payment, rates, N, tol=1e-7, max_iter=500):
    return lib.interest_rate(amt_annuity, amt_credit, cnt_payment, rates, N, tol, max_iter)