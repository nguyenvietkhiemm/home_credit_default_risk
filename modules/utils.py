import pandas as pd
import numpy as np
from helpers.config import use_cols, rename_di

def get_trte(train, test):
    trte = pd.concat([train, test])[use_cols]
    trte.rename(columns=rename_di, inplace=True)
    
    return trte