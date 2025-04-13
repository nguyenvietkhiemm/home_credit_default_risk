import pandas as pd
import numpy as np
from config import reload
from config import use_cols, rename_di, paths

# reload()

def get_pickle(name=None):
    if not name:
        print("name=None")
    return pd.read_pickle(paths[name])


def get_trte(train, test):
    trte = pd.concat([train, test])[use_cols]
    trte.rename(columns=rename_di, inplace=True)

    return trte
