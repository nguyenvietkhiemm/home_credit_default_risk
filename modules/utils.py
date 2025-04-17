import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from config import reload
from config import use_cols, rename_di, paths, pickles_paths

# reload()


def to_pickles(df, name=None, chunk_size=100000):
    if name is None:
        print("name=None")
        return

    path = pickles_paths[name]
    os.makedirs(path, exist_ok=True)

    n = len(df)
    for i in range(0, n, chunk_size):
        df.iloc[i:i+chunk_size].to_pickle(f"{path}/p_{i//chunk_size}.p")


def get_pickles(name=None, cols=None):
    if name is None:
        print("name=None")
        return
    path = pickles_paths[name]
    if not os.path.exists(path):
        print(f"folder {path} not exist")
        return None

    if not cols:
        files = sorted([f for f in os.listdir(path) if f.endswith(".p")])
        dfs = [pd.read_pickle(os.path.join(path, f)) for f in files]
    else:
        files = sorted([f for f in os.listdir(path) if f.endswith(".p")])
        dfs = [pd.read_pickle(os.path.join(path, f))[cols] for f in files]
    return pd.concat(dfs, ignore_index=True)


def get_pickle(name=None):
    if not name:
        print("name=None")
    return pd.read_pickle(paths[name])


def get_trte(train, test):
    trte = pd.concat([train, test])[use_cols]
    trte.rename(columns=rename_di, inplace=True)

    return trte
