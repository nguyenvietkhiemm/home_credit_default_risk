# đọc/ghi file/feature

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from config import reload
from config import use_cols, rename_di, paths, processed_paths, feature_paths

# reload()

CHUNK_SIZE=100000

def to_pickles(df, name=None, chunk_size=CHUNK_SIZE):
    if name is None:
        print("name=None")
        return

    path = processed_paths[name]
    os.makedirs(path, exist_ok=True)

    n = len(df)
    for i in range(0, n, chunk_size):
        df.iloc[i:i+chunk_size].to_pickle(f"{path}/p_{i//chunk_size}.p")


def get_pickles(name=None, cols=None):
    if name is None:
        print("name=None")
        return
    path = processed_paths[name]
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


def get_trte(train=None, test=None):
    if train is None or test is None:
        train = get_pickle("train")
        test = get_pickle("test")
    trte = pd.concat([train, test])[use_cols]
    trte.rename(columns=rename_di, inplace=True)

    return trte


def to_feature(df, name=None):
    if not name:
        print("name=None")
    path = feature_paths[name]
    if df.columns.duplicated().sum() > 0:
        raise Exception(
            f'duplicated!: { df.columns[df.columns.duplicated()] }')
    df.reset_index(inplace=True, drop=True)
    for c in df.columns:
        df[[c]].to_feather(f'{path}/{c}.f')
    return
