# đọc/ghi file/feature

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from config import reload
from config import use_cols, rename_di, paths, processed_paths, feature_paths, imputation_paths, tmp_path

# reload()

SIZE=10000

def to_pickles(df, name=None, size=SIZE, dir="processed", key="SK_ID_CURR"):
    if name is None:
        print("name=None")
        return

    path = processed_paths[name]
    os.makedirs(path, exist_ok=True)

    df = df.sort_values(key)
    grouped = df.groupby(key).groups  # Trả về dict {key: index_list}
    keys = list(grouped.keys())

    for i in range(0, len(keys), size):
        selected_keys = keys[i:i+size]
        idx = [i for k in selected_keys for i in grouped[k]]
        df_chunk = df.loc[idx].sort_index()
        df_chunk.to_pickle(f"{path}/p_{i//size}.p")
        
def to_pickle(df, name=None, dir="processed", file_name="None"):
    if name is None:
        print("name=None")
        return
    if dir=="processed":
        paths_dir = processed_paths[name]
    elif dir=="feature":
        paths_dir = feature_paths[name]
    elif dir=="imputation":
        paths_dir = imputation_paths[name]
    os.makedirs(paths_dir, exist_ok=True)
    
    df.to_pickle(f"{paths_dir}/{file_name}.p")


def get_pickle_paths(name=None, dir="processed"):
    if name is None:
        print("name=None")
        return
    if dir=="processed":
        paths_dir = processed_paths[name]
    elif dir=="imputation":
        paths_dir = imputation_paths[name]
    elif dir=="tmp":
        paths_dir = tmp_path
    
        
    files = sorted(
        [os.path.join(paths_dir, f) for f in os.listdir(paths_dir) if f.endswith(".p")],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1])
    )
    
    return files

def get_feature_paths(name = "train", prefixes=[], is_train=True):
    paths_dir = feature_paths[name]
    if len(prefixes)>0:
        files = []
        for prefix in prefixes:
            files += sorted([os.path.join(paths_dir, f) for f in os.listdir(paths_dir) if f.endswith(".f") and f.startswith(prefix)])
    else:
            files = sorted([os.path.join(paths_dir, f) for f in os.listdir(paths_dir) if f.endswith(".f")])
    return files

def get_TARGET_path():
    return feature_paths["target"] + "/TARGET.f"

def get_pickles(name=None, cols=None, dir="processed"):
    if name is None:
        print("name=None")
        return

    files = get_pickle_paths(name, dir)

    if not cols:
        dfs = [pd.read_pickle(f) for f in files]
    else:
        dfs = [pd.read_pickle(f)[cols] for f in files]
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


def to_feature(df, name=None, append=False):
    if not name:
        print("name=None")
        
    path = feature_paths[name]
    
    if df.columns.duplicated().sum() > 0:
        raise Exception(
            f'duplicated!: { df.columns[df.columns.duplicated()] }')
        
    df.reset_index(inplace=True, drop=True)
    
    for c in df.columns:
        f_path = f'{path}/{c.strip()}.f'.strip()
        print(f_path)
        if os.path.exists(f_path) and append:
            old = pd.read_feather(f_path)
            df_c = pd.concat([old, df[[c]]], ignore_index=True)
        else:
            df_c = df[[c]]
        df_c.reset_index(drop=True).to_feather(f_path)
    return


def check_var(df, var_limit=0, sample_size=None):
    df_ = df.sample(sample_size, random_state=71) if sample_size and df.shape[0] > sample_size else df

    var = df_.var(numeric_only=True)  # tránh warning khi có cột object
    col_var0 = var[var <= var_limit].index.tolist()

    if col_var0:
        print(f"remove var<={var_limit}: {col_var0}")
    return col_var0

def check_corr(df, corr_limit=1, sample_size=None):
    if sample_size and df.shape[0] > sample_size:
        df_ = df.sample(sample_size, random_state=71)
    else:
        df_ = df
    
    corr = df_.corr(method='pearson').abs()
    a, b = np.where(corr >= corr_limit)
    
    col_corr1 = set(b[b != a])  
    
    if col_corr1:
        col_corr1 = df.columns[list(col_corr1)]
        print(f'remove corr>={corr_limit}: {col_corr1}')
    
    return col_corr1

def remove_feature(df, var_limit=0, corr_limit=1, sample_size=None, only_var=True):
    col_var0 = check_var(df,  var_limit=var_limit, sample_size=sample_size)
    df.drop(col_var0, axis=1, inplace=True)
    if only_var==False:
        col_corr1 = check_corr(df, corr_limit=corr_limit, sample_size=sample_size)
        df.drop(col_corr1, axis=1, inplace=True)
    return