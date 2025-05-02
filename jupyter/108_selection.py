import pandas as pd
import numpy as np
import gc
import io
import os
from itertools import combinations
from tqdm import tqdm

from IPython.display import display

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

pd.reset_option('display.float_format')
pd.set_option('display.max_colwidth', None)

from config import ROOT, prev_num_aggregations  # lib này được khởi tạo ban đầu dự án

import helpers.view as view
import helpers.EDA as EDA
import modules.utils as utils
import modules.encode as encode

from helpers.cache_clear import cache_clear

get_pickle = utils.get_pickle
get_pickles = utils.get_pickles

import lightgbm as lgb

print(lgb.__file__)

HEAD = 160000

SEED = 71

param = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'max_depth': 10,
    'num_leaves': 63,
    'max_bin': 255,
    'min_child_weight': 10,
    'min_data_in_leaf': 150,
    'reg_lambda': 0.5,  # L2 regularization term on weights.
    'reg_alpha': 0.5,  # L1 regularization term on weights.
    'colsample_bytree': 0.7,
    'subsample': 0.5,
    'nthread': 16,
    'bagging_freq': 1,
    'verbose': 0,
    'seed': SEED,
    # thêm cấu hình cho GPU
    'device_type': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
}

f001 = [
    'f001_NAME_CONTRACT_TYPE',
    'f001_CODE_GENDER',
    'f001_FLAG_OWN_CAR',
    'f001_FLAG_OWN_REALTY',
    'f001_NAME_TYPE_SUITE',
    'f001_NAME_INCOME_TYPE',
    'f001_NAME_EDUCATION_TYPE',
    'f001_NAME_FAMILY_STATUS',
    'f001_NAME_HOUSING_TYPE',
    'f001_OCCUPATION_TYPE',
    'f001_WEEKDAY_APPR_PROCESS_START',
    'f001_ORGANIZATION_TYPE',
    'f001_FONDKAPREMONT_MODE',
    'f001_HOUSETYPE_MODE',
    'f001_WALLSMATERIAL_MODE',
    'f001_EMERGENCYSTATE_MODE',
]

f002 = [
    'f002_NAME_CONTRACT_TYPE',
    'f002_CODE_GENDER',
    'f002_FLAG_OWN_CAR',
    'f002_FLAG_OWN_REALTY',
    'f002_NAME_TYPE_SUITE',
    'f002_NAME_INCOME_TYPE',
    'f002_NAME_EDUCATION_TYPE',
    'f002_NAME_FAMILY_STATUS',
    'f002_NAME_HOUSING_TYPE',
    'f002_OCCUPATION_TYPE',
    'f002_WEEKDAY_APPR_PROCESS_START',
    'f002_ORGANIZATION_TYPE',
    'f002_FONDKAPREMONT_MODE',
    'f002_HOUSETYPE_MODE',
    'f002_WALLSMATERIAL_MODE',
    'f002_EMERGENCYSTATE_MODE',
]

f003 = [
    'f002_NAME_CONTRACT_TYPE',
    'f002_CODE_GENDER',
    'f002_FLAG_OWN_CAR',
    'f002_FLAG_OWN_REALTY',
    'f002_NAME_TYPE_SUITE',
    'f002_NAME_INCOME_TYPE',
    'f002_NAME_EDUCATION_TYPE',
    'f002_NAME_FAMILY_STATUS',
    'f002_NAME_HOUSING_TYPE',
    'f002_OCCUPATION_TYPE',
    'f002_WEEKDAY_APPR_PROCESS_START',
    'f002_ORGANIZATION_TYPE',
    'f002_FONDKAPREMONT_MODE',
    'f002_HOUSETYPE_MODE',
    'f002_WALLSMATERIAL_MODE',
    'f002_EMERGENCYSTATE_MODE',
]

f108 = [
    'f108_NAME_CONTRACT_TYPE',
    'f108_WEEKDAY_APPR_PROCESS_START',
    'f108_NAME_CASH_LOAN_PURPOSE',
    'f108_NAME_CONTRACT_STATUS',
    'f108_NAME_PAYMENT_TYPE',
    'f108_CODE_REJECT_REASON',
    'f108_NAME_TYPE_SUITE',
    'f108_NAME_CLIENT_TYPE',
    'f108_NAME_GOODS_CATEGORY',
    'f108_NAME_PORTFOLIO',
    'f108_NAME_PRODUCT_TYPE',
    'f108_CHANNEL_TYPE',
    'f108_NAME_SELLER_INDUSTRY',
    'f108_NAME_YIELD_GROUP',
    'f108_PRODUCT_COMBINATION',
]

f109 = [
    'f109_NAME_CONTRACT_TYPE',
    'f109_WEEKDAY_APPR_PROCESS_START',
    'f109_NAME_CASH_LOAN_PURPOSE',
    'f109_NAME_CONTRACT_STATUS',
    'f109_NAME_PAYMENT_TYPE',
    'f109_CODE_REJECT_REASON',
    'f109_NAME_TYPE_SUITE',
    'f109_NAME_CLIENT_TYPE',
    'f109_NAME_GOODS_CATEGORY',
    'f109_NAME_PORTFOLIO',
    'f109_NAME_PRODUCT_TYPE',
    'f109_CHANNEL_TYPE',
    'f109_NAME_SELLER_INDUSTRY',
    'f109_NAME_YIELD_GROUP',
    'f109_PRODUCT_COMBINATION',
]



ALL_CAT = f001 + f002 + f003 + f108 + f109

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif 
from sklearn.impute import SimpleImputer

# def handle_low_variance(df, variance_threshold=0.0001):
#     variances = df.var()
#     to_drop = variances[variances <= variance_threshold].index.tolist()
#     filtered_df = df.drop(to_drop, axis=1)
#     if to_drop:  # Kiểm tra xem to_drop có rỗng không trước khi in
#         print(f"phương sai thấp: {to_drop[0]}...")  # In phần tử đầu tiên, ...
#     else:
#         print("Không có features nào có phương sai thấp hơn ngưỡng.")
#     return filtered_df, to_drop

# feature_paths = utils.get_feature_paths(prefixes=["f0", "f101"])
# chunk_size = 500
# chunks = [feature_paths[i:i + chunk_size] for i in range(0, len(feature_paths), chunk_size)]
# with open(os.path.join(ROOT, "data/result/used.txt"), "w") as f_selected, \
#      open(os.path.join(ROOT, "data/result/unused.txt"), "w") as f_unselected:

#     all_selected_features = [] # Danh sách để theo dõi các feature đã chọn
#     all_unselected_features = []

#     for chunk in chunks:
#         X = pd.DataFrame()
#         print(f"Đang xử lý chunk: {chunk[0]}...") # Để theo dõi tiến trình
#         try:
#             chunk_df = pd.concat([pd.read_feather(os.path.join(ROOT, file_path)).head(HEAD) for file_path in chunk], axis=1)
#         except FileNotFoundError as e:
#             print(f"Lỗi: Không tìm thấy file: {e.filename}")
#             continue  # Chuyển sang chunk tiếp theo nếu có lỗi đọc file

#         X = pd.concat([X, chunk_df], axis=1)
        
#         X_filtered, dropped_variance_features = handle_low_variance(X)
#         # X_filtered, dropped_correlation_features = handle_correlation(X_filtered)

#         selected_features = X_filtered.columns
#         # unselected_features = dropped_variance_features + dropped_correlation_features # Thêm các cột đã drop do tương quan
#         unselected_features = dropped_variance_features

#         f_selected.write("\n".join(selected_features.to_list()) + "\n")
#         f_unselected.write("\n".join(unselected_features) + "\n")
        
#         all_selected_features.extend(selected_features)
#         all_unselected_features.extend(unselected_features)
        
#     print(f"'{os.path.join(ROOT, 'data/result/used.txt')}'")
#     print(f"'{os.path.join(ROOT, 'data/result/unused.txt')}'")
#     print(f"Tổng số features đã chọn: {len(set(all_selected_features))}")
#     print(f"Tổng số features không được chọn: {len(set(all_unselected_features))}")







# feature_paths = read(ROOT + "/data/result/used.txt")
# chunk_size = 500
# chunks = [feature_paths[i:i + chunk_size] for i in range(0, len(feature_paths), chunk_size)]

    # dtrain = None
    # model = None
    # CAT = list(set(X.columns) & set(ALL_CAT))
    
    

    # dtrain = lgb.Dataset(X, label=y, categorical_feature=CAT)

    # if model is None:
    #     model = lgb.train(
    #         param,
    #         dtrain,
    #         num_boost_round=5000,
    #         callbacks=[lgb.log_evaluation(100)],
    #     )
    # else:
    #     model = lgb.train(
    #         param,
    #         dtrain,
    #         num_boost_round=100,
    #         verbose_eval=5000,
    #         callbacks=[lgb.log_evaluation(100)],
    #     )
    # importance = model.feature_importance()

    # feature_names = X.columns

    # importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

    # importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # print(importance_df)
    
    
    
    
    
    
    
    
    
    
    
# mô hình cây bị ảnh hưởng bởi feature tương quan cao lẫn nhau vì nó chia node theo information gain
# tìm các feature tương quan thấp với TARGET
    
# def handle_correlation(df, threshold=0.98):
#     corr_matrix = df.corr().abs()
#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#     to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
#     # In ra các cặp tương quan cao để debug
    
#     high_corr_pairs = []
#     for col in to_drop:
#         for index in upper.index:
#             if upper.loc[index, col] > threshold:
#                 high_corr_pairs.append((index, col))
                
#     if high_corr_pairs:  # Kiểm tra xem to_drop có rỗng không trước khi in
#         print(f"tương quan cao: {high_corr_pairs[0]}...")  # In phần tử đầu tiên, ...
#     else:
#         print("Không có features nào có tương quan cao hơn ngưỡng.")
#     filtered_df = df.drop(to_drop, axis=1)
#     return filtered_df, to_drop, high_corr_pairs

def handle_low_correlation_with_target(df, target_column, threshold=0.02):
    if target_column not in df.columns:
        raise ValueError(f"Cột target '{target_column}' không tồn tại trong DataFrame.")

    correlations = df.corr()[target_column].abs().drop(target_column)
    low_corr_columns = correlations[correlations < threshold].index.tolist()

    if low_corr_pairs:
        print(f"Tương quan thấp với '{target_column}': {low_corr_pairs[0]}...")
    else:
        print(f"Không có features nào có tương quan tuyệt đối thấp hơn ngưỡng {threshold} với '{target_column}'.")

    return low_corr_columns

def read(filename):
    try:
        with open(filename, 'r') as f:
            features = [ROOT + "/data/feature/train/" + line.strip() + ".f" for line in f]
        return features
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {filename}")
        return None
    except Exception as e:
        print(f"Lỗi khi đọc file {filename}: {e}")
        return None

feature_paths = read(ROOT + "/data/result/used.txt")
chunk_size = 500
chunks = [feature_paths[i:i + chunk_size] for i in range(0, len(feature_paths), chunk_size)]

with open(os.path.join(ROOT, "data/result/used2.txt"), "w") as f_selected, \
     open(os.path.join(ROOT, "data/result/unused2.txt"), "w") as f_unselected:

    all_selected_features = []
    all_unselected_features = []
    
    target = pd.read_feather(utils.get_TARGET_path())
    target.columns = ["TARGET"]

    for chunk in chunks:
        X = pd.DataFrame()
        print(f"Đang xử lý chunk: {chunk[0]}...")
        try:
            chunk_df = pd.concat([pd.read_feather(os.path.join(ROOT, file_path)).head(HEAD) for file_path in chunk], axis=1)
        except FileNotFoundError as e:
            print(f"Lỗi: Không tìm thấy file: {e.filename}")
            continue

        X = pd.concat([chunk_df, target], axis=1)
        
        dropped_correlation_features = handle_low_correlation_with_target(X, target_column="TARGET")
            
        selected_features = X_filtered.columns
        unselected_features = dropped_correlation_features

        f_selected.write("\n".join(selected_features.to_list()) + "\n")
        f_unselected.write("\n".join(unselected_features))
        
        all_selected_features.extend(selected_features)
        all_unselected_features.extend(unselected_features)