# mã hóa dữ liệu rời rạc

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

def label_encode(df, categorical_features):
    le = LabelEncoder()
    for c in categorical_features:
        df[c] = df[c].fillna('None')
        le.fit(df[c])
        df[c] = le.transform(df[c])
    return df

def target_encode(df, categorical_features):
    skf = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=SEED)
    
    
