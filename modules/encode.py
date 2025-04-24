# mã hóa dữ liệu rời rạc

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from config import FOLD, SEED

def label_encode(df, categorical_features):
    le = LabelEncoder()
    for c in categorical_features:
        df[c] = df[c].fillna('None')
        le.fit(df[c])
        df[c] = le.transform(df[c])
    return df

def target_encode(train, test, categorical_features):
    skf = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=SEED) # chia fold cân bằng label
    
    usecols = []
    for c in tqdm(categorical_features):
        train[c+'_ta'] = 0
        
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        for i,(train_index, test_index) in enumerate(skf.split(train, train.TARGET)):
            enc = train.iloc[train_index].groupby(c)['TARGET'].mean()
            train.loc[test_index, c+'_ta'] = train.loc[test_index, c].map(enc)
            
        enc = train.groupby(c)['TARGET'].mean()
        test[c+'_ta'] = test[c].map(enc)
        
        usecols.append(c+'_ta')
        
    
    train['fold'] = 0
    for i,(train_index, test_index) in enumerate(skf.split(train, train.TARGET)):
        train.loc[test_index, 'fold'] = i

    for c in categorical_features:
        cat_min = train.groupby(['fold', c]).size().min()
        print(f'target_encode cat min {c}: {cat_min}')
        
    return train, test
