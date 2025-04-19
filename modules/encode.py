# mã hóa dữ liệu rời rạc

from sklearn.preprocessing import LabelEncoder


def label_encode(df, categorical_features):
    le = LabelEncoder()
    for c in categorical_features:
        df[c] = df[c].fillna('None')
        le.fit(df[c])
        df[c] = le.transform(df[c])
    return df
