import pandas as pd
import numpy as np
import matplotlib as plt
from sitecustomize import ROOT  # lib này được khởi tạo ban đầu dự án
from config.database.database import Database
db = Database()
import pandas as pd
def load_save(table_name=None, batch_size=100000):
    if not table_name:
        return
    file_path_csv = f"{ROOT}/data/csv/{table_name}.csv"
    file_path_pkl = f"{ROOT}/data/pkl/{table_name}.p"
    query = f"SELECT * FROM {table_name}"
    db.cursor.execute(query)
    columns = [desc[0] for desc in db.cursor.description]
    first_write = True
    dfs = []
    while True:
        data = db.cursor.fetchmany(batch_size)
        if not data:
            break
        df = pd.DataFrame(columns=columns, data=data)
        dfs.append(df)
        df.to_csv(file_path_csv, mode="w" if first_write else "a", index=False, header=first_write)
        first_write = False
        print(f"Saved {len(data)} rows...")
    if dfs:
        pd.concat(dfs).to_pickle(file_path_pkl)
        print(f"Saved all data to {file_path_pkl}")
load_save(table_name="application_train")
load_save(table_name="application_test")
load_save(table_name="previous_application")
load_save(table_name="bureau")
load_save(table_name="bureau_balance")
load_save(table_name="credit_card_balance")
load_save(table_name="installments_payments")
load_save(table_name="pos_cash_balance")
load_save(table_name="homecredit_columns_description")
df = pd.read_csv(ROOT + "/data/csv/homecredit_columns_description.csv")
df["Table"] = df["Table"].replace("application_{train|test}.csv", "train_test")
df
tmp = df[df["Table"] == "train_test"]
tmp["Table"] = "test"
tmp
tmp = tmp[tmp["Row"] != "TARGET"]
df = pd.concat([df, tmp])
df = df.replace("train_test", "train")
df["Table"] = df["Table"].str.replace(r".csv$", "", regex=True)
df
df.to_csv(ROOT + "/data/csv/description.csv", index=False)
df.to_pickle(ROOT + "/data/pkl/description.p")
