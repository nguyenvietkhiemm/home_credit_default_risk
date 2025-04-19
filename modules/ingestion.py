# lấy dữ liệu từ database

import pandas as pd
from config import ROOT  # lib này được khởi tạo ban đầu dự án


class Ingestion:
    def __init__(self, db=None):
        if not db:
            print("no database")
        self.db = db

    def load_save(self, table_name=None, batch_size=100000):
        if not table_name:
            return

        db = self.db

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

            df.to_csv(file_path_csv, mode="w" if first_write else "a",
                      index=False, header=first_write)
            first_write = False
            print(f"Saved {len(data)} rows...")

        if dfs:
            pd.concat(dfs).to_pickle(file_path_pkl)
            print(f"Saved all data to {file_path_pkl}")
