import pandas as pd
import numpy as np
import matplotlib as plt
from config import ROOT  # lib này được khởi tạo ban đầu dự án
import os
from dotenv import load_dotenv
load_dotenv()
from config.database import Database
db = Database()
from modules.ingestion import Ingestion
load_save = Ingestion(db).load_save
table_names = os.getenv("TABLE_NAMES").split(',')
for table_name in table_names:
    print("===", table_name, "===")
    load_save(table_name)
