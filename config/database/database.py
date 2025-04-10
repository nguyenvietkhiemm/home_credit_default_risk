import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        try:
            self.conn = mysql.connector.connect(
                host=os.getenv("HOST"),
                user=os.getenv("USER"),
                password=os.getenv("PASSWORD"),
                database=os.getenv("DATABASE")
            )
            self.cursor = self.conn.cursor()
            print("Kết nối MySQL thành công!")
        except mysql.connector.Error as e:
            print(f"Lỗi kết nối MySQL: {e}")
            self.conn = None

    def fetch_data(self, query):
        if not self.conn or not self.conn.is_connected():
            print("Kết nối MySQL đã đóng!")
            return None
        
        self.cursor.execute(query)
        columns = [desc[0] for desc in self.cursor.description]
        data = self.cursor.fetchall()
        return columns, data

    def close(self):
        if self.conn:
            self.cursor.close()
            self.conn.close()
            print("Đã đóng kết nối MySQL.")
