import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = 'data/diagnosis_history.db'

def init_db():
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            age INTEGER,
            sex TEXT,
            cp REAL,
            trestbps REAL,
            chol REAL,
            fbs TEXT,
            restecg REAL,
            thalach REAL,
            exang TEXT,
            oldpeak REAL,
            slope REAL,
            ca REAL,
            thal REAL,
            prediction INTEGER,
            probability REAL,
            diagnosis_date TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_patient_record(patient_id, raw_input_data, prediction, probability):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('''
        INSERT INTO patients (
            patient_id, age, sex, cp, trestbps, chol, fbs, restecg, 
            thalach, exang, oldpeak, slope, ca, thal, prediction, probability, diagnosis_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        patient_id,
        raw_input_data.get('Tuổi', 0),
        raw_input_data.get('Giới tính', ''),
        raw_input_data.get('Loại đau ngực', 0),
        raw_input_data.get('Huyết áp', 0),
        raw_input_data.get('Cholesterol', 0),
        raw_input_data.get('Đường huyết cao', ''),
        raw_input_data.get('Điện tâm đồ', 0),
        raw_input_data.get('Nhịp tim max', 0),
        raw_input_data.get('Đau ngực gắng sức', ''),
        raw_input_data.get('Suy giảm ST', 0),
        raw_input_data.get('Độ dốc ST', 0),
        raw_input_data.get('Mạch máu', 0),
        raw_input_data.get('Thalassemia', 0),
        int(prediction),
        float(probability),
        date_str
    ))
    conn.commit()
    conn.close()

def get_all_records():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM patients ORDER BY id DESC", conn)
    conn.close()
    return df
