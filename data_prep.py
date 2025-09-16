# src/data_prep.py
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_data
from preprocessing import basic_cleaning, derive_time_features

def prepare_train_test(path: str, test_size=0.2, random_state=42):
    df = load_data(path)
    df = basic_cleaning(df)
    df = derive_time_features(df)
    X = df.drop(columns=['refund_id','order_id','user_id','request_time','order_time','label'])
    y = df['label']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
