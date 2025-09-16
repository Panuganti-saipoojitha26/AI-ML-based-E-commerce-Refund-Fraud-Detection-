# src/utils.py
import pandas as pd
import joblib
import os

def load_data(path: str) -> pd.DataFrame:
    """Load CSV and parse datetime columns."""
    try:
        df = pd.read_csv(path)
        for col in ['request_time', 'order_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"✅ Data loaded successfully from {path}")
        return df
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        return None
    except Exception as e:
        print("❌ Error while loading data:", e)
        return None

def save_model(model, path='../models/clf.joblib'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")

def load_model(path='../models/clf.joblib'):
    try:
        model = joblib.load(path)
        print(f"✅ Model loaded from {path}")
        return model
    except Exception as e:
        print("❌ Error loading model:", e)
        return None
