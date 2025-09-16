# src/preprocessing.py
import pandas as pd
import re

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df['refund_reason_text'] = df['refund_reason_text'].fillna('')
    df['images_provided'] = df['images_provided'].fillna(0).astype(int)
    df['partial_refund'] = df['partial_refund'].fillna(0).astype(int)
    df['previous_refunds'] = df['previous_refunds'].fillna(0).astype(int)
    df['user_account_age_days'] = df['user_account_age_days'].fillna(
        df['user_account_age_days'].median()
    )
    return df

def derive_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['request_hour'] = df['request_time'].dt.hour
    df['days_since_order'] = (df['request_time'] - df['order_time']).dt.days
    return df

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()
