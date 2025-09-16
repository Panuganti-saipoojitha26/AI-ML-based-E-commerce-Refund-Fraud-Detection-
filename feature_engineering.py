# src/feature_engineering.py
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def create_features(df, fit_tfidf=True, text_col='refund_reason_text_clean', tfidf_path='../models/tfidf.joblib'):
    df = df.copy()
    for col in ['images_provided','partial_refund','previous_refunds','user_account_age_days']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if fit_tfidf:
        vec = TfidfVectorizer(max_features=200, stop_words='english')
        tfidf_matrix = vec.fit_transform(df[text_col].astype(str))
        os.makedirs(os.path.dirname(tfidf_path), exist_ok=True)
        joblib.dump(vec, tfidf_path)
    else:
        vec = joblib.load(tfidf_path)
        tfidf_matrix = vec.transform(df[text_col].astype(str))

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
    tfidf_df.index = df.index
    df = pd.concat([df, tfidf_df], axis=1)
    return df
