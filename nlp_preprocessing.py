# src/nlp_preprocessing.py
import pandas as pd
from preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def add_text_features(df, text_col='refund_reason_text'):
    df = df.copy()
    df[text_col + '_clean'] = df[text_col].apply(clean_text)
    df['reason_len'] = df[text_col].apply(lambda x: len(str(x)))
    df['reason_word_count'] = df[text_col].apply(lambda x: len(str(x).split()))
    return df

def fit_tfidf(df, text_col='refund_reason_text_clean', max_features=200, save_path='../models/tfidf.joblib'):
    vec = TfidfVectorizer(max_features=max_features, stop_words='english')
    vec.fit(df[text_col].astype(str))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(vec, save_path)
    return vec

def transform_tfidf(df, vec, text_col='refund_reason_text_clean'):
    return vec.transform(df[text_col].astype(str))
