# src/model_training.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import joblib

from nlp_preprocessing import add_text_features
from feature_engineering import create_features
from utils import save_model

def train_save_model():
    # 🔹 Safe path resolution
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
    DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'synthetic_ecommerce_churn_dataset.csv'))
    MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'models', 'xgb.joblib'))

    # 1️⃣ Load dataset
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"✅ Dataset loaded: {df.shape} rows")

    # 2️⃣ Convert datetime columns if they exist
    for col in ['request_time', 'order_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 3️⃣ Add NLP features
    df = add_text_features(df)

    # 4️⃣ Time-based feature if both columns exist
    if 'request_time' in df.columns and 'order_time' in df.columns:
        df['days_since_order'] = (df['request_time'] - df['order_time']).dt.days
    else:
        df['days_since_order'] = 0  # default value if no columns

    # 5️⃣ Features & Target
    if 'label' not in df.columns:
        print("❌ Dataset must have 'label' column for fraud indicator (0/1)")
        return

    drop_cols = ['refund_id','order_id','user_id','label']
    if 'request_time' in df.columns:
        drop_cols.append('request_time')
    if 'order_time' in df.columns:
        drop_cols.append('order_time')

    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['label']

    # 6️⃣ Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7️⃣ Feature Engineering (TF-IDF + numeric)
    X_train_fe = create_features(X_train, fit_tfidf=True)
    X_test_fe = create_features(X_test, fit_tfidf=False)

    # 8️⃣ Train XGBoost
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    clf.fit(X_train_fe, y_train)

    # 9️⃣ Evaluate
    preds = clf.predict(X_test_fe)
    proba = clf.predict_proba(X_test_fe)[:,1]

    print("\n✅ Classification Report:\n")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    # 🔟 Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(X_test_fe, os.path.normpath(os.path.join(BASE_DIR, '..', 'models', 'X_sample.joblib')))
    print(f"\n✅ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_save_model()
