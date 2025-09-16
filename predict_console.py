# predict_console.py
import os
import pandas as pd
import joblib
from nlp_preprocessing import clean_text, add_text_features
from feature_engineering import create_features

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'data', 'synthetic_ecommerce_churn_dataset.csv'))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'models', 'xgb.joblib'))

# Load dataset
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded: {df.shape}")

# Pick a few rows to test
df_test = df.sample(n=5, random_state=42)  # 5 refund requests

# Add NLP features
df_test = add_text_features(df_test)

# Prepare features
drop_cols = ['refund_id','order_id','user_id','label']
X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors='ignore')

# Load model
clf = joblib.load(MODEL_PATH)

# Feature engineering
X_test_fe = create_features(X_test, fit_tfidf=False)  # no fitting, just transform

# Predict
preds = clf.predict(X_test_fe)
proba = clf.predict_proba(X_test_fe)[:,1]

# Show output like the app
for i, row in df_test.iterrows():
    print(f"\nRefund Request ID: {row.get('refund_id', i)}")
    print(f"User ID: {row.get('user_id', 'NA')}")
    print(f"Predicted Fraud: {int(preds[i])}")
    print(f"Fraud Probability: {proba[i]:.2f}")
