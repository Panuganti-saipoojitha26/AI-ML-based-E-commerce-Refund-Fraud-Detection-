# src/inference_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from utils import load_model
from nlp_preprocessing import clean_text
from feature_engineering import create_features
from typing import Optional
import uvicorn

app = FastAPI(title="Refund Fraud Detection API")

class RefundRequest(BaseModel):
    order_id: str
    user_id: str
    amount: float
    request_time: str
    order_time: str
    product_id: Optional[str] = None
    product_category: Optional[str] = None
    refund_reason_text: Optional[str] = ''
    images_provided: int = 0
    partial_refund: int = 0
    previous_refunds: int = 0
    user_account_age_days: int = 0
    shipping_country: Optional[str] = None
    is_guest: int = 0
    device_type: Optional[str] = None

@app.post("/predict")
async def predict(refund: RefundRequest):
    row = pd.DataFrame([refund.dict()])
    row['request_time'] = pd.to_datetime(row['request_time'])
    row['order_time'] = pd.to_datetime(row['order_time'])
    row['refund_reason_text_clean'] = row['refund_reason_text'].apply(clean_text)
    row['reason_len'] = row['refund_reason_text'].apply(lambda x: len(x or ""))
    row['reason_word_count'] = row['refund_reason_text'].apply(lambda x: len((x or "").split()))
    row['days_since_order'] = (row['request_time'] - row['order_time']).dt.days
    row['request_hour'] = row['request_time'].dt.hour

    X = create_features(row, fit_tfidf=False)

    model = load_model('../models/xgb.joblib')
    if model is None:
        return {"error": "Model not found. Please train first."}

    proba = model.predict_proba(X)[:,1][0]
    pred = int(proba > 0.5)

    return {"prediction": pred, "fraud_probability": float(proba)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
