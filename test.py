# test.py
import pandas as pd
from nlp_preprocessing import add_text_features

data = {'refund_reason_text': [
    "I want a refund!!",
    "Product arrived late, very disappointed.",
    "GOOD SERVICE, THANK YOU!"
]}

df = pd.DataFrame(data)
df = add_text_features(df)
print(df)
