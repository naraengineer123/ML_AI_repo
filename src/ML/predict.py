import joblib
import pandas as pd

from feature_engineering import create_features

model = joblib.load("models/xgboost_model.pkl")

data = pd.read_csv("data/sample_data.csv")

data = create_features(data)

predictions = model.predict(data)

print(predictions)