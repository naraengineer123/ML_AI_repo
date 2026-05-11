import joblib
import xgboost as xgb

from data_preprocessing import load_data, split_data
from feature_engineering import create_features

TARGET = "target_label"

df = load_data("data/sample_data.csv")

df = create_features(df)

X_train, X_test, y_train, y_test = split_data(df, TARGET)

model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8
)

model.fit(X_train, y_train)

joblib.dump(model, "models/xgboost_model.pkl")

print("Model trained and saved")