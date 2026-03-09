import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from data_preprocessing import load_data, split_data
from feature_engineering import create_features

TARGET = "target_label"

df = load_data("data/sample_data.csv")

df = create_features(df)

X_train, X_test, y_train, y_test = split_data(df, TARGET)

param_grid = {
    "max_depth": [4,6,8],
    "learning_rate": [0.01,0.05,0.1],
    "n_estimators": [100,200,300],
    "subsample": [0.7,0.8,1]
}

model = xgb.XGBClassifier()

grid = GridSearchCV(model, param_grid, cv=3, scoring="accuracy")

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)