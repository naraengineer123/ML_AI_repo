import joblib
import xgboost as xgb
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from google.cloud import aiplatform

from data_preprocessing import load_data, split_data
from feature_engineering import feature_engineering_pipeline


# ==========================
# CONFIGURATION
# ==========================

TARGET = "target_label"
DATA_PATH = "data/sample_data.csv"
MODEL_PATH = "models/xgboost_best_model.pkl"

PROJECT_ID = "your-project-id"
REGION = "us-central1"
MODEL_DISPLAY_NAME = "xgboost-ml-model"


# ==========================
# LOAD DATA
# ==========================

df = load_data(DATA_PATH)

df, card_info = feature_engineering_pipeline(df)

X_train, X_test, y_train, y_test = split_data(df, TARGET)


# ==========================
# XGBOOST MODEL
# ==========================

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss"
)


# ==========================
# HYPERPARAMETER GRID
# ==========================

param_grid = {

    "max_depth": [3, 5, 7, 9],

    "learning_rate": [0.01, 0.05, 0.1, 0.2]

}


# ==========================
# GRID SEARCH
# ==========================

grid_search = GridSearchCV(

    estimator=model,

    param_grid=param_grid,

    cv=3,

    scoring="accuracy",

    verbose=2

)

grid_search.fit(X_train, y_train)


# ==========================
# BEST MODEL
# ==========================

best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)


# ==========================
# MODEL EVALUATION
# ==========================

predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)


# ==========================
# SAVE MODEL
# ==========================

joblib.dump(best_model, MODEL_PATH)

print("Model saved:", MODEL_PATH)


# ==========================
# REGISTER MODEL IN VERTEX AI
# ==========================

aiplatform.init(

    project=PROJECT_ID,

    location=REGION

)

model = aiplatform.Model.upload(

    display_name=MODEL_DISPLAY_NAME,

    artifact_uri="models/",

    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-5:latest"

)

print("Model registered in Vertex AI")

print(model.resource_name)