# MLOps Architecture

## AI Architect / Engineer Overview

This module provides the MLOps foundation for predictive healthcare modeling. It supports model development, feature engineering, training, tuning, serialization, registry integration, and future production deployment.

The current implementation uses XGBoost as the baseline predictive model and Vertex AI as the model registry target.

## MLOps Lifecycle

```text
Data Source
  -> Data Validation
  -> Feature Engineering
  -> PCA Dimensionality Reduction
  -> Train/Test Split
  -> Model Training
  -> Hyperparameter Tuning
  -> Model Evaluation
  -> Model Serialization
  -> Vertex AI Model Registry
  -> Deployment / Batch Inference
  -> Monitoring and Retraining
```

## Model Training Architecture

```text
Raw Data
  -> Feature Engineering
  -> PCA Transformation
  -> XGBoost Training
  -> Grid Search Tuning
  -> Accuracy Evaluation
  -> Save Best Model
  -> Register Model in Vertex AI
```

## Current Capabilities

- CSV-based data ingestion
- Reusable preprocessing functions
- Date validation
- Null-value checks
- Categorical feature identification
- Cardinality analysis
- One-hot encoding for low-cardinality categorical columns
- Derived feature creation
- PCA dimensionality reduction for related feature groups
- Train/test split
- XGBoost model training
- Grid search hyperparameter tuning
- Accuracy evaluation
- `.pkl` model serialization
- Vertex AI model upload
- Local prediction script

## Model Components

| Component | File | Purpose |
| --- | --- | --- |
| Data preprocessing | `data_preprocessing.py` | Loads source data and creates train/test splits. |
| Feature engineering | `feature_engineering.py` | Creates derived features, validates dates, checks nulls, and handles categorical columns. |
| Training | `train_model.py` | Trains the baseline XGBoost classifier. |
| Hyperparameter tuning | `hyperparameter_tuning.py` | Runs grid search and saves the best model. |
| Model registration | `register_vertex_model.py` | Uploads the trained model artifact to Vertex AI. |
| Prediction | `predict.py` | Loads the saved model and runs local inference. |

## PCA in the MLOps Pipeline

PCA, or Principal Component Analysis, is used to reduce many related input features into a smaller number of model-ready components while preserving the strongest patterns in the data.

In healthcare risk modeling, multiple columns may describe a similar signal. For example:

```text
national_income_percentile
state_estimated_income_index
county_estimated_income_index
zip_income_rank
household_income_score
```

These columns are all related to socioeconomic or income profile. PCA can transform them into one or more compact components:

```text
income_component_1
income_component_2
```

The same pattern can be applied to health-risk features:

```text
diabetes_risk_score
hospitalization_risk_score
chronic_condition_count
medication_gap_score
care_gap_score
```

PCA helps the model by:

- Reducing feature count
- Removing highly correlated signals
- Lowering noise
- Making training and inference faster
- Creating compact features for downstream models

In production, the PCA transformer should be saved as a `.pkl` file and reused during inference. The same PCA transformation learned during training must be applied to the BigQuery inference DataFrame before calling the model.

Example PCA artifact:

```text
models/pca_transformer.pkl
```

Example model artifact:

```text
models/xgboost_model.pkl
```

## Inference with `.pkl` Files

The inference flow loads trained `.pkl` artifacts, creates a DataFrame from BigQuery query results, applies the same feature engineering and PCA transformation used during training, and finally calls the model for prediction.

Production inference flow:

```text
BigQuery Source Table
  -> SQL Query
  -> pandas DataFrame
  -> Feature Engineering
  -> Load PCA .pkl
  -> PCA Transform
  -> Load Model .pkl
  -> Model Inference
  -> Prediction DataFrame
  -> Write Scores back to BigQuery
```

Example inference pattern:

```python
import joblib
import pandas as pd
from google.cloud import bigquery

from feature_engineering import create_features


client = bigquery.Client()

query = """
SELECT
  member_id,
  feature1,
  feature2,
  feature3,
  feature4
FROM `project.dataset.member_features`
"""

# 1. Create DataFrame from BigQuery
df = client.query(query).to_dataframe()

# 2. Keep member identifiers for final output
member_ids = df["member_id"]
features_df = df.drop(columns=["member_id"])

# 3. Apply the same feature engineering used during training
features_df = create_features(features_df)

# 4. Load PCA transformer and model artifacts
pca = joblib.load("models/pca_transformer.pkl")
model = joblib.load("models/xgboost_model.pkl")

# 5. Transform features using trained PCA
pca_features = pca.transform(features_df)

# 6. Run model inference
predictions = model.predict(pca_features)
prediction_probabilities = model.predict_proba(pca_features)[:, 1]

# 7. Create final scoring DataFrame
scores_df = pd.DataFrame(
    {
        "member_id": member_ids,
        "prediction": predictions,
        "risk_score": prediction_probabilities,
    }
)

# 8. Load final scores into BigQuery
client.load_table_from_dataframe(
    scores_df,
    "project.dataset.member_risk_scores",
).result()
```

This approach ensures the same preprocessing, PCA transformation, and model inference logic is used consistently across training and production scoring.

## Setup

Install root ML dependencies from the repository root:

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python src/ML/train_model.py
```

## Hyperparameter Tuning

```bash
python src/ML/hyperparameter_tuning.py
```

## Register Model in Vertex AI

```bash
python src/ML/register_vertex_model.py
```

## Run Predictions

```bash
python src/ML/predict.py
```

## Vertex AI Registry Pattern

Vertex AI is used as the model registry layer for production MLOps.

Registry responsibilities:

- Store versioned model artifacts
- Track deployable models
- Support controlled promotion from experiment to production
- Enable downstream deployment to managed serving or batch inference
- Provide a foundation for model governance and auditability

## Production MLOps Design

For production readiness, the model lifecycle should include:

- Automated training pipeline execution
- Dataset versioning
- Feature validation
- Model evaluation gates
- Model registry approval workflow
- Batch inference scheduling
- BigQuery-to-DataFrame inference jobs
- Reuse of saved PCA and model `.pkl` artifacts during scoring
- Model drift monitoring
- Performance monitoring
- Retraining triggers
- CI/CD for model code and deployment artifacts

## Module Structure

```text
ML/
├─ data_preprocessing.py
├─ feature_engineering.py
├─ train_model.py
├─ hyperparameter_tuning.py
├─ register_vertex_model.py
└─ predict.py
```

## Engineering Highlights

- Built a reusable model training pipeline for healthcare risk-scoring use cases.
- Added feature engineering utilities for numeric, date, and categorical processing.
- Included hyperparameter tuning with `GridSearchCV`.
- Serialized model artifacts for repeatable inference.
- Integrated Vertex AI model registration for enterprise MLOps.
- Designed the lifecycle for future monitoring, retraining, and deployment automation.
