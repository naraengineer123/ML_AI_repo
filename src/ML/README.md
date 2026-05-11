# MLOps Architecture

## AI Architect / Engineer Overview

This module provides the MLOps foundation for predictive healthcare modeling. It supports model development, feature engineering, training, tuning, serialization, registry integration, and future production deployment.

The current implementation uses XGBoost as the baseline predictive model and Vertex AI as the model registry target.

## MLOps Lifecycle

```text
Data Source
  -> Data Validation
  -> Feature Engineering
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

