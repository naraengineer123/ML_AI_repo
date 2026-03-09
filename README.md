# ML/AI Model Training Repository

Reusable machine learning pipeline including:

• Feature Engineering
• XGBoost Model Training
• Hyperparameter Tuning
• Model Serialization (.pkl)
• Vertex AI Model Registry
• Reusable pipeline for multiple datasets

## Pipeline

Raw Data → Feature Engineering → Train/Test Split → Model Training → Hyperparameter Tuning → Model Save → Vertex AI Registration

## Setup

pip install -r requirements.txt

## Train Model

python src/train_model.py

## Hyperparameter Tuning

python src/hyperparameter_tuning.py

## Register Model in Vertex AI

python src/register_vertex_model.py

## Run Predictions

python src/predict.py# ML_AI_repo
# ML_AI_repo
