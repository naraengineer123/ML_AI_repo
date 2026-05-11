# Enterprise AI Data Agent Platform

This repository demonstrates an AI Architect / Engineer implementation of a healthcare-focused enterprise AI platform. It is organized into two major architecture areas:

1. [Enterprise AI Data Agent](src/Health_AI_Agent/README.md)
2. [MLOps Architecture](src/ML/README.md)

## Project Summary

The platform is designed to help care managers and healthcare business users identify members with similar health-related issues, such as diabetes risk, chronic disease indicators, hospitalization risk, medication gaps, and other high-risk clinical patterns.

Instead of requiring users to manually search across claims, clinical, pharmacy, lab, policy, and care-management datasets, the system provides a natural language AI agent that retrieves context, generates governed SQL, executes BigQuery analytics, and returns care-manager-ready insights.

## Architecture at a Glance

```text
Care Manager / Business User
        |
        v
Enterprise AI Data Agent
        |
        +--> Multi-Agent Orchestration
        +--> RAG + pgvector Retrieval
        +--> Semantic Chunking
        +--> Clinical Transformer Embeddings
        +--> BigQuery Query Execution
        +--> Care-Management Report Generation
        |
        v
MLOps Architecture
        |
        +--> Feature Engineering
        +--> XGBoost Model Training
        +--> Hyperparameter Tuning
        +--> Model Serialization
        +--> Vertex AI Model Registry
```

## Key Engineering Highlights

- Designed a modular multi-agent AI architecture for enterprise healthcare analytics.
- Implemented RAG with pgvector to ground responses in clinical, policy, metadata, and schema context.
- Added semantic chunking based on topic shifts and cosine similarity instead of fixed-size splitting.
- Built a BigQuery-based clinical embedding pipeline for 50M input records.
- Designed dynamic `n1-standard-16` worker scaling based on source data volume.
- Used a Flash Attention-based Clinical Transformer model pattern to generate 256-dimensional embeddings.
- Added SQL validation guardrails before BigQuery execution.
- Included an MLOps-ready training flow with XGBoost, tuning, serialization, and Vertex AI registration.
- Used Claude as an AI coding assistant to accelerate Python implementation while owning architecture, review, and integration decisions.

## Repository Layout

```text
enterprise-ai-data-agent-main/
├─ README.md
├─ requirements.txt
├─ config/
│  └─ model_config.yaml
├─ data/
│  └─ sample_data.csv
└─ src/
   ├─ Health_AI_Agent/
   │  ├─ README.md
   │  ├─ app/
   │  ├─ agents/
   │  ├─ db/
   │  └─ ingestion/
   └─ ML/
      ├─ README.md
      ├─ data_preprocessing.py
      ├─ feature_engineering.py
      ├─ train_model.py
      ├─ hyperparameter_tuning.py
      ├─ register_vertex_model.py
      └─ predict.py
```

## Detailed Documentation

- [Enterprise AI Data Agent](src/Health_AI_Agent/README.md): Multi-agent architecture, RAG, semantic chunking, BigQuery, pgvector, embedding pipeline, dynamic GCP worker scaling, and API usage.
- [MLOps Architecture](src/ML/README.md): Model training lifecycle, feature engineering, XGBoost training, hyperparameter tuning, Vertex AI registry, and production MLOps design.

