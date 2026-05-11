# Enterprise AI Data Agent

This repository is divided into two major sections:

1. Build AI Agent for Care Managers
2. ML/AI Model Training Repository

## 1. Build AI Agent for Care Managers

This section implements a multi-agent AI platform for care managers, healthcare analysts, and business users who need to query enterprise healthcare data using natural language.

The system reads data from source BigQuery tables, generates clinical embeddings, stores retrieval-ready vectors, and uses a multi-RAG agent architecture to answer care-management questions.

Example care-manager questions:

- Which members have diabetes risk?
- Show members with high cardiovascular risk.
- Identify members with multiple chronic conditions.
- Which members have hospitalization risk greater than 0.7?
- Which members should be prioritized for care management outreach?

### Core Capabilities

- Natural language querying across enterprise healthcare data
- Source data ingestion from BigQuery
- Transformer embedding generation for clinical text
- 256-dimensional clinical embeddings
- Distributed batch embedding across GCP compute instances
- Final embedding output loaded into BigQuery
- Multi-RAG architecture for retrieval and response generation
- pgvector support for vector search and policy/context retrieval
- SQL generation, validation, analysis, and report generation agents

### Multi-Agent Architecture

The care-manager AI agent is implemented as a cooperative multi-agent system.

Agents:

- SQL Agent: Converts natural language questions into BigQuery SQL.
- RAG Agent: Retrieves relevant context using embeddings and vector search.
- QA Agent: Validates generated SQL and checks for unsafe operations.
- Analyst Agent: Converts raw query results into concise insights.
- Report Agent: Generates final care-manager-facing summaries and recommendations.
- Orchestrator: Coordinates the end-to-end flow across all agents.

High-level request flow:

```text
Care Manager Question
  -> FastAPI Query Endpoint
  -> Agent Orchestrator
  -> RAG Context Retrieval
  -> SQL Generation
  -> SQL Validation
  -> BigQuery Execution
  -> Result Analysis
  -> Final Report
```

### RAG and Vector Architecture

The platform implements a multi-RAG architecture using:

- BigQuery for enterprise source data and final embedding storage
- pgvector for vector similarity search
- Clinical Transformer embeddings for healthcare text
- Retrieved policy, metadata, schema, and clinical context
- LLM-based SQL and report generation

RAG flow:

```text
Source Documents / Clinical Text
  -> Clinical Transformer Embedding
  -> 256-D Vector
  -> BigQuery Embedding Table
  -> pgvector Retrieval Layer
  -> Agent Context
  -> SQL / Analysis / Report Output
```

### BigQuery Clinical Embedding Pipeline

The ingestion pipeline reads records from a source BigQuery table, applies a Flash Attention-based Clinical Transformer model, creates 256-dimensional embeddings, and loads the final embedded records back into BigQuery.

Target design:

| Item | Value |
| --- | --- |
| Input data volume | 50M records |
| Source | BigQuery source table |
| Model | Flash Attention-based Clinical Transformer model |
| Embedding size | 256 dimensions |
| GCP instance type | n1-standard-16 |
| Records per instance | 100,000 |
| Total instances | 500 |
| Batch size | 2,000 |
| Final destination | BigQuery embedding table |
| Vector retrieval | pgvector |

Each instance runs one shard of the embedding job. With 50M records and 100k records per instance, the full job spins up 500 workers.

```text
50,000,000 records / 100,000 records per instance = 500 instances
```

### Embedding Worker Configuration

Set these environment variables before running an embedding worker:

```bash
export BQ_SOURCE_TABLE="project.dataset.source_table"
export BQ_DEST_TABLE="project.dataset.clinical_embeddings"
export BQ_ID_COLUMN="member_id"
export BQ_TEXT_COLUMNS="diagnosis_text,clinical_notes"
export EMBEDDING_MODEL_NAME="emilyalsentzer/Bio_ClinicalBERT"
export EMBEDDING_TOTAL_SHARDS=500
export EMBEDDING_RECORDS_PER_INSTANCE=100000
export EMBEDDING_BATCH_SIZE=2000
export EMBEDDING_DIMENSION=256
```

Run one worker shard:

```bash
cd src/Health_AI_Agent
python -m ingestion.ingest_docs --shard-index 0
```

For the full workload, launch shard indexes `0` through `499` across 500 GCP `n1-standard-16` instances.

### Care Agent API

The FastAPI service exposes the AI agent interface.

Run locally:

```bash
cd src/Health_AI_Agent
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Health check:

```bash
GET /
```

Query endpoint:

```bash
POST /query
```

Example request:

```json
{
  "question": "Which members have diabetes risk?"
}
```

### Main Care Agent Files

```text
src/Health_AI_Agent/
├─ app/
│  ├─ main.py
│  └─ config.py
├─ agents/
│  ├─ orchestrator.py
│  ├─ sql_agent.py
│  ├─ rag_agent.py
│  ├─ qa_agent.py
│  ├─ analyst_agent.py
│  └─ report_agent.py
├─ db/
│  ├─ bigquery_client.py
│  └─ pgvector_client.py
└─ ingestion/
   ├─ ingest_docs.py
   ├─ chunking.py
   └─ embed_store.py
```

## 2. ML/AI Model Training Repository

This section contains a reusable machine learning pipeline for training and registering predictive models.

Included capabilities:

- Data loading from CSV
- Feature engineering
- Train/test split
- XGBoost model training
- Hyperparameter tuning
- Model serialization with `.pkl`
- Vertex AI model registration
- Prediction script for local inference

### ML Pipeline

```text
Raw Data
  -> Feature Engineering
  -> Train/Test Split
  -> XGBoost Model Training
  -> Hyperparameter Tuning
  -> Model Save
  -> Vertex AI Registration
```

### Setup

```bash
pip install -r requirements.txt
```

### Train Model

```bash
python src/ML/train_model.py
```

### Hyperparameter Tuning

```bash
python src/ML/hyperparameter_tuning.py
```

### Register Model in Vertex AI

```bash
python src/ML/register_vertex_model.py
```

### Run Predictions

```bash
python src/ML/predict.py
```

### Main ML Files

```text
src/ML/
├─ data_preprocessing.py
├─ feature_engineering.py
├─ train_model.py
├─ hyperparameter_tuning.py
├─ register_vertex_model.py
└─ predict.py
```
