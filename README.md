# Enterprise AI Data Agent

## AI Architect / Engineer Project Overview

I designed and implemented this project as an enterprise-grade AI data agent platform for healthcare care-management teams. The platform enables care managers, analysts, and business users to ask natural language questions over large-scale enterprise healthcare data without manually writing SQL or searching across fragmented data dictionaries, policy documents, and clinical datasets.

The architecture combines multi-agent orchestration, Retrieval Augmented Generation (RAG), transformer-based clinical embeddings, BigQuery analytics, pgvector retrieval, and an MLOps-ready model training pipeline.

The goal is to provide a scalable, governed, and production-oriented AI system that can support healthcare use cases such as risk stratification, chronic condition identification, hospitalization risk review, and care outreach prioritization.

## Business Problem

Large healthcare enterprises often manage thousands of tables across cloud data warehouses. Care managers and business users need answers quickly, but the data landscape usually requires:

- Strong SQL knowledge
- Understanding of complex healthcare schemas
- Awareness of data governance policies
- Manual data validation
- Analyst or data engineer support

This creates delays in decision-making and limits direct access to actionable insights.

This platform solves that problem by converting natural language questions into governed, validated, and explainable data workflows.

## Solution Summary

The system provides an AI-powered care-manager assistant that can:

- Understand natural language healthcare questions
- Retrieve relevant metadata, policies, and clinical context using RAG
- Generate BigQuery SQL from user intent
- Validate generated SQL before execution
- Query enterprise data from BigQuery
- Analyze returned results
- Produce care-manager-ready summaries and recommendations
- Support large-scale clinical embedding generation for 50M records
- Store and retrieve embeddings using BigQuery and pgvector
- Support model training, tuning, registration, and deployment through MLOps patterns

## High-Level Architecture

```text
Care Manager / Business User
        |
        v
FastAPI AI Agent API
        |
        v
Agent Orchestrator
        |
        +--> RAG Agent
        |       |
        |       +--> pgvector Context Retrieval
        |       +--> Policy / Metadata / Clinical Context
        |
        +--> SQL Agent
        |       |
        |       +--> Natural Language to BigQuery SQL
        |
        +--> QA Agent
        |       |
        |       +--> SQL Safety Validation
        |       +--> Query Governance Checks
        |
        +--> BigQuery Client
        |       |
        |       +--> Enterprise Healthcare Tables
        |
        +--> Analyst Agent
        |       |
        |       +--> Result Summarization
        |       +--> Risk and Trend Interpretation
        |
        +--> Report Agent
                |
                +--> Final Care Manager Response
```

## Core AI Agent Architecture

The AI agent is implemented as a multi-agent system. Each agent owns a specific responsibility, which keeps the architecture modular, testable, and easier to govern.

### 1. Orchestrator Agent

The orchestrator coordinates the full request lifecycle:

- Accepts the user question
- Calls the RAG layer for context
- Sends context and question to the SQL agent
- Validates the SQL output
- Executes BigQuery queries
- Passes results to the analyst agent
- Generates the final report through the report agent

Main file:

```text
src/Health_AI_Agent/agents/orchestrator.py
```

### 2. RAG Agent

The RAG agent retrieves supporting context before SQL generation and response generation.

It is designed to retrieve:

- Data dictionary context
- Table and column metadata
- Business definitions
- Healthcare policy documents
- Clinical documentation
- Governance rules
- Previously embedded clinical text

Main file:

```text
src/Health_AI_Agent/agents/rag_agent.py
```

### 3. SQL Agent

The SQL agent converts natural language into BigQuery Standard SQL.

Design goals:

- Generate only read-only `SELECT` queries
- Avoid unsafe SQL operations
- Use retrieved schema and policy context
- Produce optimized BigQuery SQL
- Limit large result sets where needed

Main file:

```text
src/Health_AI_Agent/agents/sql_agent.py
```

### 4. QA Agent

The QA agent validates generated SQL before execution.

Validation responsibilities:

- Block destructive SQL operations
- Ensure the query is a `SELECT` query
- Reduce hallucinated or unsafe query execution
- Support future governance and compliance checks

Main file:

```text
src/Health_AI_Agent/agents/qa_agent.py
```

### 5. Analyst Agent

The analyst agent converts query results into concise insights.

Responsibilities:

- Summarize returned records
- Identify result patterns
- Prepare findings for report generation
- Support care-management interpretation

Main file:

```text
src/Health_AI_Agent/agents/analyst_agent.py
```

### 6. Report Agent

The report agent creates the final business-facing response.

Report output includes:

- Summary of findings
- Key insights
- Business interpretation
- Recommended actions
- Potential risks

Main file:

```text
src/Health_AI_Agent/agents/report_agent.py
```

## RAG and Vector Search Architecture

The platform uses a multi-RAG architecture to improve answer quality, SQL accuracy, and governance awareness.

```text
Enterprise Documents / Metadata / Clinical Text
        |
        v
Clinical Transformer Embedding Pipeline
        |
        v
256-D Embeddings
        |
        +--> BigQuery Embedding Table
        |
        +--> pgvector Vector Index
                |
                v
RAG Context Retrieval
        |
        v
Agent Prompt Context
        |
        v
SQL Generation + Report Generation
```

### RAG Components

- BigQuery stores enterprise source data and final embedding outputs.
- pgvector supports similarity search for policy, metadata, and contextual retrieval.
- Clinical Transformer embeddings represent clinical text in vector form.
- Retrieved context is injected into agent prompts to ground SQL and reporting.

## Clinical Transformer Embedding Pipeline

I added a distributed embedding pipeline that reads source records from BigQuery, generates clinical embeddings, and loads final embeddings back into BigQuery.

### Embedding Design

| Area | Design |
| --- | --- |
| Source system | BigQuery source table |
| Input volume | 50M records |
| Model type | Flash Attention-based Clinical Transformer model |
| Default model | `emilyalsentzer/Bio_ClinicalBERT` |
| Embedding size | 256 dimensions |
| Batch size | 2,000 |
| GCP instance type | `n1-standard-16` |
| Records per instance | 100,000 |
| Instance scaling | Dynamically calculated from input record count |
| Total instances for 50M records | 500 |
| Output destination | BigQuery embedding table |
| Retrieval store | pgvector |

### Dynamic Instance Scaling

The embedding architecture dynamically determines how many `n1-standard-16` GCP instances to spin up based on the input data volume.

Scaling formula:

```text
required_instances = ceil(total_input_records / records_per_instance)
```

For the 50M-record workload:

```text
50,000,000 records / 100,000 records per instance = 500 instances
```

If the source table grows or shrinks, the architecture adjusts the worker count accordingly. For example, 10M records require 100 instances, while 75M records require 750 instances at the same 100k-records-per-instance target.

Each dynamically created worker receives a unique `shard_index`. For the 50M-record workload, shards run from `0` to `499`. The source table is split using a deterministic BigQuery `FARM_FINGERPRINT` shard filter, avoiding inefficient offset-based scans.

### Embedding Worker Flow

```text
Worker Shard
  -> Read 100k records from BigQuery
  -> Process records in batches of 2,000
  -> Tokenize clinical text
  -> Run Clinical Transformer model
  -> Mean-pool token embeddings
  -> Produce normalized 256-D vectors
  -> Append embeddings to BigQuery destination table
```

Main files:

```text
src/Health_AI_Agent/ingestion/ingest_docs.py
src/Health_AI_Agent/ingestion/embed_store.py
src/Health_AI_Agent/ingestion/chunking.py
```

### Worker Configuration

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

Run one shard:

```bash
cd src/Health_AI_Agent
python -m ingestion.ingest_docs --shard-index 0
```

For full-scale execution, launch shards `0` through `499` across 500 GCP instances.

## BigQuery Data Architecture

BigQuery is used for:

- Enterprise healthcare source tables
- Generated SQL execution
- Final embedding storage
- Large-scale analytical workloads
- Downstream reporting and dashboard integration

The AI agent uses BigQuery as the system of record for structured enterprise analytics while pgvector supports semantic context retrieval.

## pgvector Architecture

pgvector is used as the vector search layer for RAG context retrieval.

Use cases:

- Retrieve schema documentation
- Retrieve healthcare policy context
- Retrieve clinical terminology context
- Retrieve data governance rules
- Support semantic matching between user questions and enterprise knowledge

This lets the AI agent generate better SQL and more reliable reports by grounding the model in enterprise-specific context.

## MLOps Architecture

The repository also includes an MLOps-ready machine learning pipeline for predictive modeling.

The ML pipeline supports:

- Data ingestion from CSV
- Feature engineering
- Train/test split
- XGBoost model training
- Hyperparameter tuning
- Model serialization
- Vertex AI model registration
- Batch/local prediction

### MLOps Lifecycle

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

### Model Training Flow

```text
Raw Data
  -> Feature Engineering
  -> XGBoost Training
  -> Grid Search Tuning
  -> Accuracy Evaluation
  -> Save Best Model
  -> Register Model in Vertex AI
```

Main files:

```text
src/ML/data_preprocessing.py
src/ML/feature_engineering.py
src/ML/train_model.py
src/ML/hyperparameter_tuning.py
src/ML/register_vertex_model.py
src/ML/predict.py
```

## Deployment Architecture

The care-manager AI agent is exposed as a FastAPI service.

```text
Client / UI / API Consumer
        |
        v
FastAPI Service
        |
        v
Agent Orchestrator
        |
        +--> OpenAI / LLM APIs
        +--> BigQuery
        +--> pgvector / PostgreSQL
        +--> Clinical Embedding Pipeline
```

The service can be containerized with Docker and deployed to a cloud runtime such as Cloud Run, GKE, or a VM-based service environment.

Main service files:

```text
src/Health_AI_Agent/app/main.py
src/Health_AI_Agent/app/config.py
src/Health_AI_Agent/Dockerfile
```

## API Usage

Run the API locally:

```bash
cd src/Health_AI_Agent
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Health check:

```bash
GET /
```

Ask a care-management question:

```bash
POST /query
```

Example request:

```json
{
  "question": "Which members have diabetes risk and should be prioritized for outreach?"
}
```

Example response fields:

```json
{
  "question": "Which members have diabetes risk and should be prioritized for outreach?",
  "sql_generated": "SELECT ...",
  "insights": "1245 rows returned",
  "report": "Summary, key insights, business interpretation, and recommended actions."
}
```

## Repository Structure

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
   │  ├─ app/
   │  │  ├─ main.py
   │  │  └─ config.py
   │  ├─ agents/
   │  │  ├─ orchestrator.py
   │  │  ├─ sql_agent.py
   │  │  ├─ rag_agent.py
   │  │  ├─ qa_agent.py
   │  │  ├─ analyst_agent.py
   │  │  └─ report_agent.py
   │  ├─ db/
   │  │  ├─ bigquery_client.py
   │  │  └─ pgvector_client.py
   │  ├─ ingestion/
   │  │  ├─ ingest_docs.py
   │  │  ├─ embed_store.py
   │  │  └─ chunking.py
   │  ├─ Dockerfile
   │  └─ requirements.txt
   └─ ML/
      ├─ data_preprocessing.py
      ├─ feature_engineering.py
      ├─ train_model.py
      ├─ hyperparameter_tuning.py
      ├─ register_vertex_model.py
      └─ predict.py
```

## Engineering Highlights

- Designed a modular multi-agent architecture for enterprise healthcare analytics.
- Implemented a RAG-first workflow to ground SQL and reporting in enterprise context.
- Built a scalable BigQuery embedding pipeline for 50M clinical records.
- Used deterministic sharding to distribute embedding workloads across 500 workers.
- Added Clinical Transformer embedding logic with 256-dimensional output vectors.
- Integrated BigQuery for analytical execution and embedding storage.
- Integrated pgvector as the semantic retrieval layer.
- Added SQL validation guardrails before query execution.
- Included MLOps scripts for model training, tuning, serialization, and Vertex AI registration.
- Structured the project for cloud-native API deployment with FastAPI and Docker.

## AI-Assisted Engineering Workflow

As the AI Architect / Engineer, I used Claude as an AI coding assistant to accelerate Python code generation, boilerplate creation, and implementation iteration. The architecture, component design, scaling strategy, cloud patterns, validation approach, and final integration decisions are owned and reviewed from an engineering perspective.

This reflects a modern AI engineering workflow where generative AI is used to improve development velocity while the architect remains responsible for:

- System architecture and component boundaries
- Cloud scaling strategy
- BigQuery and pgvector integration design
- RAG and agent orchestration patterns
- Code review and validation
- MLOps lifecycle design
- Production-readiness decisions

## Technology Stack

- Python
- FastAPI
- Claude for AI-assisted Python code generation
- OpenAI API
- BigQuery
- PostgreSQL
- pgvector
- PyTorch
- Hugging Face Transformers
- Flash Attention-compatible transformer loading
- XGBoost
- scikit-learn
- Vertex AI
- Docker

## Future Enhancements

- Add CI/CD workflow for tests, linting, Docker build, and deployment.
- Add automated evaluation for SQL correctness and answer quality.
- Add prompt and retrieval evaluation datasets.
- Add policy-based access controls for restricted healthcare data.
- Add observability for latency, token usage, BigQuery cost, and retrieval quality.
- Add model drift monitoring and scheduled retraining.
- Add Terraform infrastructure for repeatable GCP deployment.
- Add batch orchestration using Cloud Batch, Vertex AI Pipelines, or GKE jobs.

## Summary

This repository demonstrates an AI Architect / Engineer implementation of an enterprise healthcare AI data agent. It combines multi-agent orchestration, RAG, clinical transformer embeddings, BigQuery-scale data processing, pgvector semantic retrieval, and an MLOps model lifecycle into a single extensible platform.
