# Enterprise AI Data Agent

This repository is organized into two architecture sections:

1. **Enterprise AI Data Agent**
2. **MLOps Architecture**

The project demonstrates an AI Architect / Engineer implementation of a healthcare-focused enterprise data agent. It combines multi-agent orchestration, Retrieval Augmented Generation (RAG), semantic chunking, transformer-based clinical embeddings, BigQuery-scale data processing, pgvector semantic retrieval, and an MLOps model lifecycle.

## 1. Enterprise AI Data Agent

### AI Architect / Engineer Overview

I designed this platform as an enterprise-grade AI data agent for healthcare care-management teams. The agent helps care managers, analysts, and business users ask natural language questions over large-scale healthcare data without manually writing SQL or searching across fragmented claims, clinical, lab, pharmacy, policy, and care-management datasets.

The agent is designed to identify members with similar health-related issues such as diabetes risk, chronic disease indicators, hospitalization risk, medication gaps, and other high-risk clinical patterns. The system converts those natural language questions into governed, validated, explainable data workflows.

Example questions:

- Find members similar to diabetic high-risk patients.
- Show members with elevated health risk.
- Which members have diabetes risk and should be prioritized for outreach?
- Identify members with multiple chronic conditions.
- Which members have hospitalization risk greater than 0.7?

### Business Problem

Care managers need to identify at-risk members quickly, but member data is often spread across many enterprise tables and documentation sources. Finding the right population for outreach usually requires:

- SQL expertise
- Knowledge of complex healthcare schemas
- Understanding of clinical terminology
- Awareness of governance and data-access policies
- Manual validation by analysts or data engineers

This creates delays in care-management outreach, population health interventions, and operational decision-making.

### Solution Summary

The Enterprise AI Data Agent provides:

- Natural language access to enterprise healthcare data
- RAG-based retrieval of metadata, clinical context, and governance rules
- SQL generation for BigQuery
- SQL safety validation before execution
- BigQuery query execution
- Result summarization and care-manager-facing reporting
- Large-scale clinical embedding generation for 50M records
- BigQuery embedding storage
- pgvector semantic retrieval
- Dynamic GCP worker scaling based on input data volume

### Complete AI Agent Architecture

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

### Multi-Agent Components

| Agent | Responsibility |
| --- | --- |
| Orchestrator Agent | Coordinates the full request lifecycle across RAG, SQL, QA, BigQuery, analysis, and reporting. |
| RAG Agent | Retrieves relevant schema, policy, metadata, and clinical context using embeddings and vector search. |
| SQL Agent | Converts natural language questions into BigQuery Standard SQL. |
| QA Agent | Blocks unsafe SQL and validates generated queries before execution. |
| Analyst Agent | Converts raw query results into concise insights for care-management use cases. |
| Report Agent | Generates business-facing summaries, recommendations, and risk interpretation. |

Main files:

```text
src/Health_AI_Agent/agents/orchestrator.py
src/Health_AI_Agent/agents/rag_agent.py
src/Health_AI_Agent/agents/sql_agent.py
src/Health_AI_Agent/agents/qa_agent.py
src/Health_AI_Agent/agents/analyst_agent.py
src/Health_AI_Agent/agents/report_agent.py
```

### RAG and Vector Search Architecture

The platform uses a multi-RAG architecture to ground SQL generation and report generation in enterprise-specific context.

```text
Enterprise Documents / Metadata / Clinical Text
        |
        v
Semantic Chunking
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
SQL Generation + Analysis + Report Output
```

RAG components:

- BigQuery stores enterprise source data and final embedding outputs.
- pgvector supports vector similarity search for policy, metadata, and clinical context.
- Clinical Transformer embeddings represent clinical text as semantic vectors.
- Retrieved context is injected into agent prompts to improve SQL accuracy and reporting quality.

### Semantic Chunking Strategy

The ingestion pipeline uses semantic chunking instead of splitting documents by a fixed number of words or tokens. Clinical notes are split when the topic changes, which keeps related sentences together and prevents unrelated concepts from being embedded into the same vector.

Example source note:

```text
Member called regarding diabetes medication refill.
Agent explained insulin dosage instructions.
Member asked about cardiology appointment next week.
```

Semantic chunks:

```text
Chunk 1:
Member called regarding diabetes medication refill.
Agent explained insulin dosage instructions.

Chunk 2:
Member asked about cardiology appointment next week.
```

The chunking logic embeds each sentence and compares adjacent sentence embeddings using cosine similarity. If similarity drops below the configured threshold, a new chunk is created.

```text
sentence_embeddings = embed(sentences)
current_chunk = [sentences[0]]

for i in range(1, len(sentences)):
    sim = cosine_similarity(sentence_embeddings[i - 1], sentence_embeddings[i])

    if sim < similarity_threshold:
        chunks.append(current_chunk)
        current_chunk = [sentences[i]]
    else:
        current_chunk.append(sentences[i])
```

This improves retrieval quality because diabetes, medication adherence, cardiology, hospitalization risk, and other clinical topics can be retrieved as focused passages instead of mixed-context chunks.

Main file:

```text
src/Health_AI_Agent/ingestion/chunking.py
```

### Clinical Transformer Embedding Pipeline

The embedding pipeline reads source records from BigQuery, applies semantic chunking, generates clinical embeddings, and loads final embeddings back into BigQuery.

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

Worker flow:

```text
Worker Shard
  -> Read records from BigQuery
  -> Apply semantic chunking by clinical topic
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

### Dynamic GCP Instance Scaling

The embedding architecture dynamically determines how many `n1-standard-16` GCP instances to spin up based on input data volume.

Scaling formula:

```text
required_instances = ceil(total_input_records / records_per_instance)
```

For the 50M-record workload:

```text
50,000,000 records / 100,000 records per instance = 500 instances
```

Examples:

| Input records | Records per instance | Required instances |
| --- | ---: | ---: |
| 10M | 100,000 | 100 |
| 50M | 100,000 | 500 |
| 75M | 100,000 | 750 |

Each dynamically created worker receives a unique `shard_index`. For the 50M-record workload, shards run from `0` to `499`. The source table is split using a deterministic BigQuery `FARM_FINGERPRINT` shard filter, avoiding inefficient offset-based scans.

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

Run one worker shard:

```bash
cd src/Health_AI_Agent
python -m ingestion.ingest_docs --shard-index 0
```

For full-scale execution, launch shard indexes `0` through `499` across the dynamically provisioned GCP workers.

### BigQuery Data Architecture

BigQuery is used for:

- Enterprise healthcare source tables
- Generated SQL execution
- Final embedding storage
- Large-scale analytical workloads
- Downstream reporting and dashboard integration

The AI agent uses BigQuery as the system of record for structured enterprise analytics while pgvector supports semantic context retrieval.

### pgvector Architecture

pgvector is used as the vector search layer for RAG context retrieval.

Use cases:

- Retrieve schema documentation
- Retrieve healthcare policy context
- Retrieve clinical terminology context
- Retrieve data governance rules
- Support semantic matching between user questions and enterprise knowledge

### Deployment Architecture

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

The service can be containerized with Docker and deployed to Cloud Run, GKE, or a VM-based service environment.

Main service files:

```text
src/Health_AI_Agent/app/main.py
src/Health_AI_Agent/app/config.py
src/Health_AI_Agent/Dockerfile
```

### API Usage

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

### Enterprise AI Data Agent Files

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
├─ ingestion/
│  ├─ ingest_docs.py
│  ├─ embed_store.py
│  └─ chunking.py
├─ Dockerfile
└─ requirements.txt
```

## 2. MLOps Architecture

### MLOps Overview

The repository includes an MLOps-ready machine learning pipeline for predictive modeling. This part of the system supports model development, tuning, serialization, registry integration, and future deployment for healthcare risk-scoring and classification use cases.

The MLOps architecture is designed to support the full model lifecycle:

- Data ingestion
- Data validation
- Feature engineering
- Model training
- Hyperparameter tuning
- Model evaluation
- Model serialization
- Vertex AI model registration
- Batch or online inference
- Monitoring and retraining

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

### Model Training Architecture

```text
Raw Data
  -> Feature Engineering
  -> XGBoost Training
  -> Grid Search Tuning
  -> Accuracy Evaluation
  -> Save Best Model
  -> Register Model in Vertex AI
```

The current training pipeline includes:

- CSV-based data ingestion
- Reusable preprocessing functions
- Derived feature creation
- Train/test split
- XGBoost model training
- Grid search hyperparameter tuning
- Accuracy evaluation
- `.pkl` model serialization
- Vertex AI model upload
- Local prediction script

### Model Components

| Component | File | Purpose |
| --- | --- | --- |
| Data preprocessing | `src/ML/data_preprocessing.py` | Loads source data and creates train/test splits. |
| Feature engineering | `src/ML/feature_engineering.py` | Creates derived features, validates dates, checks nulls, and handles categorical columns. |
| Training | `src/ML/train_model.py` | Trains the baseline XGBoost classifier. |
| Hyperparameter tuning | `src/ML/hyperparameter_tuning.py` | Runs grid search and saves the best model. |
| Model registration | `src/ML/register_vertex_model.py` | Uploads the trained model artifact to Vertex AI. |
| Prediction | `src/ML/predict.py` | Loads the saved model and runs local inference. |

### Vertex AI Registry Pattern

Vertex AI is used as the model registry layer for production MLOps.

Registry responsibilities:

- Store versioned model artifacts
- Track deployable models
- Support controlled promotion from experiment to production
- Enable downstream deployment to managed serving or batch inference
- Provide a foundation for model governance and auditability

### Production MLOps Design

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

### AI-Assisted Engineering Workflow

As the AI Architect / Engineer, I used Claude as an AI coding assistant to accelerate Python code generation, boilerplate creation, and implementation iteration. The architecture, component design, scaling strategy, cloud patterns, validation approach, and final integration decisions are owned and reviewed from an engineering perspective.

This reflects a modern AI engineering workflow where generative AI improves development velocity while the architect remains responsible for:

- System architecture and component boundaries
- Cloud scaling strategy
- BigQuery and pgvector integration design
- RAG and agent orchestration patterns
- Code review and validation
- MLOps lifecycle design
- Production-readiness decisions

### Technology Stack

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

### Repository Structure

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
   │  ├─ agents/
   │  ├─ db/
   │  └─ ingestion/
   └─ ML/
      ├─ data_preprocessing.py
      ├─ feature_engineering.py
      ├─ train_model.py
      ├─ hyperparameter_tuning.py
      ├─ register_vertex_model.py
      └─ predict.py
```

### Engineering Highlights

- Designed a modular multi-agent architecture for enterprise healthcare analytics.
- Implemented a RAG-first workflow to ground SQL and reporting in enterprise context.
- Added semantic chunking based on topic shifts and cosine similarity.
- Built a scalable BigQuery embedding pipeline for 50M clinical records.
- Designed dynamic `n1-standard-16` worker scaling based on input record volume.
- Used deterministic sharding to distribute embedding workloads across workers.
- Added Clinical Transformer embedding logic with 256-dimensional output vectors.
- Integrated BigQuery for analytical execution and embedding storage.
- Integrated pgvector as the semantic retrieval layer.
- Added SQL validation guardrails before query execution.
- Included MLOps scripts for model training, tuning, serialization, and Vertex AI registration.
- Structured the project for cloud-native API deployment with FastAPI and Docker.

### Future Enhancements

- Add CI/CD workflow for tests, linting, Docker build, and deployment.
- Add automated evaluation for SQL correctness and answer quality.
- Add prompt and retrieval evaluation datasets.
- Add policy-based access controls for restricted healthcare data.
- Add observability for latency, token usage, BigQuery cost, and retrieval quality.
- Add model drift monitoring and scheduled retraining.
- Add Terraform infrastructure for repeatable GCP deployment.
- Add batch orchestration using Cloud Batch, Vertex AI Pipelines, or GKE jobs.
