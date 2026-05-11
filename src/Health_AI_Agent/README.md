# Enterprise AI Data Agent

## AI Architect / Engineer Overview

I designed this module as an enterprise-grade AI data agent for healthcare care-management teams. The agent helps users ask natural language questions over large-scale healthcare data without manually writing SQL or navigating fragmented claims, clinical, lab, pharmacy, policy, and care-management datasets.

The primary use case is identifying members with similar health-related issues, including diabetes risk, chronic disease indicators, hospitalization risk, medication gaps, and other high-risk clinical patterns.

## Business Problem

Care managers need to identify at-risk members quickly, but member data is often distributed across many enterprise tables and documentation sources. Finding the right population for outreach usually requires SQL expertise, healthcare schema knowledge, clinical terminology understanding, and manual validation by analysts or data engineers.

This creates delays in care-management outreach, population health interventions, and operational decision-making.

## Solution

The Enterprise AI Data Agent converts natural language questions into governed, validated, and explainable data workflows.

Example questions:

- Find members similar to diabetic high-risk patients.
- Show members with elevated health risk.
- Which members have diabetes risk and should be prioritized for outreach?
- Identify members with multiple chronic conditions.
- Which members have hospitalization risk greater than 0.7?

## Complete AI Agent Architecture

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

## Multi-Agent Components

| Agent | Responsibility |
| --- | --- |
| Orchestrator Agent | Coordinates RAG, SQL, validation, BigQuery execution, analysis, and reporting. |
| RAG Agent | Retrieves schema, metadata, policy, and clinical context using embeddings and vector search. |
| SQL Agent | Converts natural language questions into BigQuery Standard SQL. |
| QA Agent | Blocks unsafe SQL and validates generated queries before execution. |
| Analyst Agent | Converts raw query results into concise care-management insights. |
| Report Agent | Generates business-facing summaries, risks, and recommended actions. |

## RAG and Vector Search Architecture

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

- BigQuery stores source data and final embedding outputs.
- pgvector supports semantic similarity search.
- Clinical Transformer embeddings represent healthcare text as vectors.
- Retrieved context grounds SQL generation and final reports.

## Semantic Chunking Strategy

The ingestion pipeline uses semantic chunking instead of splitting documents by a fixed number of words or tokens. Clinical notes are split when the topic changes, keeping related sentences together and preventing unrelated clinical concepts from being embedded into the same vector.

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

Main file:

```text
ingestion/chunking.py
```

## Clinical Transformer Embedding Pipeline

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

## Dynamic GCP Instance Scaling

The embedding architecture dynamically determines how many `n1-standard-16` GCP instances to spin up based on input data volume.

```text
required_instances = ceil(total_input_records / records_per_instance)
```

For the 50M-record workload:

```text
50,000,000 records / 100,000 records per instance = 500 instances
```

| Input records | Records per instance | Required instances |
| --- | ---: | ---: |
| 10M | 100,000 | 100 |
| 50M | 100,000 | 500 |
| 75M | 100,000 | 750 |

Each dynamically created worker receives a unique `shard_index`. The source table is split using a deterministic BigQuery `FARM_FINGERPRINT` shard filter, avoiding inefficient offset scans.

## Worker Configuration

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
python -m ingestion.ingest_docs --shard-index 0
```

## API Usage

Run the API locally:

```bash
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

## Module Structure

```text
Health_AI_Agent/
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

