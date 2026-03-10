Multi-Agent AI System for Enterprise Data Query

AI-powered platform that enables natural language querying of 10,000+ enterprise tables using a multi-agent architecture, RAG pipelines, and SQL automation.

The system allows business users to ask questions such as:

Which members have diabetes risk?

Show members with high cardiovascular risk.

Members with hospitalization risk above threshold.

Members with chronic disease indicators.

The platform automatically translates these questions into SQL queries, policy-aware retrieval, analytics, and reports.


<img width="1677" height="598" alt="image" src="https://github.com/user-attachments/assets/c099b9e9-ea1a-4a1f-91c0-b7edad5b9af9" />




System Components
FastAPI Application Layer

The system exposes APIs using FastAPI to allow users or applications to interact with the AI platform.

Responsibilities:

Accept user questions

Send requests to agent orchestrator

Return structured responses

Provide REST endpoints




Multi-Agent Architecture

The system uses specialized AI agents working together to process queries.

1. SQL / BigQuery Agent

Responsible for:

  Translating natural language questions into SQL
  Discovering relevant tables among 10,000+ datasets
  Executing queries on the enterprise data warehouse
  Optimizing joins and filters

Example question:
Which members have diabetes risk?


2. RAG Policy Agent
Ensures compliance and governance.
  Responsibilities:
  Retrieve policy documents
  Validate query permissions
  Mask restricted healthcare data
  Provide contextual knowledge

The agent uses Retrieval Augmented Generation (RAG) to access policy and metadata knowledge.

3. QA Agent
Ensures correctness of generated responses.
Capabilities:
  Detect hallucinations
  Verify query results
  Cross-check policy compliance
  Validate SQL outputs

4. Analyst Agent
Transforms raw data into meaningful insights.

Responsibilities:
  Risk analysis
  Member segmentation
  Trend detection
  Health outcome analysis

Example output:

Diabetes Risk Summary
---------------------
  High Risk Members: 1,245
  Moderate Risk: 3,980
  Low Risk: 12,421


5. Report / Action Agent
Responsible for generating final responses and triggering actions.

Outputs:
  Summaries
  Reports
  Dashboards
  Alerts
  Care management triggers
