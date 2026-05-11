import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

PG_HOST=os.getenv("PG_HOST")
PG_DB=os.getenv("PG_DB")
PG_USER=os.getenv("PG_USER")
PG_PASSWORD=os.getenv("PG_PASSWORD")

BQ_PROJECT=os.getenv("BQ_PROJECT")

# BigQuery embedding pipeline configuration
BQ_SOURCE_TABLE = os.getenv("BQ_SOURCE_TABLE")
BQ_DEST_TABLE = os.getenv("BQ_DEST_TABLE")
BQ_ID_COLUMN = os.getenv("BQ_ID_COLUMN", "id")
BQ_TEXT_COLUMNS = [
    col.strip()
    for col in os.getenv("BQ_TEXT_COLUMNS", "clinical_text").split(",")
    if col.strip()
]

# Distributed plan:
# 50M records / 100k records per n1-standard-16 instance = 500 instances.
EMBEDDING_TOTAL_RECORDS = int(os.getenv("EMBEDDING_TOTAL_RECORDS", "50000000"))
EMBEDDING_RECORDS_PER_INSTANCE = int(os.getenv("EMBEDDING_RECORDS_PER_INSTANCE", "100000"))
EMBEDDING_TOTAL_SHARDS = int(os.getenv("EMBEDDING_TOTAL_SHARDS", "500"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "2000"))
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "256"))
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "emilyalsentzer/Bio_ClinicalBERT",
)
EMBEDDING_USE_FLASH_ATTENTION = (
    os.getenv("EMBEDDING_USE_FLASH_ATTENTION", "true").lower() == "true"
)
