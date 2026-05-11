import argparse
import logging
import re
from datetime import datetime, timezone
from typing import Dict, Iterable, List

from google.cloud import bigquery

from app.config import (
    BQ_DEST_TABLE,
    BQ_ID_COLUMN,
    BQ_SOURCE_TABLE,
    BQ_TEXT_COLUMNS,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_RECORDS_PER_INSTANCE,
    EMBEDDING_TOTAL_SHARDS,
)
from ingestion.chunking import shard_filter_sql
from ingestion.embed_store import ClinicalTransformerEmbedder

logging.basicConfig(level=logging.INFO)

SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SAFE_TABLE = re.compile(r"^[A-Za-z0-9_-]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+$")


def validate_table_name(table_name: str) -> None:
    if not table_name or not SAFE_TABLE.match(table_name):
        raise ValueError(
            "BigQuery table names must be fully qualified as "
            "project.dataset.table"
        )


def quote_column(column_name: str) -> str:
    if not SAFE_IDENTIFIER.match(column_name):
        raise ValueError(f"Unsafe BigQuery column name: {column_name}")
    return f"`{column_name}`"


def build_source_query(
    source_table: str,
    id_column: str,
    text_columns: List[str],
    shard_index: int,
    total_shards: int,
    records_per_instance: int,
) -> str:
    """
    Build the source query for one distributed embedding worker.
    """

    validate_table_name(source_table)

    if not text_columns:
        raise ValueError("At least one text column is required")

    id_expr = quote_column(id_column)
    text_expr = ", '\\n', ".join(
        f"IFNULL(CAST({quote_column(column)} AS STRING), '')"
        for column in text_columns
    )
    shard_filter = shard_filter_sql(id_column, shard_index, total_shards)

    return f"""
SELECT
  CAST({id_expr} AS STRING) AS source_id,
  TRIM(CONCAT({text_expr})) AS clinical_text
FROM `{source_table}`
WHERE {shard_filter}
LIMIT {records_per_instance}
"""


def iter_source_batches(
    client: bigquery.Client,
    query: str,
    batch_size: int,
) -> Iterable[List[Dict[str, str]]]:
    """
    Stream BigQuery records into fixed-size batches.
    """

    rows = client.query(query).result(page_size=batch_size)
    batch: List[Dict[str, str]] = []

    for row in rows:
        batch.append(
            {
                "source_id": row["source_id"],
                "clinical_text": row["clinical_text"],
            }
        )

        if len(batch) == batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def load_embeddings(
    client: bigquery.Client,
    destination_table: str,
    records: List[Dict],
) -> None:
    """
    Append embedded records to the destination BigQuery table.
    """

    validate_table_name(destination_table)

    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("source_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField(
                "embedding",
                "FLOAT64",
                mode="REPEATED",
                description=f"{EMBEDDING_DIMENSION}-dimensional clinical embedding",
            ),
            bigquery.SchemaField("embedding_dim", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("model_name", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("shard_index", "INT64", mode="REQUIRED"),
            bigquery.SchemaField("embedded_at", "TIMESTAMP", mode="REQUIRED"),
        ],
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
    )

    load_job = client.load_table_from_json(
        records,
        destination_table,
        job_config=job_config,
    )
    load_job.result()


def run_embedding_worker(
    shard_index: int,
    source_table: str = BQ_SOURCE_TABLE,
    destination_table: str = BQ_DEST_TABLE,
    id_column: str = BQ_ID_COLUMN,
    text_columns: List[str] = BQ_TEXT_COLUMNS,
    total_shards: int = EMBEDDING_TOTAL_SHARDS,
    records_per_instance: int = EMBEDDING_RECORDS_PER_INSTANCE,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> int:
    """
    Run one n1-standard-16 worker shard.

    For the requested 50M-record job, launch shard_index 0..499. Each worker
    reads up to 100k records and writes 256-dimensional embeddings to BigQuery
    in 2,000-record batches.
    """

    client = bigquery.Client()
    embedder = ClinicalTransformerEmbedder(model_name=model_name)

    query = build_source_query(
        source_table=source_table,
        id_column=id_column,
        text_columns=text_columns,
        shard_index=shard_index,
        total_shards=total_shards,
        records_per_instance=records_per_instance,
    )

    logging.info("Starting embedding shard %s/%s", shard_index, total_shards)
    logging.info("Source table: %s", source_table)
    logging.info("Destination table: %s", destination_table)

    total_written = 0

    for batch_number, batch in enumerate(
        iter_source_batches(client, query, batch_size),
        start=1,
    ):
        texts = [record["clinical_text"] for record in batch]
        embeddings = embedder.encode(texts, batch_size=batch_size)
        embedded_at = datetime.now(timezone.utc).isoformat()

        rows = [
            {
                "source_id": record["source_id"],
                "embedding": embedding,
                "embedding_dim": EMBEDDING_DIMENSION,
                "model_name": model_name,
                "shard_index": shard_index,
                "embedded_at": embedded_at,
            }
            for record, embedding in zip(batch, embeddings)
        ]

        load_embeddings(client, destination_table, rows)

        total_written += len(rows)
        logging.info(
            "Shard %s batch %s wrote %s rows; total=%s",
            shard_index,
            batch_number,
            len(rows),
            total_written,
        )

    logging.info("Completed shard %s with %s rows", shard_index, total_written)

    return total_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 256-d clinical transformer embeddings from BigQuery."
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--total-shards", type=int, default=EMBEDDING_TOTAL_SHARDS)
    parser.add_argument(
        "--records-per-instance",
        type=int,
        default=EMBEDDING_RECORDS_PER_INSTANCE,
    )
    parser.add_argument("--batch-size", type=int, default=EMBEDDING_BATCH_SIZE)
    parser.add_argument("--source-table", default=BQ_SOURCE_TABLE)
    parser.add_argument("--destination-table", default=BQ_DEST_TABLE)
    parser.add_argument("--id-column", default=BQ_ID_COLUMN)
    parser.add_argument("--text-columns", default=",".join(BQ_TEXT_COLUMNS))
    parser.add_argument("--model-name", default=EMBEDDING_MODEL_NAME)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_embedding_worker(
        shard_index=args.shard_index,
        source_table=args.source_table,
        destination_table=args.destination_table,
        id_column=args.id_column,
        text_columns=[
            column.strip()
            for column in args.text_columns.split(",")
            if column.strip()
        ],
        total_shards=args.total_shards,
        records_per_instance=args.records_per_instance,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )
