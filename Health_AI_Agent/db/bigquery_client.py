import logging
from typing import List, Dict
from google.cloud import bigquery
from app.config import BQ_PROJECT

logging.basicConfig(level=logging.INFO)

# Initialize BigQuery client
client = bigquery.Client(project=BQ_PROJECT)


def run_query(sql: str) -> List[Dict]:
    """
    Execute a SQL query in BigQuery and return results as a list of dictionaries.

    Args:
        sql (str): SQL query to execute

    Returns:
        List[Dict]: List of rows as dictionaries
    """

    try:
        logging.info(f"Running SQL on BigQuery:\n{sql}")

        query_job = client.query(sql)
        results = query_job.result()  # Wait for job to complete

        rows = [dict(row) for row in results]

        logging.info(f"Query returned {len(rows)} rows")

        return rows

    except Exception as e:
        logging.error(f"BigQuery execution failed: {e}")
        raise