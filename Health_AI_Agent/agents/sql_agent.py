import logging
import re
from openai import OpenAI
from app.config import OPENAI_API_KEY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO)


def build_prompt(question: str, context: str) -> str:
    """
    Build SQL generation prompt
    """

    prompt = f"""
You are a senior data engineer specializing in BigQuery SQL.

Your job is to convert a natural language question into a valid SQL query.

Rules:
- Use BigQuery Standard SQL
- Only generate SELECT queries
- Never generate DELETE, UPDATE, DROP, INSERT
- Avoid SELECT *
- Use clear table aliases
- Use LIMIT 100 if query could return large results

Context information (business rules / schema info):
{context}

User Question:
{question}

Return ONLY SQL query.
"""

    return prompt


def clean_sql(sql: str) -> str:
    """
    Remove markdown or formatting from LLM output
    """

    if not sql:
        return ""

    # remove ```sql ``` wrappers
    sql = re.sub(r"```sql", "", sql)
    sql = re.sub(r"```", "", sql)

    return sql.strip()


def validate_sql(sql: str):
    """
    Prevent dangerous SQL
    """

    forbidden = ["DELETE", "DROP", "UPDATE", "INSERT", "ALTER"]

    for keyword in forbidden:
        if keyword in sql.upper():
            raise Exception(f"Unsafe SQL detected: {keyword}")

    if "SELECT" not in sql.upper():
        raise Exception("Generated query is not a SELECT statement")


def generate_sql(question: str, context: str = "") -> str:
    """
    Main function used by orchestrator
    """

    try:

        prompt = build_prompt(question, context)

        logging.info("Generating SQL for question")

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You generate optimized BigQuery SQL queries."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1
        )

        sql = response.choices[0].message.content

        sql = clean_sql(sql)

        validate_sql(sql)

        logging.info(f"Generated SQL: {sql}")

        return sql

    except Exception as e:

        logging.error(f"SQL generation failed: {e}")

        raise