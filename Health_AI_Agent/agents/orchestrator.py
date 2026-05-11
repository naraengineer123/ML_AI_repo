from agents.rag_agent import retrieve_context
from agents.sql_agent import generate_sql
from agents.qa_agent import validate_sql
from agents.analyst_agent import analyze
from agents.report_agent import generate_report
from db.bigquery_client import run_query

def run_agents(question):

    context = retrieve_context(question)

    sql = generate_sql(question,context)

    validate_sql(sql)

    data = run_query(sql)

    insights = analyze(data)

    report = generate_report(question,insights)

    return {
        "sql":sql,
        "insights":insights,
        "report":report
    }
