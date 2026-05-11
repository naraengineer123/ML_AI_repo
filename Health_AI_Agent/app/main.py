from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Import orchestrator that runs all agents
from agents.orchestrator import run_agents

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AI Multi-Agent Data Query Platform",
    description="Query 10,000+ tables using LLM agents (SQL, RAG, QA, Analyst, Report)",
    version="1.0.0"
)

# Pydantic model for user query
class QueryRequest(BaseModel):
    question: str


@app.get("/")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "ok", "message": "AI Multi-Agent Data Query Platform running."}


@app.post("/query")
async def query(request: QueryRequest):
    """
    Main endpoint to ask questions
    """
    try:
        logging.info(f"Received query: {request.question}")

        # Run the orchestrator that coordinates all agents
        result = run_agents(request.question)

        return {
            "question": request.question,
            "sql_generated": result.get("sql"),
            "insights": result.get("insights"),
            "report": result.get("report")
        }

    except Exception as e:
        logging.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))