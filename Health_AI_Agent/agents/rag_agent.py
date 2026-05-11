import logging
from typing import List, Dict
from openai import OpenAI
from db.pgvector_client import similarity_search
from app.config import OPENAI_API_KEY

# initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO)


def create_embedding(text: str) -> List[float]:
    """
    Generate embedding vector for a query using OpenAI embeddings.
    """

    try:

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        embedding = response.data[0].embedding

        return embedding

    except Exception as e:

        logging.error(f"Embedding generation failed: {e}")

        return []


def search_documents(query: str, top_k: int = 5) -> List[Dict]:
    """
    Perform similarity search in pgvector database.
    """

    try:

        embedding = create_embedding(query)

        if not embedding:
            return []

        results = similarity_search(
            embedding=embedding,
            top_k=top_k
        )

        return results

    except Exception as e:

        logging.error(f"Document search failed: {e}")

        return []


def format_context(docs: List[Dict]) -> str:
    """
    Combine retrieved documents into a context string
    that can be passed to LLM prompt.
    """

    if not docs:
        return ""

    context_parts = []

    for i, doc in enumerate(docs):

        content = doc.get("content", "")

        source = doc.get("source", "unknown")

        formatted = f"""
Document {i+1}
Source: {source}
Content:
{content}
"""

        context_parts.append(formatted)

    context = "\n".join(context_parts)

    return context


def retrieve_context(question: str, top_k: int = 5) -> str:
    """
    Main function called by orchestrator.

    Steps
    -----
    1. Embed user query
    2. Retrieve similar docs from pgvector
    3. Format documents as context
    """

    logging.info(f"RAG retrieving context for question: {question}")

    docs = search_documents(question, top_k)

    if not docs:

        logging.warning("No documents found")

        return ""

    context = format_context(docs)

    return context