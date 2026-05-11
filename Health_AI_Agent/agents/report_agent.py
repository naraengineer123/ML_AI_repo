import logging
from openai import OpenAI
from app.config import OPENAI_API_KEY

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

logging.basicConfig(level=logging.INFO)


def build_prompt(question: str, insights: str) -> str:
    """
    Create a structured prompt for the LLM
    """

    prompt = f"""
You are a healthcare data analyst.

User Question:
{question}

Insights from data analysis:
{insights}

Generate a professional report including:

1. Summary of findings
2. Key insights
3. Business interpretation
4. Recommended actions
5. Any potential risks

Write the report clearly for business stakeholders.
"""

    return prompt


def generate_report(question: str, insights: str) -> str:
    """
    Generate final business report using LLM
    """

    try:

        prompt = build_prompt(question, insights)

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You generate professional business reports from analytics results."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2
        )

        report = response.choices[0].message.content

        return report

    except Exception as e:

        logging.error(f"Report generation failed: {e}")

        return "Unable to generate report."