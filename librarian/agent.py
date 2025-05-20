"""
Librarian Agent construction using OpenAI Agents SDK.
"""

from agents import Agent, Runner
from .utils import health_check
from .search import text_search, semantic_search
from .io import read_document
from .ingest import ingest_document
from .schema import AgentOutput
from .config import settings

librarian: Agent = Agent(
    name="Librarian",
    handoffs=[],
    instructions="""
        You are the Librarian. Given a query:
        1. Decide between keywords or semantic retrieval.
        2. Call text_search or semantic_search.
        3. Aggregate under headings: Summary, Results, Next Steps.
        4. Use numbered citations matching metadata (filename, page).
    """,
    tools=[text_search, semantic_search, read_document, ingest_document, health_check],
    output_type=AgentOutput,
    model=settings.AGENT_MODEL
)


# Example runner invocation
# result = Runner.run_sync(librarian, user_query)
# print(result.final_output)
