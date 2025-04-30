"""
Librarian Agent construction using OpenAI Agents SDK.
"""

from agents import Agent, Runner, function_tool
from .tools import text_search, semantic_search, read_document, ingest_document

# Decorate tools (if not already decorated in tools.py)
text_search = function_tool(text_search)
semantic_search = function_tool(semantic_search)
read_document = function_tool(read_document)
ingest_document = function_tool(ingest_document)

librarian = Agent(
    name="Librarian",
    instructions="""
        You are the Librarian. Given a query:
        1. Decide between keywords or semantic retrieval.
        2. Call text_search or semantic_search.
        3. Aggregate under headings: Summary, Results, Next Steps.
        4. Use numbered citations matching metadata (filename, page).
    """,
    tools=[text_search, semantic_search, read_document, ingest_document],
    output_type=dict
)

# Example runner invocation
# result = Runner.run_sync(librarian, user_query)
# print(result.final_output)
