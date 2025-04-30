import os
import pytest
from librarian.agent import librarian, Runner
import dotenv   

dotenv.load_dotenv()

MONGODB_ATLAS_URI = os.getenv("MONGODB_ATLAS_URI")
S3_BUCKET = os.getenv("S3_BUCKET")

def test_agent_keyword_query_real():
    user_query = "Find the PDF of Project X spec."
    result = Runner.run_sync(librarian, user_query)
    # Output is a Pydantic model (AgentOutput)
    output = result.final_output
    assert hasattr(output, 'summary')
    assert hasattr(output, 'results')
    assert hasattr(output, 'next_steps')

def test_agent_semantic_query_real():
    user_query = "Show me all design decisions about caching."
    result = Runner.run_sync(librarian, user_query)
    output = result.final_output
    assert hasattr(output, 'summary')
    assert hasattr(output, 'results')
    assert hasattr(output, 'next_steps')

def test_agent_handles_no_results_real():
    user_query = "ThisIsAGibberishQueryThatShouldReturnNothing"
    result = Runner.run_sync(librarian, user_query)
    output = result.final_output
    assert hasattr(output, 'summary')
    assert hasattr(output, 'results')
    assert isinstance(output.results, list)

def test_agent_document_ingestion_real():
    path = os.path.join(os.path.dirname(__file__), "sample_docs", "sample.md")
    from librarian.tools import ingest_document
    result = ingest_document(path)
    assert "Ingested" in result

def test_agent_response_format_real():
    user_query = "List all whitepapers on Topic Y."
    result = Runner.run_sync(librarian, user_query)
    output = result.final_output
    assert hasattr(output, 'summary')
    assert hasattr(output, 'results')
    assert hasattr(output, 'next_steps')
    if output.results:
        for r in output.results:
            assert hasattr(r, 'excerpt')
            assert hasattr(r, 'source')
