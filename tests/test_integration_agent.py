import os
import pytest

# Ensure the new OpenAI Python SDK is installed and API key is set
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("OPENAI_API_KEY must be set in the environment for these tests to run.")

from librarian.agent import librarian, Runner

def test_agent_keyword_query_real():
    user_query = "Find the PDF of Project X spec."
    result = Runner.run_sync(librarian, user_query)
    assert isinstance(result.final_output, dict)
    assert "summary" in result.final_output
    assert "results" in result.final_output
    assert "next_steps" in result.final_output

def test_agent_semantic_query_real():
    user_query = "Show me all design decisions about caching."
    result = Runner.run_sync(librarian, user_query)
    assert isinstance(result.final_output, dict)
    assert "summary" in result.final_output
    assert "results" in result.final_output
    assert "next_steps" in result.final_output

def test_agent_handles_no_results_real():
    user_query = "ThisIsAGibberishQueryThatShouldReturnNothing"
    result = Runner.run_sync(librarian, user_query)
    assert isinstance(result.final_output, dict)
    assert "summary" in result.final_output
    assert "results" in result.final_output
    assert isinstance(result.final_output["results"], list)

def test_agent_document_ingestion_real():
    path = os.path.join(os.path.dirname(__file__), "sample_docs", "sample.md")
    from librarian.tools import ingest_document
    result = ingest_document(path)
    assert "Ingested" in result

def test_agent_response_format_real():
    user_query = "List all whitepapers on Topic Y."
    result = Runner.run_sync(librarian, user_query)
    output = result.final_output
    assert isinstance(output, dict)
    assert "summary" in output
    assert "results" in output
    assert "next_steps" in output
    if output["results"]:
        for r in output["results"]:
            assert "excerpt" in r
            assert "source" in r
