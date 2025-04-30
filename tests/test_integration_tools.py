import os
import pytest

# Ensure the new OpenAI Python SDK is installed and API key is set
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("OPENAI_API_KEY must be set in the environment for these tests to run.")

from librarian.tools import text_search, semantic_search, read_document, ingest_document

os.environ["MONGODB_DB"] = "librarian_kb"
S3_BUCKET = "librarian-agent-bucket"

def test_text_search_real_mongodb():
    results = text_search("design", max_results=3)
    assert isinstance(results, list)
    assert all("text" in r and "metadata" in r for r in results)
    assert len(results) > 0

def test_semantic_search_real_mongodb_openai():
    results = semantic_search("caching strategy", k=3)
    assert isinstance(results, list)
    assert all("text" in r and "metadata" in r for r in results)
    assert len(results) > 0

@pytest.mark.parametrize("filename", ["sample.pdf", "sample.docx", "sample.md"])
def test_read_document_local(filename):
    path = os.path.join(os.path.dirname(__file__), "sample_docs", filename)
    text = read_document(path)
    assert isinstance(text, str)
    assert len(text) > 0

@pytest.mark.parametrize("s3_key", [
    "Building a High.pdf",
    "Building a High.docx",
    "README.md"
])
def test_read_document_s3(s3_key):
    s3_path = f"s3://{S3_BUCKET}/{s3_key}"
    text = read_document(s3_path)
    assert isinstance(text, str)
    assert len(text) > 0

@pytest.mark.parametrize("filename", ["sample.pdf", "sample.docx", "sample.md"])
def test_ingest_document_local(filename):
    path = os.path.join(os.path.dirname(__file__), "sample_docs", filename)
    result = ingest_document(path)
    assert isinstance(result, str)
    assert "Ingested" in result

def test_read_document_unsupported_type_raises():
    path = os.path.join(os.path.dirname(__file__), "sample_docs", "unsupported.xyz")
    with open(path, "w") as f:
        f.write("dummy data")
    with pytest.raises(ValueError):
        read_document(path)
    os.remove(path)
