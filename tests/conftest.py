import os
import pytest
from pymongo import MongoClient

@pytest.fixture(scope="session", autouse=True)
def cleanup_test_chunks():
    """
    Cleanup test data from MongoDB after tests.
    Removes all chunks with metadata.source containing 'sample_docs/' or 'sample.'
    """
    yield  # Run tests first
    mongo_uri = os.getenv("MONGODB_ATLAS_URI")
    db_name = os.getenv("MONGODB_DB", "librarian_kb")
    if not mongo_uri:
        print("[CLEANUP] Skipping MongoDB cleanup: MONGODB_ATLAS_URI not set.")
        return
    client = MongoClient(mongo_uri)
    db = client[db_name]
    # Remove test chunks by source pattern
    result = db.chunks.delete_many({"metadata.source": {"$regex": r"sample_docs/|sample\\.(pdf|docx|md)$"}})
    print(f"[CLEANUP] Removed {result.deleted_count} test chunks from MongoDB.")
