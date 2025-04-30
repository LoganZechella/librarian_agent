"""
Function tools for Librarian Agent: text search, semantic search, document reading, and ingestion.
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("librarian.tools")

# Placeholder for function_tool decorator import
# from agents import function_tool

client = OpenAI()

# Health check tool
def health_check() -> dict:
    """Check connectivity to MongoDB, OpenAI, and S3."""
    status = {"mongodb": False, "openai": False, "s3": False, "details": {}}
    try:
        from pymongo import MongoClient
        client_db = MongoClient(os.getenv("MONGODB_ATLAS_URI"), serverSelectionTimeoutMS=3000)
        db = client_db[os.getenv("MONGODB_DB", "librarian_kb")]
        db.list_collection_names()
        status["mongodb"] = True
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        status["details"]["mongodb"] = str(e)
    try:
        resp = client.embeddings.create(model="text-embedding-3-small", input="health check")
        if resp and resp.data and resp.data[0].embedding:
            status["openai"] = True
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")
        status["details"]["openai"] = str(e)
    try:
        import boto3
        s3 = boto3.client("s3")
        bucket = os.getenv("S3_BUCKET") or "librarian-agent-bucket"
        s3.list_objects_v2(Bucket=bucket, MaxKeys=1)
        status["s3"] = True
    except Exception as e:
        logger.error(f"S3 health check failed: {e}")
        status["details"]["s3"] = str(e)
    return status

def text_search(query: str, max_results: int = 5) -> List[Dict]:
    """Use MongoDB Atlas text search to find keyword matches."""
    logger.info(f"text_search called with query='{query}' max_results={max_results}")
    try:
        from pymongo import MongoClient
        client_db = MongoClient(os.getenv("MONGODB_ATLAS_URI"))
        db = client_db[os.getenv("MONGODB_DB", "librarian_kb")]
        pipeline = [
            {"$search": {"text": {"query": query, "path": "text"}}},
            {"$limit": max_results},
            {"$project": {"_id": 1, "text": 1, "metadata": 1}}
        ]
        results = list(db.chunks.aggregate(pipeline))
        logger.info(f"text_search returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"text_search error: {e}")
        return [{"error": str(e)}]

def semantic_search(query: str, k: int = 5) -> List[Dict]:
    """Embed query & use Atlas vectorSearch to find top-k chunks."""
    logger.info(f"semantic_search called with query='{query}' k={k}")
    try:
        from pymongo import MongoClient
        response = client.embeddings.create(
            model="text-embedding-3-small", input=query
        )
        embedding = response.data[0].embedding
        client_db = MongoClient(os.getenv("MONGODB_ATLAS_URI"))
        db = client_db[os.getenv("MONGODB_DB", "librarian_kb")]
        pipeline = [
            {
                "$searchBeta": {
                    "vector": {
                        "embedding": embedding,
                        "path": "embedding",
                        "k": k
                    }
                }
            },
            {"$limit": k},
            {"$project": {"_id": 1, "text": 1, "metadata": 1}}
        ]
        results = list(db.chunks.aggregate(pipeline))
        logger.info(f"semantic_search returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"semantic_search error: {e}")
        return [{"error": str(e)}]

def read_document(path: str, start_page: int = 1, end_page: Optional[int] = None) -> str:
    """Load raw text from a stored document on disk or S3. Supports PDF, Word, Markdown, and S3."""
    logger.info(f"read_document called with path='{path}' start_page={start_page} end_page={end_page}")
    import mimetypes
    import io
    text = ""
    try:
        # S3 support
        if path.startswith("s3://"):
            import boto3
            s3 = boto3.client("s3")
            bucket, key = path[5:].split("/", 1)
            obj = s3.get_object(Bucket=bucket, Key=key)
            file_stream = io.BytesIO(obj["Body"].read())
            mime, _ = mimetypes.guess_type(key)
        else:
            file_stream = open(path, "rb")
            mime, _ = mimetypes.guess_type(path)
        # PDF
        if (mime and "pdf" in mime) or path.lower().endswith(".pdf"):
            from PyPDF2 import PdfReader
            reader = PdfReader(file_stream)
            start = max(0, start_page - 1)
            end = end_page if end_page is not None else len(reader.pages)
            for i in range(start, min(end, len(reader.pages))):
                text += reader.pages[i].extract_text() or ""
        # Word
        elif (mime and "word" in mime) or path.lower().endswith(".docx"):
            from docx import Document
            doc = Document(file_stream)
            text = "\n".join([p.text for p in doc.paragraphs])
        # Markdown or plain text
        elif (mime and ("markdown" in mime or "text" in mime)) or path.lower().endswith((".md", ".txt")):
            text = file_stream.read().decode("utf-8")
        else:
            raise ValueError(f"Unsupported file type: {path}")
        if not path.startswith("s3://"):
            file_stream.close()
        logger.info(f"read_document loaded {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"read_document error: {e}")
        return f"Error: {e}"

def ingest_document(path: str) -> str:
    """Extract, chunk, embed, and upsert into MongoDB Atlas."""
    logger.info(f"ingest_document called with path='{path}'")
    import tiktoken
    from pymongo import MongoClient
    import uuid
    try:
        # 1. Extract text
        text = read_document(path)
        # 2. Chunking (500 tokens, 20% overlap)
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunk_size = 500
        overlap = int(chunk_size * 0.2)
        chunks = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens = tokens[i:i+chunk_size]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
        # 3. Embed and upsert
        client_db = MongoClient(os.getenv("MONGODB_ATLAS_URI"))
        db = client_db[os.getenv("MONGODB_DB", "librarian_kb")]
        meta = {"source": path}
        for idx, chunk_text in enumerate(chunks):
            response = client.embeddings.create(
                model="text-embedding-3-small", input=chunk_text
            )
            embedding = response.data[0].embedding
            chunk_id = str(uuid.uuid4())
            db.chunks.update_one(
                {"_id": chunk_id},
                {"$set": {"text": chunk_text, "embedding": embedding, "metadata": {**meta, "chunk": idx}}},
                upsert=True
            )
        logger.info(f"ingest_document ingested {len(chunks)} chunks from {path}")
        return f"Ingested {len(chunks)} chunks from {path}."
    except Exception as e:
        logger.error(f"ingest_document error: {e}")
        return f"Error: {e}"
