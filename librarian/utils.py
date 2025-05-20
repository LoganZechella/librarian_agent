# Utility tools will be migrated here 

import os
import logging
from openai import OpenAI
import dotenv
from agents import function_tool
from .config import settings

dotenv.load_dotenv()

logger = logging.getLogger("librarian.utils")

client = OpenAI(api_key=settings.OPENAI_API_KEY)

@function_tool
def health_check() -> dict:
    """Check connectivity to MongoDB, OpenAI, and S3."""
    status = {"mongodb": False, "openai": False, "s3": False, "details": {}}
    try:
        from pymongo import MongoClient
        client_db = MongoClient(settings.MONGODB_ATLAS_URI, serverSelectionTimeoutMS=settings.HEALTH_CHECK_MONGO_TIMEOUT_MS)
        db = client_db[settings.MONGODB_DB_NAME]
        db.list_collection_names()
        status["mongodb"] = True
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        status["details"]["mongodb"] = str(e)
    try:
        resp = client.embeddings.create(model=settings.EMBEDDING_MODEL_SEARCH, input="health check")
        if resp and resp.data and resp.data[0].embedding:
            status["openai"] = True
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")
        status["details"]["openai"] = str(e)
    try:
        import boto3
        s3 = boto3.client("s3")
        bucket = settings.S3_BUCKET_NAME or settings.HEALTH_CHECK_S3_BUCKET_FALLBACK
        s3.list_objects_v2(Bucket=bucket, MaxKeys=1)
        status["s3"] = True
    except Exception as e:
        logger.error(f"S3 health check failed: {e}")
        status["details"]["s3"] = str(e)
    return status 