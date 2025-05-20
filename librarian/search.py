# Search tools will be migrated here 

import os
from typing import List, Dict, Optional, Union
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError, APITimeoutError
import logging
import dotenv
from agents import function_tool
from pymongo.errors import ConnectionFailure, OperationFailure, PyMongoError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .config import settings
from librarian.schema import ToolErrorOutput

dotenv.load_dotenv()

logger = logging.getLogger("librarian.search") # Changed logger name

client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=settings.DEFAULT_REQUEST_TIMEOUT)

# Retry decorator for OpenAI calls
openai_retry_decorator = retry(
    wait=wait_exponential(multiplier=1, min=1, max=settings.DEFAULT_REQUEST_TIMEOUT // 2),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((APIConnectionError, RateLimitError, APIStatusError, APITimeoutError))
)

# Retry decorator for MongoDB calls
mongodb_retry_decorator = retry(
    wait=wait_exponential(multiplier=1, min=1, max=settings.DEFAULT_REQUEST_TIMEOUT // 2),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((ConnectionFailure, OperationFailure)) # Retry on OperationFailure for some transient issues
)

@function_tool
def text_search(query: str, max_results: Optional[int]) -> Union[List[Dict], ToolErrorOutput]:
    """Use MongoDB Atlas text search to find keyword matches. Returns ToolErrorOutput on failure."""
    effective_max_results = max_results if max_results is not None else settings.MAX_TEXT_SEARCH_RESULTS
    logger.info(f"text_search called with query='{query}' max_results={effective_max_results}")
    
    @mongodb_retry_decorator
    def _execute_text_search_with_retry():
        from pymongo import MongoClient 
        # Timeout for DB connection itself, request timeout handled by pymongo default or specific operation params
        client_db = MongoClient(settings.MONGODB_ATLAS_URI, serverSelectionTimeoutMS=settings.HEALTH_CHECK_MONGO_TIMEOUT_MS) 
        db = client_db[settings.MONGODB_DB_NAME]
        pipeline = [
            {"$search": {"text": {"query": query, "path": "text"}}},
            {"$limit": effective_max_results},
            {"$project": {"_id": 1, "text": 1, "metadata": 1}}
        ]
        # Consider adding maxTimeMS to aggregate if long queries are an issue
        return list(db.chunks.aggregate(pipeline))

    try:
        results = _execute_text_search_with_retry()
        logger.info(f"text_search returned {len(results)} results")
        return results
    except (ConnectionFailure, OperationFailure) as e: # More specific catch after retry
        logger.error(f"MongoDB permanent failure in text_search after retries: {e}", exc_info=True)
        return ToolErrorOutput(error_type="DATABASE_ERROR", message="MongoDB unavailable for text search after retries.", details=str(e))
    except PyMongoError as e: # Catch other PyMongo errors
        logger.error(f"MongoDB general error in text_search: {e}", exc_info=True)
        return ToolErrorOutput(error_type="DATABASE_ERROR", message="A MongoDB error occurred during text search.", details=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in text_search: {e}")
        return ToolErrorOutput(error_type="TEXT_SEARCH_ERROR", message="An unexpected error occurred during text search.", details=str(e))

@function_tool
def semantic_search(query: str, k: Optional[int]) -> Union[List[Dict], ToolErrorOutput]:
    """Embed query & use Atlas vectorSearch to find top-k chunks. Returns ToolErrorOutput on failure."""
    effective_k = k if k is not None else settings.DEFAULT_SEMANTIC_SEARCH_K
    logger.info(f"semantic_search called with query='{query}' k={effective_k}")

    @openai_retry_decorator
    def _get_embedding_with_retry():
        response = client.embeddings.create(
            model=settings.EMBEDDING_MODEL_SEARCH, input=query
        )
        if not response.data or not response.data[0].embedding:
            raise ValueError("OpenAI embedding response is empty or invalid.")
        return response.data[0].embedding
    
    @mongodb_retry_decorator
    def _execute_vector_search_with_retry(embedding_vector):
        from pymongo import MongoClient 
        client_db = MongoClient(settings.MONGODB_ATLAS_URI, serverSelectionTimeoutMS=settings.HEALTH_CHECK_MONGO_TIMEOUT_MS)
        db = client_db[settings.MONGODB_DB_NAME]
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index", 
                    "queryVector": embedding_vector,
                    "path": "embedding", 
                    "numCandidates": 100, # This could be configurable
                    "limit": effective_k
                }
            },
            {"$project": {"_id": 1, "text": 1, "metadata": 1}}
        ]
        return list(db.chunks.aggregate(pipeline))

    try:
        embedding = _get_embedding_with_retry()
        results = _execute_vector_search_with_retry(embedding_vector=embedding)
        logger.info(f"semantic_search returned {len(results)} results")
        return results
    except (APIConnectionError, RateLimitError, APIStatusError, APITimeoutError) as e: # Specific catch after retry
        logger.error(f"OpenAI API permanent error in semantic_search after retries: {e}", exc_info=True)
        return ToolErrorOutput(error_type="API_ERROR", message="OpenAI API error during query embedding after retries.", details=str(e))
    except ValueError as e: # Catch specific ValueError from _get_embedding_with_retry
        logger.error(f"ValueError (likely OpenAI response issue) in semantic_search: {e}", exc_info=True)
        return ToolErrorOutput(error_type="API_ERROR", message="Invalid response from OpenAI embedding API.", details=str(e))
    except (ConnectionFailure, OperationFailure) as e: # Specific catch after retry
        logger.error(f"MongoDB permanent failure in semantic_search after retries: {e}", exc_info=True)
        return ToolErrorOutput(error_type="DATABASE_ERROR", message="MongoDB unavailable for semantic search after retries.", details=str(e))
    except PyMongoError as e: # Catch other PyMongo errors
        logger.error(f"MongoDB general error in semantic_search: {e}", exc_info=True)
        return ToolErrorOutput(error_type="DATABASE_ERROR", message="A MongoDB error occurred during semantic search.", details=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in semantic_search: {e}")
        return ToolErrorOutput(error_type="SEMANTIC_SEARCH_ERROR", message="An unexpected error occurred during semantic search.", details=str(e)) 