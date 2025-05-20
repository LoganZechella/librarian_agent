# Ingestion tools will be migrated here 

import os
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError, APITimeoutError
import logging
import dotenv
import tiktoken
from pymongo import MongoClient, InsertOne
from pymongo.errors import ConnectionFailure, OperationFailure, PyMongoError
import uuid
from agents import function_tool
from typing import List, Dict, Any, Union
from tiktoken.core import Encoding
from pymongo.database import Database
from pymongo.collection import Collection
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.embedding import Embedding
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from librarian.io import read_document
from .config import settings
from librarian.schema import ToolErrorOutput

dotenv.load_dotenv()

logger = logging.getLogger("librarian.ingest")

client = OpenAI(api_key=settings.OPENAI_API_KEY, timeout=settings.DEFAULT_REQUEST_TIMEOUT)

# Retry decorators (can be shared if moved to a common utils or kept per-module if specific)
openai_retry_decorator = retry(
    wait=wait_exponential(multiplier=1, min=1, max=settings.DEFAULT_REQUEST_TIMEOUT // 2),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((APIConnectionError, RateLimitError, APIStatusError, APITimeoutError))
)

mongodb_retry_decorator = retry(
    wait=wait_exponential(multiplier=1, min=1, max=settings.DEFAULT_REQUEST_TIMEOUT // 2),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((ConnectionFailure, OperationFailure))
)

@function_tool
def ingest_document(path: str) -> Union[str, ToolErrorOutput]:
    """Extract, chunk, embed, and upsert into MongoDB Atlas. Returns ToolErrorOutput on failure."""
    logger.info(f"ingest_document called with path='{path}'")
    try:
        # 1. Extract text (read_document now returns Union[str, ToolErrorOutput])
        extracted_content = read_document(path=path, start_page=1, end_page=None)
        if isinstance(extracted_content, ToolErrorOutput):
            logger.error(f"Failed to read document for ingestion: {path} - Error: {extracted_content.message}")
            return extracted_content # Propagate the error
        
        text: str = extracted_content
        if not text.strip():
            logger.warning(f"Document {path} is empty or contains no extractable text for ingestion.")
            return ToolErrorOutput(error_type="EMPTY_DOCUMENT", message=f"Document {path} is empty or yielded no text.")

        # 2. Chunking
        enc: Encoding = tiktoken.get_encoding("cl100k_base")
        tokens: List[int] = enc.encode(text)
        chunk_size: int = settings.CHUNK_SIZE
        overlap: int = settings.CHUNK_OVERLAP
        chunks_text_list: List[str] = []
        for i in range(0, len(tokens), chunk_size - overlap):
            chunk_tokens: List[int] = tokens[i:i+chunk_size]
            chunk_text_item: str = enc.decode(chunk_tokens)
            chunks_text_list.append(chunk_text_item)
        
        if not chunks_text_list:
            logger.warning(f"No chunks generated for document {path}. Text length: {len(text)}")
            return ToolErrorOutput(error_type="NO_CHUNKS_GENERATED", message=f"No text chunks were generated from {path}.")

        # 3. Embed and upsert
        @mongodb_retry_decorator
        def _connect_mongo_db(): # Helper to retry connection
            mongo_client = MongoClient(settings.MONGODB_ATLAS_URI, serverSelectionTimeoutMS=settings.HEALTH_CHECK_MONGO_TIMEOUT_MS)
            return mongo_client[settings.MONGODB_DB_NAME]
        
        db: Database = _connect_mongo_db()
        chunks_collection: Collection = db.chunks
        
        meta: Dict[str, Any] = {"source": path}
        operations = [] # For potential batching later, but will do one-by-one with retry for now

        for idx, chunk_text_item in enumerate(chunks_text_list):
            @openai_retry_decorator
            def _get_embedding_for_chunk():
                response_embed: CreateEmbeddingResponse = client.embeddings.create(
                    model=settings.EMBEDDING_MODEL_INGEST, input=chunk_text_item 
                )
                if not response_embed.data or not response_embed.data[0].embedding:
                    raise ValueError("OpenAI embedding response for chunk is empty or invalid.")
                return response_embed.data[0].embedding
            
            embedding_vector: List[float] = _get_embedding_for_chunk()
            chunk_id: str = str(uuid.uuid4())
            
            @mongodb_retry_decorator
            def _upsert_chunk():
                chunks_collection.update_one(
                    {"_id": chunk_id},
                    {"$set": {"text": chunk_text_item, "embedding": embedding_vector, "metadata": {**meta, "chunk": idx}}},
                    upsert=True
                )
            _upsert_chunk()

        logger.info(f"ingest_document successfully ingested {len(chunks_text_list)} chunks from {path}")
        return f"Ingested {len(chunks_text_list)} chunks from {path}."
    
    except (APIConnectionError, RateLimitError, APIStatusError, APITimeoutError) as e:
        logger.error(f"OpenAI API permanent error in ingest_document after retries for {path}: {e}", exc_info=True)
        return ToolErrorOutput(error_type="API_ERROR", message=f"OpenAI API error during ingestion for {path} after retries.", details=str(e))
    except ValueError as e: # Catch specific ValueError from OpenAI response check
        logger.error(f"ValueError (likely OpenAI response issue) in ingest_document for {path}: {e}", exc_info=True)
        return ToolErrorOutput(error_type="API_ERROR", message=f"Invalid response from OpenAI embedding API during ingestion for {path}.", details=str(e))
    except (ConnectionFailure, OperationFailure) as e: 
        logger.error(f"MongoDB permanent failure in ingest_document after retries for {path}: {e}", exc_info=True)
        return ToolErrorOutput(error_type="DATABASE_ERROR", message=f"MongoDB unavailable for ingestion for {path} after retries.", details=str(e))
    except PyMongoError as e: 
        logger.error(f"MongoDB general error in ingest_document for {path}: {e}", exc_info=True)
        return ToolErrorOutput(error_type="DATABASE_ERROR", message=f"A MongoDB error occurred during ingestion for {path}.", details=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error in ingest_document for {path}: {e}")
        return ToolErrorOutput(error_type="INGESTION_ERROR", message=f"An unexpected error occurred during document ingestion for {path}.", details=str(e)) 