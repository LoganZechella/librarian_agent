# I/O tools will be migrated here 

import os
from typing import Optional, Union, BinaryIO # Added Union, BinaryIO
import logging
import dotenv # Added import
import mimetypes # Added import
import io # Added import
from agents import function_tool # Added import
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type # Added tenacity
from botocore.exceptions import ClientError as BotoClientError # For S3 errors
from PyPDF2.errors import PdfReadError # For PDF errors
# from docx.opc.exceptions import PackageNotFoundError # Example, if specific docx error exists

from .config import settings # Assuming settings might be used later, though not directly now
from librarian.schema import ToolErrorOutput # Corrected import path for schema

dotenv.load_dotenv() # Added

logger = logging.getLogger("librarian.io") # Changed logger name

# Helper for S3 calls if needed, or apply tenacity directly
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=settings.DEFAULT_REQUEST_TIMEOUT // 3), # Max wait from config
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(BotoClientError)
)
def _get_s3_object_with_retry(s3_client, bucket: str, key: str) -> bytes:
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

@function_tool # Added decorator
def read_document(path: str, start_page: Optional[int], end_page: Optional[int]) -> Union[str, ToolErrorOutput]:
    """Load raw text from a stored document on disk or S3. Supports PDF, Word, Markdown, and S3.
    On error, returns a ToolErrorOutput object."""
    if start_page is None:
        start_page = 1 # Default to 1-indexed start page

    logger.info(f"read_document called with path='{path}' start_page={start_page} end_page={end_page}")
    text_content: str = ""
    file_stream: Optional[BinaryIO] = None # For ensuring file is closed

    try:
        if path.startswith("s3://"):
            import boto3
            s3 = boto3.client("s3")
            bucket_key = path[5:]
            if '/' not in bucket_key:
                err_msg = f"S3 path must be in the format s3://bucket/key, got: {path}"
                logger.error(err_msg)
                return ToolErrorOutput(error_type="INVALID_INPUT", message=err_msg)
            
            bucket, key = bucket_key.split("/", 1)
            if not key:
                err_msg = f"S3 key could not be parsed from path: {path}"
                logger.error(err_msg)
                return ToolErrorOutput(error_type="INVALID_INPUT", message=err_msg)

            try:
                s3_object_bytes = _get_s3_object_with_retry(s3_client=s3, bucket=bucket, key=key)
                file_stream_internal: io.BytesIO = io.BytesIO(s3_object_bytes)
                mime_type, _ = mimetypes.guess_type(key)
            except BotoClientError as e:
                logger.error(f"S3 client error for {path}: {e}")
                return ToolErrorOutput(error_type="S3_ERROR", message=f"Failed to retrieve from S3: {path}", details=str(e))

        else: # Local file path
            if not os.path.exists(path):
                logger.error(f"File not found: {path}")
                return ToolErrorOutput(error_type="FILE_NOT_FOUND", message=f"File not found at path: {path}")
            if not os.path.isfile(path):
                logger.error(f"Path is not a file: {path}")
                return ToolErrorOutput(error_type="INVALID_INPUT", message=f"Path is not a file: {path}")
            
            file_stream_internal = open(path, "rb")
            file_stream = file_stream_internal # Assign to outer scope for finally block
            mime_type, _ = mimetypes.guess_type(path)

        # Process based on MIME type or extension
        if (mime_type and "pdf" in mime_type) or path.lower().endswith(".pdf"):
            from PyPDF2 import PdfReader
            try:
                reader = PdfReader(file_stream_internal)
                actual_start_page = max(0, start_page - 1) # PyPDF2 is 0-indexed
                actual_end_page = end_page if end_page is not None else len(reader.pages)
                
                pages_to_read = range(actual_start_page, min(actual_end_page, len(reader.pages)))
                if not pages_to_read: # handles cases where start_page is out of bounds
                     logger.warning(f"Page range {start_page}-{end_page} resulted in no pages for PDF {path} with {len(reader.pages)} pages.")
                
                for i in pages_to_read:
                    page_text = reader.pages[i].extract_text()
                    if page_text:
                        text_content += page_text + "\\n" # Add newline between pages
            except PdfReadError as e:
                logger.error(f"PDF processing error for {path}: {e}")
                return ToolErrorOutput(error_type="PDF_PROCESSING_ERROR", message=f"Error reading PDF file: {path}", details=str(e))
        
        elif (mime_type and "word" in mime_type.lower()) or path.lower().endswith(".docx"):
            from docx import Document
            try:
                doc = Document(file_stream_internal)
                text_content = "\\n".join([p.text for p in doc.paragraphs])
            except Exception as e: # Catching general exception for docx, can be more specific if known
                logger.error(f"DOCX processing error for {path}: {e}")
                return ToolErrorOutput(error_type="DOCX_PROCESSING_ERROR", message=f"Error reading DOCX file: {path}", details=str(e))

        elif (mime_type and ("markdown" in mime_type.lower() or "text" in mime_type.lower())) or path.lower().endswith((".md", ".txt")):
            try:
                text_content = file_stream_internal.read().decode("utf-8")
            except UnicodeDecodeError as e:
                logger.error(f"Text decoding error for {path}: {e}")
                return ToolErrorOutput(error_type="FILE_DECODING_ERROR", message=f"Error decoding text file: {path}", details=str(e))
        else:
            logger.warning(f"Unsupported file type or extension for path: {path} (MIME: {mime_type})")
            return ToolErrorOutput(error_type="UNSUPPORTED_FILE_TYPE", message=f"Unsupported file type: {path}. MIME: {mime_type}")

        logger.info(f"read_document successfully loaded {len(text_content)} characters from '{path}'")
        return text_content.strip()

    except FileNotFoundError as e: # Should be caught by os.path.exists for local files
        logger.error(f"File not found (outer catch) for {path}: {e}")
        return ToolErrorOutput(error_type="FILE_NOT_FOUND", message=str(e))
    except ValueError as e: # Catch other ValueErrors (e.g. S3 path issues if not caught earlier)
        logger.error(f"ValueError in read_document for {path}: {e}")
        return ToolErrorOutput(error_type="INVALID_INPUT", message=str(e), details=e.__class__.__name__)
    except Exception as e:
        logger.exception(f"Unexpected error in read_document for {path}: {e}") # Use logger.exception for stack trace
        return ToolErrorOutput(error_type="DOCUMENT_READ_ERROR", message="An unexpected error occurred while reading the document.", details=str(e))
    finally:
        if file_stream and not file_stream.closed:
            file_stream.close()
        # For BytesIO from S3 or other in-memory streams, close is a no-op but good practice if it were a real file handle.
        # If file_stream_internal was BytesIO, it doesn't need explicit closing in the same way a file object does.
        if 'file_stream_internal' in locals() and hasattr(file_stream_internal, 'close') and not path.startswith("s3://"):
             if not file_stream_internal.closed: # type: ignore
                file_stream_internal.close() # type: ignore 