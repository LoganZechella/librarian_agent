from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class AppSettings(BaseSettings):
    # Environment-loaded variables from .env file
    MONGODB_ATLAS_URI: str
    MONGODB_DB_NAME: str = "librarian_kb" # Default if not in .env
    S3_BUCKET_NAME: str # Existing name from .env, to be used as S3_BUCKET in code
    OPENAI_API_KEY: str

    # Application-specific configurations with defaults
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP_PERCENT: float = 0.2 # As a percentage for easier understanding
    EMBEDDING_MODEL_INGEST: str = "text-embedding-3-large"
    EMBEDDING_MODEL_SEARCH: str = "text-embedding-3-large"
    AGENT_MODEL: str = "o4-mini" # Current model from agent.py
    DEFAULT_REQUEST_TIMEOUT: int = 30 # seconds, for external API calls
    MAX_TEXT_SEARCH_RESULTS: int = 5
    DEFAULT_SEMANTIC_SEARCH_K: int = 5
    HEALTH_CHECK_MONGO_TIMEOUT_MS: int = 3000
    HEALTH_CHECK_S3_BUCKET_FALLBACK: str = "librarian-agent-bucket"

    # SUPPORTED_FILE_EXTENSIONS: List[str] = [".pdf", ".docx", ".md", ".txt"] # Not used directly by tools, logic is mimetypes

    # Pydantic settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        extra='ignore'  # Ignore extra fields from .env if any
    )

    @property
    def CHUNK_OVERLAP(self) -> int:
        return int(self.CHUNK_SIZE * self.CHUNK_OVERLAP_PERCENT)

settings = AppSettings() 