from pydantic import BaseModel
from typing import List, Optional

class ResultItem(BaseModel):
    citation_id: int
    excerpt: str
    source: str
    page: Optional[int] = None

class AgentOutput(BaseModel):
    summary: str
    results: List[ResultItem]
    next_steps: List[str]

# New Pydantic model for structured error responses from tools
class ToolErrorOutput(BaseModel):
    error_type: str  # e.g., "API_ERROR", "FILE_NOT_FOUND", "PROCESSING_ERROR", "DATABASE_ERROR"
    message: str
    details: Optional[str] = None
