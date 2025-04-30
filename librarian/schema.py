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
