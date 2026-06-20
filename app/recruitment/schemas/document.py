"""Document schemas (Phase 12)."""
from pydantic import BaseModel, ConfigDict
from typing import Optional
from uuid import UUID
from datetime import datetime


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    candidate_id: UUID
    application_id: Optional[UUID] = None
    kind: str
    filename: str
    content_type: Optional[str] = None
    size: int
    source: str
    created_at: datetime
