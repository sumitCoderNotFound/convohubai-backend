"""ATS integration schemas (Phase 9)."""
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class AtsConnectionCreate(BaseModel):
    provider: str  # greenhouse | lever | workable
    name: Optional[str] = None
    api_key: Optional[str] = None
    subdomain: Optional[str] = None
    base_url: Optional[str] = None


class AtsConnectionUpdate(BaseModel):
    name: Optional[str] = None
    api_key: Optional[str] = None
    subdomain: Optional[str] = None
    base_url: Optional[str] = None
    enabled: Optional[bool] = None


class AtsConnectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    provider: str
    name: Optional[str] = None
    subdomain: Optional[str] = None
    base_url: Optional[str] = None
    enabled: bool
    has_key: bool = False
    last_sync_at: Optional[datetime] = None
    created_at: datetime


class TestResult(BaseModel):
    ok: bool
    message: str


class ImportRequest(BaseModel):
    job_position_id: Optional[UUID] = None  # for candidate import: attach to this job
    limit: int = Field(100, ge=1, le=500)


class ImportResult(BaseModel):
    created: int = 0
    matched: int = 0
    applications_created: int = 0
    errors: List[str] = Field(default_factory=list)
    total_seen: int = 0


class PushResult(BaseModel):
    ok: bool
    message: str
