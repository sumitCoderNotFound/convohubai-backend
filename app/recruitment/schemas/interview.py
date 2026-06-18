"""Interview template / version schemas (Feature 3)."""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from app.recruitment.models.enums import InterviewMode


class InterviewCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    job_position_id: Optional[UUID] = None
    mode: InterviewMode = InterviewMode.VOICE_ONLY
    language: str = "en"
    # Optional: clone an existing version's content into the new draft.
    clone_from_version_id: Optional[UUID] = None


class InterviewVersionUpdate(BaseModel):
    """Edits to a DRAFT version only."""
    mode: Optional[InterviewMode] = None
    language: Optional[str] = None
    introduction: Optional[str] = None
    instructions: Optional[str] = None
    expected_duration_minutes: Optional[int] = Field(None, ge=1, le=240)
    attempt_limit: Optional[int] = Field(None, ge=1, le=10)
    deadline_at: Optional[datetime] = None
    completion_rules: Optional[Dict[str, Any]] = None
    recording_enabled: Optional[bool] = None
    ai_identity_disclosure: Optional[str] = None


class VersionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    version_number: int
    status: str
    mode: str
    language: str
    is_immutable: bool
    published_at: Optional[datetime] = None
    created_at: datetime


class InterviewVersionResponse(VersionSummary):
    template_id: UUID
    introduction: Optional[str] = None
    instructions: Optional[str] = None
    expected_duration_minutes: Optional[int] = None
    attempt_limit: int
    deadline_at: Optional[datetime] = None
    completion_rules: Optional[Dict[str, Any]] = None
    recording_enabled: bool
    ai_identity_disclosure: Optional[str] = None


class InterviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    workspace_id: UUID
    job_position_id: Optional[UUID] = None
    name: str
    description: Optional[str] = None
    archived: bool
    latest_published_version_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    versions: List[VersionSummary] = []


class InterviewListResponse(BaseModel):
    items: List[InterviewResponse]
    total: int
    page: int
    page_size: int


class PublishResult(BaseModel):
    published: bool
    version_id: UUID
    version_number: int
    errors: List[str] = []


class GenerateRequest(BaseModel):
    """Generate draft questions + rubric from the linked job (or pasted context)."""
    context: Optional[str] = None
    num_questions: int = Field(5, ge=1, le=20)
