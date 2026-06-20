"""Candidate / Application schemas (Feature 2)."""
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import Optional, List, Any
from datetime import datetime
from uuid import UUID
from app.recruitment.models.enums import (
    CandidateSource, ApplicationStage, ApplicationStatus,
)


class CandidateCreate(BaseModel):
    full_name: Optional[str] = Field(None, max_length=255)
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    language: str = "en"
    source: CandidateSource = CandidateSource.MANUAL
    tags: List[str] = []
    notes: Optional[str] = None


class CandidateUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    language: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None


class CandidateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    workspace_id: UUID
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    language: str
    source: str
    consent_given: bool
    tags: Optional[List[Any]] = None
    notes: Optional[str] = None
    cv_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class CandidateListResponse(BaseModel):
    items: List[CandidateResponse]
    total: int
    page: int
    page_size: int


class ApplicationCreate(BaseModel):
    candidate_id: UUID
    job_position_id: Optional[UUID] = None
    stage: ApplicationStage = ApplicationStage.APPLIED


class ApplicationDecision(BaseModel):
    to_stage: ApplicationStage
    reason: Optional[str] = None


class ApplicationHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    from_stage: Optional[str] = None
    to_stage: str
    actor_user_id: Optional[UUID] = None
    reason: Optional[str] = None
    created_at: datetime


class ApplicationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    workspace_id: UUID
    candidate_id: UUID
    job_position_id: Optional[UUID] = None
    stage: str
    status: str
    assigned_reviewer_id: Optional[UUID] = None
    internal_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class ApplicationListResponse(BaseModel):
    items: List[ApplicationResponse]
    total: int
    page: int
    page_size: int


# ---------------- Bulk import (Phase 5) ----------------
class BulkImportRow(BaseModel):
    full_name: str
    email: EmailStr
    phone: Optional[str] = None


class BulkImportRequest(BaseModel):
    job_position_id: Optional[UUID] = None
    rows: List[BulkImportRow] = Field(default_factory=list)


class BulkImportResult(BaseModel):
    created: int = 0
    matched: int = 0
    applications_created: int = 0
    skipped: int = 0
    errors: List[str] = Field(default_factory=list)
    candidate_ids: List[UUID] = Field(default_factory=list)


# ---------------- Outcome notifications (Phase 11) ----------------
class NotifyRequest(BaseModel):
    kind: str = Field(..., description="selected | advance | rejected | reminder | completed | score_ready")
    base_url: Optional[str] = None


class NotifyResult(BaseModel):
    sent: bool
    message: str
