"""Session schemas (recruiter-facing views; Phase 2)."""
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Any, Dict
from datetime import datetime
from uuid import UUID


class SessionAnswerResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    question_id: Optional[UUID] = None
    order_index: int
    question_text_snapshot: Optional[str] = None
    transcript_text: Optional[str] = None
    duration_seconds: Optional[float] = None
    is_follow_up: bool
    created_at: datetime


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    workspace_id: UUID
    invite_id: Optional[UUID] = None
    version_id: UUID
    candidate_id: Optional[UUID] = None
    application_id: Optional[UUID] = None
    status: str
    language: str
    consent_given: bool
    consent_at: Optional[datetime] = None
    recording_enabled: bool
    current_question_index: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    transcript: Optional[List[Any]] = None
    risk_signals: Optional[Dict[str, Any]] = None
    created_at: datetime


class SessionDetailResponse(SessionResponse):
    answers: List[SessionAnswerResponse] = []
