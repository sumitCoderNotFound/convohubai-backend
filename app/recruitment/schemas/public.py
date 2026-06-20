"""Public (candidate-facing, token-authenticated) schemas (Phase 2)."""
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class PublicQuestion(BaseModel):
    id: UUID
    order_index: int
    question_type: str
    prompt_text: str
    response_type: str = "text"          # text | single_select | multi_select | number | rating | yes_no | info
    options: List[str] = Field(default_factory=list)
    scale: Optional[int] = None          # for rating


class PortalRole(BaseModel):
    template_id: UUID
    name: str
    description: Optional[str] = None
    job_title: Optional[str] = None
    location: Optional[str] = None
    department: Optional[str] = None
    mode: str = "voice_only"
    expected_duration_minutes: Optional[int] = None


class PortalView(BaseModel):
    workspace_name: str
    brand_name: Optional[str] = None
    brand_logo_url: Optional[str] = None
    roles: List[PortalRole] = Field(default_factory=list)


class PortalApplyRequest(BaseModel):
    full_name: str = Field(..., min_length=1)
    email: EmailStr


class PortalApplyResult(BaseModel):
    token: str


class StatusStep(BaseModel):
    label: str
    state: str  # done | current | upcoming


class PublicStatusView(BaseModel):
    candidate_name: Optional[str] = None
    interview_name: str
    brand_name: Optional[str] = None
    status: str          # not_started | in_progress | under_review | advanced | not_selected | ineligible | expired | closed
    headline: str
    message: str
    attempts_remaining: Optional[int] = None
    expires_at: Optional[datetime] = None
    can_resume: bool = False
    steps: List[StatusStep] = Field(default_factory=list)


class InvitePublicView(BaseModel):
    """What a candidate sees before starting (no internal data)."""
    token: str
    status: str
    interview_name: str
    introduction: Optional[str] = None
    instructions: Optional[str] = None
    ai_identity_disclosure: str
    consent_text: str
    mode: str
    language: str
    brand_name: Optional[str] = None
    brand_logo_url: Optional[str] = None
    interviewer_avatar_url: Optional[str] = None
    expected_duration_minutes: Optional[int] = None
    already_completed: bool = False
    attempts_remaining: Optional[int] = None
    expires_at: Optional[datetime] = None
    email_locked: bool = False


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    consent_given: bool
    phone: Optional[str] = None
    language: Optional[str] = None


class SessionStateResponse(BaseModel):
    session_token: str
    status: str
    current_question_index: int
    total_questions: int
    current_question: Optional[PublicQuestion] = None
    ai_identity_disclosure: Optional[str] = None
    finished: bool = False


class AnswerSubmit(BaseModel):
    question_id: UUID
    transcript_text: str = Field(..., min_length=1)
    duration_seconds: Optional[float] = None
    is_follow_up: bool = False
    # Optional integrity signals captured client-side
    risk_signals: Optional[dict] = None


class PublicResult(BaseModel):
    status: str
    message: str
    # Only populated when the workspace allows candidates to see scores
    overall_score: Optional[float] = None
    recommendation: Optional[str] = None
    summary: Optional[str] = None


class RiskSignalsUpdate(BaseModel):
    """Behavioural integrity signals captured client-side during the interview."""
    signals: dict = Field(default_factory=dict)
