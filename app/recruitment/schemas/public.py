"""Public (candidate-facing, token-authenticated) schemas (Phase 2)."""
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import Optional, List
from uuid import UUID


class PublicQuestion(BaseModel):
    id: UUID
    order_index: int
    question_type: str
    prompt_text: str


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
    expected_duration_minutes: Optional[int] = None
    already_completed: bool = False


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
