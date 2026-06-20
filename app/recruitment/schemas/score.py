"""Score schemas (recruiter-facing; Phase 2)."""
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


class CriterionScoreResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    criterion_id: Optional[UUID] = None
    criterion_name: Optional[str] = None
    weight: float
    raw_score: Optional[float] = None
    weighted_contribution: Optional[float] = None
    confidence: Optional[float] = None
    evidence: Optional[str] = None
    reasoning: Optional[str] = None


class ScoreResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    id: UUID
    session_id: UUID
    application_id: Optional[UUID] = None
    version_id: Optional[UUID] = None
    status: str
    overall_score: Optional[float] = None
    recommendation: Optional[str] = None
    summary: Optional[str] = None
    quality_flag: Optional[str] = None
    needs_human_review: bool
    risk_level: Optional[str] = None
    risk_signals: Optional[Dict[str, Any]] = None
    model_used: Optional[str] = None
    scored_at: Optional[datetime] = None
    error: Optional[str] = None
    criterion_scores: List[CriterionScoreResponse] = []


class ApplicationResultResponse(BaseModel):
    """Recruiter result view: score + transcript pointer."""
    application_id: UUID
    session_id: Optional[UUID] = None
    score: Optional[ScoreResponse] = None
    has_session: bool = False


# ---------------- Score review / override (Phase 13) ----------------
class ScoreReviewCreate(BaseModel):
    override_recommendation: Optional[str] = None  # strong|moderate|weak|insufficient
    override_score: Optional[float] = Field(None, ge=0, le=100)
    note: Optional[str] = None


class ScoreReviewResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    override_recommendation: Optional[str] = None
    override_score: Optional[float] = None
    note: Optional[str] = None
    reviewer_user_id: Optional[UUID] = None
    created_at: datetime


# ---------------- Speech analytics (Phase 14) ----------------
class SpeechAnalytics(BaseModel):
    words_per_minute: Optional[int] = None
    total_words: int = 0
    total_seconds: int = 0
    filler_count: int = 0
    filler_rate: float = 0.0
    pace_label: str = "unknown"
    sentiment_label: str = "neutral"
    sentiment_score: float = 0.0
    positive_hits: int = 0
    negative_hits: int = 0
