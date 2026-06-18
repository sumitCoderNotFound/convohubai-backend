"""Score schemas (recruiter-facing; Phase 2)."""
from pydantic import BaseModel, ConfigDict
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
