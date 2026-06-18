"""Dashboard + shortlist schemas (Phase 4)."""
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict
from datetime import datetime
from uuid import UUID


class RecentSession(BaseModel):
    session_id: UUID
    candidate_name: Optional[str] = None
    job_title: Optional[str] = None
    overall_score: Optional[float] = None
    recommendation: Optional[str] = None
    completed_at: Optional[datetime] = None


class DashboardResponse(BaseModel):
    jobs_open: int
    candidates_total: int
    applications_total: int
    applications_by_stage: Dict[str, int]
    interviews_published: int
    sessions_total: int
    sessions_completed: int
    scored_count: int
    avg_score: Optional[float] = None
    needs_review_count: int
    recent: List[RecentSession] = []


class ShortlistItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    application_id: UUID
    candidate_id: Optional[UUID] = None
    candidate_name: Optional[str] = None
    candidate_email: Optional[str] = None
    session_id: Optional[UUID] = None
    overall_score: Optional[float] = None
    recommendation: Optional[str] = None
    needs_human_review: bool = False
    risk_level: Optional[str] = None
    quality_flag: Optional[str] = None
    stage: str
    completed_at: Optional[datetime] = None


class ShortlistResponse(BaseModel):
    job_id: UUID
    job_title: str
    items: List[ShortlistItem] = []
    not_interviewed: int = 0
