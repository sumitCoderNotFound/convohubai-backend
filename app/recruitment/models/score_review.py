"""Phase 13 - recruiter score override / human review (audit trail)."""
from sqlalchemy import Column, String, Text, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class ScoreReview(BaseModel):
    """One human review event for a score. Multiple rows = full audit trail."""
    __tablename__ = "score_reviews"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    score_id = Column(UUID(as_uuid=True), ForeignKey("interview_scores.id", ondelete="CASCADE"), nullable=True, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    application_id = Column(UUID(as_uuid=True), ForeignKey("applications.id", ondelete="SET NULL"), nullable=True, index=True)
    reviewer_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    override_recommendation = Column(String(20), nullable=True)  # strong|moderate|weak|insufficient
    override_score = Column(Float, nullable=True)                # optional 0-100
    note = Column(Text, nullable=True)
