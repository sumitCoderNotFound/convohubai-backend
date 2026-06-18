"""Phase 2 - InterviewScore + CriterionScore: explainable rubric scoring."""
from sqlalchemy import Column, String, Text, JSON, ForeignKey, DateTime, Float, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class InterviewScore(BaseModel):
    """Aggregate result for one completed session (one per session)."""
    __tablename__ = "interview_scores"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    application_id = Column(UUID(as_uuid=True), ForeignKey("applications.id", ondelete="SET NULL"), nullable=True, index=True)
    version_id = Column(UUID(as_uuid=True), ForeignKey("interview_versions.id", ondelete="SET NULL"), nullable=True)

    status = Column(String(20), default="pending", nullable=False)  # pending|scoring|completed|failed
    overall_score = Column(Float, nullable=True)  # 0-100, deterministic weighted total (computed in app code)
    recommendation = Column(String(20), nullable=True)  # strong|moderate|weak|insufficient
    summary = Column(Text, nullable=True)

    # Quality gate + integrity
    quality_flag = Column(String(40), nullable=True)  # e.g. 'low_confidence', 'needs_human_review'
    needs_human_review = Column(Boolean, default=False, nullable=False)
    risk_level = Column(String(20), nullable=True)  # low|medium|high
    risk_signals = Column(JSON, default=dict)

    model_used = Column(String(100), nullable=True)
    scored_at = Column(DateTime, nullable=True)
    error = Column(Text, nullable=True)

    criterion_scores = relationship(
        "CriterionScore", back_populates="score", cascade="all, delete-orphan",
        order_by="CriterionScore.order_index",
    )

    def __repr__(self):
        return f"<InterviewScore session={self.session_id} {self.overall_score}>"


class CriterionScore(BaseModel):
    """Per-criterion score with evidence and reasoning (explainability)."""
    __tablename__ = "criterion_scores"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    score_id = Column(UUID(as_uuid=True), ForeignKey("interview_scores.id", ondelete="CASCADE"), nullable=False, index=True)
    criterion_id = Column(UUID(as_uuid=True), ForeignKey("rubric_criteria.id", ondelete="SET NULL"), nullable=True)

    order_index = Column(Float, default=0, nullable=False)
    criterion_name = Column(String(255), nullable=True)  # snapshot
    weight = Column(Float, default=0, nullable=False)     # snapshot
    raw_score = Column(Float, nullable=True)              # 0-100 model judgement for this criterion
    weighted_contribution = Column(Float, nullable=True)  # raw_score * weight/100
    confidence = Column(Float, nullable=True)             # 0-1
    evidence = Column(Text, nullable=True)
    reasoning = Column(Text, nullable=True)

    score = relationship("InterviewScore", back_populates="criterion_scores")
