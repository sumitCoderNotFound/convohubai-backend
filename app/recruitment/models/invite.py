"""Phase 2 - InterviewInvite: a tokenised link to a published interview."""
from datetime import datetime
from sqlalchemy import Column, String, ForeignKey, DateTime, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class InterviewInvite(BaseModel):
    """A shareable invite to take a specific published interview version."""
    __tablename__ = "interview_invites"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    template_id = Column(UUID(as_uuid=True), ForeignKey("interview_templates.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id = Column(UUID(as_uuid=True), ForeignKey("interview_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    job_position_id = Column(UUID(as_uuid=True), ForeignKey("job_positions.id", ondelete="SET NULL"), nullable=True)
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    token = Column(String(64), unique=True, index=True, nullable=False)
    # Optional pre-fill / targeting
    email = Column(String(255), nullable=True)
    candidate_name = Column(String(255), nullable=True)

    status = Column(String(30), default="pending", nullable=False)
    max_attempts = Column(Integer, default=1, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    sent_at = Column(DateTime, nullable=True)

    # Set once the candidate registers
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="SET NULL"), nullable=True)
    application_id = Column(UUID(as_uuid=True), ForeignKey("applications.id", ondelete="SET NULL"), nullable=True)

    sessions = relationship("InterviewSession", back_populates="invite", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<InterviewInvite {self.token} {self.status}>"
