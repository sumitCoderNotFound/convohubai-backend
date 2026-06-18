"""Recruitment - Candidate, Application, ApplicationHistory (Feature 2)."""
from datetime import datetime
from sqlalchemy import Column, String, Text, JSON, ForeignKey, Boolean, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class Candidate(BaseModel):
    """Tenant candidate identity, consent, source and profile."""
    __tablename__ = "candidates"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)

    full_name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True, index=True)
    phone = Column(String(50), nullable=True)
    language = Column(String(10), default="en", nullable=False)
    source = Column(String(50), default="manual", nullable=False)

    # Consent (lifecycle hooks; recording/consent defaults pending product decision D3)
    consent_given = Column(Boolean, default=False, nullable=False)
    consent_version = Column(String(50), nullable=True)
    consent_at = Column(DateTime, nullable=True)

    tags = Column(JSON, default=list)
    notes = Column(Text, nullable=True)
    cv_url = Column(String(500), nullable=True)

    applications = relationship("Application", back_populates="candidate", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Candidate {self.email or self.full_name}>"


class Application(BaseModel):
    """Job-candidate relationship, pipeline stage and history."""
    __tablename__ = "applications"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    job_position_id = Column(UUID(as_uuid=True), ForeignKey("job_positions.id", ondelete="SET NULL"), nullable=True, index=True)

    stage = Column(String(50), default="applied", nullable=False)
    status = Column(String(50), default="active", nullable=False)
    assigned_reviewer_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    internal_notes = Column(Text, nullable=True)

    candidate = relationship("Candidate", back_populates="applications")
    job = relationship("JobPosition", back_populates="applications")
    history = relationship("ApplicationHistory", back_populates="application", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Application {self.id} stage={self.stage}>"


class ApplicationHistory(BaseModel):
    """Timestamped, attributable stage transitions (FR-APP-001 / decision trail)."""
    __tablename__ = "application_history"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    application_id = Column(UUID(as_uuid=True), ForeignKey("applications.id", ondelete="CASCADE"), nullable=False, index=True)

    from_stage = Column(String(50), nullable=True)
    to_stage = Column(String(50), nullable=False)
    actor_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reason = Column(Text, nullable=True)

    application = relationship("Application", back_populates="history")
