"""Phase 2 - InterviewSession + SessionAnswer: a candidate's interview attempt."""
from sqlalchemy import Column, String, Text, JSON, ForeignKey, DateTime, Integer, Boolean, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class InterviewSession(BaseModel):
    """One candidate attempt at a published interview version."""
    __tablename__ = "interview_sessions"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    invite_id = Column(UUID(as_uuid=True), ForeignKey("interview_invites.id", ondelete="CASCADE"), nullable=True, index=True)
    version_id = Column(UUID(as_uuid=True), ForeignKey("interview_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="SET NULL"), nullable=True, index=True)
    application_id = Column(UUID(as_uuid=True), ForeignKey("applications.id", ondelete="SET NULL"), nullable=True, index=True)

    # Candidate addresses the session by this opaque token (no login).
    session_token = Column(String(64), unique=True, index=True, nullable=False)

    status = Column(String(30), default="created", nullable=False)
    language = Column(String(10), default="en", nullable=False)

    # Consent + disclosure captured at start (jurisdiction-aware text stored as snapshot)
    consent_given = Column(Boolean, default=False, nullable=False)
    consent_version = Column(String(50), nullable=True)
    consent_text_snapshot = Column(Text, nullable=True)
    consent_at = Column(DateTime, nullable=True)
    ai_identity_disclosed = Column(Boolean, default=False, nullable=False)
    recording_enabled = Column(Boolean, default=False, nullable=False)

    current_question_index = Column(Integer, default=0, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Full turn-by-turn log [{role, text, question_id, ts}]
    transcript = Column(JSON, default=list)
    # Anti-cheating / integrity signals captured client-side or derived
    risk_signals = Column(JSON, default=dict)
    meta = Column(JSON, default=dict)

    invite = relationship("InterviewInvite", back_populates="sessions")
    answers = relationship(
        "SessionAnswer", back_populates="session", cascade="all, delete-orphan",
        order_by="SessionAnswer.order_index",
    )

    def __repr__(self):
        return f"<InterviewSession {self.session_token} {self.status}>"


class SessionAnswer(BaseModel):
    """A candidate's answer to one question within a session."""
    __tablename__ = "session_answers"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("interview_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    question_id = Column(UUID(as_uuid=True), ForeignKey("interview_questions.id", ondelete="SET NULL"), nullable=True)

    order_index = Column(Integer, default=0, nullable=False)
    question_text_snapshot = Column(Text, nullable=True)
    transcript_text = Column(Text, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    audio_url = Column(String(500), nullable=True)
    is_follow_up = Column(Boolean, default=False, nullable=False)

    session = relationship("InterviewSession", back_populates="answers")
