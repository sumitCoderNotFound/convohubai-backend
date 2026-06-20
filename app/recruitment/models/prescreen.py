"""Phase 11 - pre-screening / knockout models."""
from sqlalchemy import Column, String, Integer, Boolean, Text, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class PreScreenQuestion(BaseModel):
    """An eligibility question asked before the interview. Optional knockout rule."""
    __tablename__ = "prescreen_questions"

    version_id = Column(UUID(as_uuid=True), ForeignKey("interview_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    order_index = Column(Integer, default=0, nullable=False)
    prompt = Column(Text, nullable=False)
    qtype = Column(String(20), default="yes_no", nullable=False)  # yes_no | single_select | number | text
    options = Column(JSON, default=list)        # for single_select
    knockout = Column(JSON, default=dict)        # {"op": "equals|not_equals|in|not_in|min|max", "value": ...}
    required = Column(Boolean, default=True, nullable=False)


class PreScreenResult(BaseModel):
    """Stored eligibility outcome for one candidate attempt, with optional recruiter override."""
    __tablename__ = "prescreen_results"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id = Column(UUID(as_uuid=True), ForeignKey("interview_versions.id", ondelete="SET NULL"), nullable=True)
    invite_id = Column(UUID(as_uuid=True), ForeignKey("interview_invites.id", ondelete="SET NULL"), nullable=True, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("interview_sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    application_id = Column(UUID(as_uuid=True), ForeignKey("applications.id", ondelete="SET NULL"), nullable=True, index=True)
    candidate_email = Column(String(255), nullable=True)
    answers = Column(JSON, default=list)         # [{question_id, prompt, value}]
    auto_eligible = Column(Boolean, default=True, nullable=False)
    eligible = Column(Boolean, default=True, nullable=False)
    overridden = Column(Boolean, default=False, nullable=False)
    override_note = Column(Text, nullable=True)
