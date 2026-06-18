"""Recruitment - InterviewTemplate + InterviewVersion (Feature 3, versioning/immutability)."""
from sqlalchemy import Column, String, Text, JSON, ForeignKey, Boolean, DateTime, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class InterviewTemplate(BaseModel):
    """Editable interview container. The version is the immutable unit."""
    __tablename__ = "interview_templates"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    job_position_id = Column(UUID(as_uuid=True), ForeignKey("job_positions.id", ondelete="SET NULL"), nullable=True, index=True)
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    archived = Column(Boolean, default=False, nullable=False)

    # Pointer to the most recent published version (no FK to avoid a cycle).
    latest_published_version_id = Column(UUID(as_uuid=True), nullable=True)

    versions = relationship(
        "InterviewVersion",
        back_populates="template",
        cascade="all, delete-orphan",
        order_by="InterviewVersion.version_number",
    )

    def __repr__(self):
        return f"<InterviewTemplate {self.name}>"


class InterviewVersion(BaseModel):
    """A draft or published, immutable-once-published interview definition."""
    __tablename__ = "interview_versions"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    template_id = Column(UUID(as_uuid=True), ForeignKey("interview_templates.id", ondelete="CASCADE"), nullable=False, index=True)

    version_number = Column(Integer, default=1, nullable=False)
    status = Column(String(50), default="draft", nullable=False)  # draft|published|archived

    mode = Column(String(50), default="voice_only", nullable=False)
    language = Column(String(10), default="en", nullable=False)
    introduction = Column(Text, nullable=True)
    instructions = Column(Text, nullable=True)
    expected_duration_minutes = Column(Integer, nullable=True)
    attempt_limit = Column(Integer, default=1, nullable=False)
    deadline_at = Column(DateTime, nullable=True)
    completion_rules = Column(JSON, nullable=True)

    # Recording disabled by default (safest pending product decision D3).
    recording_enabled = Column(Boolean, default=False, nullable=False)
    # The AI must always identify itself to the candidate (responsible-AI principle).
    ai_identity_disclosure = Column(
        Text,
        default="You are speaking with an AI interviewer, not a human. This session may be recorded as disclosed.",
        nullable=False,
    )

    published_at = Column(DateTime, nullable=True)
    published_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    is_immutable = Column(Boolean, default=False, nullable=False)

    template = relationship("InterviewTemplate", back_populates="versions")
    questions = relationship(
        "InterviewQuestion",
        back_populates="version",
        cascade="all, delete-orphan",
        order_by="InterviewQuestion.order_index",
    )
    criteria = relationship(
        "RubricCriterion",
        back_populates="version",
        cascade="all, delete-orphan",
        order_by="RubricCriterion.order_index",
    )

    def __repr__(self):
        return f"<InterviewVersion {self.template_id} v{self.version_number} {self.status}>"
