"""Recruitment - JobPosition model (Feature 1)."""
from sqlalchemy import Column, String, Text, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class JobPosition(BaseModel):
    """Role and competency source of truth."""
    __tablename__ = "job_positions"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    department = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    employment_type = Column(String(50), default="full_time", nullable=False)
    status = Column(String(50), default="draft", nullable=False)

    # Competency profile (drafted from a job description, recruiter-editable)
    # { "skills": [...], "responsibilities": [...], "experience": [...] }
    competency_profile = Column(JSON, nullable=True)

    # Criteria used by pre-screening and scoring (FR-JOB-004)
    required_criteria = Column(JSON, default=list)
    preferred_criteria = Column(JSON, default=list)
    disqualifying_criteria = Column(JSON, default=list)

    is_general_assessment = Column(Boolean, default=False, nullable=False)

    applications = relationship("Application", back_populates="job")

    def __repr__(self):
        return f"<JobPosition {self.title}>"
