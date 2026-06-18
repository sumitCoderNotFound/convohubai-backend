"""Recruitment - RubricCriterion + ScoreAnchor (Feature 5)."""
from sqlalchemy import Column, String, Text, ForeignKey, Boolean, Integer, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class RubricCriterion(BaseModel):
    """A recruiter-defined scoring criterion (weight + evidence guidance)."""
    __tablename__ = "rubric_criteria"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id = Column(UUID(as_uuid=True), ForeignKey("interview_versions.id", ondelete="CASCADE"), nullable=False, index=True)

    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    weight = Column(Float, default=0.0, nullable=False)  # percentage; criteria sum to 100
    evidence_instructions = Column(Text, nullable=True)
    order_index = Column(Integer, default=0, nullable=False)

    # Blocks publication if the criterion looks like a protected/sensitive trait (FR-RUB-006).
    is_blocked_sensitive = Column(Boolean, default=False, nullable=False)

    version = relationship("InterviewVersion", back_populates="criteria")
    anchors = relationship(
        "ScoreAnchor",
        back_populates="criterion",
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<RubricCriterion {self.name} w={self.weight}>"


class ScoreAnchor(BaseModel):
    """Qualitative anchor describing weak / moderate / strong performance."""
    __tablename__ = "score_anchors"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    criterion_id = Column(UUID(as_uuid=True), ForeignKey("rubric_criteria.id", ondelete="CASCADE"), nullable=False, index=True)

    level = Column(String(20), nullable=False)  # weak|moderate|strong
    descriptor = Column(Text, nullable=False)

    criterion = relationship("RubricCriterion", back_populates="anchors")
