"""Recruitment - InterviewQuestion + BranchRule (Feature 4)."""
from sqlalchemy import Column, String, Text, JSON, ForeignKey, Boolean, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class InterviewQuestion(BaseModel):
    """A structured question attached to an interview version."""
    __tablename__ = "interview_questions"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id = Column(UUID(as_uuid=True), ForeignKey("interview_versions.id", ondelete="CASCADE"), nullable=False, index=True)

    order_index = Column(Integer, default=0, nullable=False)
    question_type = Column(String(50), default="open_response", nullable=False)
    prompt_text = Column(Text, nullable=False)

    # { max_answer_seconds, probing_depth, required, options: [...], min, max, ... }
    config = Column(JSON, default=dict)

    # Self-reference for AI-generated follow-ups (FR-QUE-003 traceability).
    parent_question_id = Column(UUID(as_uuid=True), ForeignKey("interview_questions.id", ondelete="CASCADE"), nullable=True)
    is_knockout = Column(Boolean, default=False, nullable=False)

    version = relationship("InterviewVersion", back_populates="questions")
    branch_rules = relationship(
        "BranchRule",
        back_populates="question",
        cascade="all, delete-orphan",
        order_by="BranchRule.order_index",
    )

    def __repr__(self):
        return f"<InterviewQuestion {self.order_index} {self.question_type}>"


class BranchRule(BaseModel):
    """Conditional branching / knockout rule evaluated after a question."""
    __tablename__ = "branch_rules"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id = Column(UUID(as_uuid=True), ForeignKey("interview_versions.id", ondelete="CASCADE"), nullable=False, index=True)
    question_id = Column(UUID(as_uuid=True), ForeignKey("interview_questions.id", ondelete="CASCADE"), nullable=False, index=True)

    order_index = Column(Integer, default=0, nullable=False)
    # { type: "equals|contains|gt|lt|knockout", value: ... }
    condition = Column(JSON, default=dict)
    # { action: "skip_to|end|knockout|continue", target_question_id: ... }
    action = Column(JSON, default=dict)

    question = relationship("InterviewQuestion", back_populates="branch_rules")
