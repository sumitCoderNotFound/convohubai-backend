"""Phase 12 - candidate documents (resume / cover letter)."""
from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class CandidateDocument(BaseModel):
    __tablename__ = "candidate_documents"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    application_id = Column(UUID(as_uuid=True), ForeignKey("applications.id", ondelete="SET NULL"), nullable=True, index=True)

    kind = Column(String(30), default="resume", nullable=False)   # resume | cover_letter | other
    filename = Column(String(400), nullable=False)
    content_type = Column(String(120), nullable=True)
    size = Column(Integer, default=0)
    storage_path = Column(String(700), nullable=False)
    source = Column(String(20), default="candidate", nullable=False)  # candidate | recruiter
