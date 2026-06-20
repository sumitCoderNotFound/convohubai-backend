"""Phase 13 - customisable email templates per workspace."""
from sqlalchemy import Column, String, Text, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class EmailTemplate(BaseModel):
    __tablename__ = "email_templates"
    __table_args__ = (UniqueConstraint("workspace_id", "kind", name="uq_email_template"),)

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    kind = Column(String(30), nullable=False)  # invite | selected | advance | rejected | reminder | completed | score_ready
    subject = Column(String(300), nullable=False)
    body_html = Column(Text, nullable=False)
    enabled = Column(Boolean, default=True, nullable=False)
