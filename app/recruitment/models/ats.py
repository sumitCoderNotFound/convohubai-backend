"""Phase 9 - ATS integration models (provider-agnostic)."""
from sqlalchemy import Column, String, Boolean, ForeignKey, DateTime, JSON, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class AtsConnection(BaseModel):
    """A workspace's connection to an external ATS (Greenhouse, Lever, Workable...)."""
    __tablename__ = "ats_connections"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    provider = Column(String(30), nullable=False)  # greenhouse | lever | workable
    name = Column(String(255), nullable=True)
    api_key = Column(String(500), nullable=True)     # NOTE: encrypt at rest in production
    subdomain = Column(String(255), nullable=True)   # used by Workable
    base_url = Column(String(500), nullable=True)    # optional override
    enabled = Column(Boolean, default=True, nullable=False)
    last_sync_at = Column(DateTime, nullable=True)
    meta = Column(JSON, default=dict)

    def __repr__(self):
        return f"<AtsConnection {self.provider} ws={self.workspace_id}>"


class AtsMapping(BaseModel):
    """Maps an external ATS record to an internal record, to avoid duplicates on re-sync."""
    __tablename__ = "ats_mappings"
    __table_args__ = (
        UniqueConstraint("connection_id", "entity_type", "external_id", name="uq_ats_mapping"),
    )

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, index=True)
    connection_id = Column(UUID(as_uuid=True), ForeignKey("ats_connections.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_type = Column(String(20), nullable=False)  # job | candidate
    external_id = Column(String(255), nullable=False)
    internal_id = Column(UUID(as_uuid=True), nullable=False)
    extra = Column(JSON, default=dict)
