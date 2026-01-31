"""
ConvoHubAI - Webhook Model
Webhook configurations for agents
"""
from sqlalchemy import Column, String, Boolean, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class Webhook(BaseModel):
    """Webhook configuration model."""
    __tablename__ = "webhooks"
    
    # Basic info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Configuration
    url = Column(String(500), nullable=False)
    secret = Column(String(255), nullable=True)  # For signature verification
    
    # Events to trigger
    events = Column(JSON, default=[])  # List of event types: conversation_started, message_received, etc.
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Stats
    total_calls = Column(String(20), default="0")
    successful_calls = Column(String(20), default="0")
    failed_calls = Column(String(20), default="0")
    last_triggered_at = Column(String(50), nullable=True)
    last_error = Column(Text, nullable=True)
    
    # Agent relationship
    agent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Workspace relationship
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False
    )
    
    def __repr__(self):
        return f"<Webhook {self.name}>"