"""
ConvoHubAI - Conversation Model
Conversations and messages between users and agents
"""
from sqlalchemy import Column, String, Boolean, Text, JSON, Enum, ForeignKey, Float, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import enum

from app.models.base import BaseModel


class ConversationType(str, enum.Enum):
    """Conversation types."""
    VOICE_INBOUND = "voice_inbound"
    VOICE_OUTBOUND = "voice_outbound"
    CHAT = "chat"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    VIDEO = "video"


class ConversationStatus(str, enum.Enum):
    """Conversation status."""
    ACTIVE = "active"
    COMPLETED = "completed"
    TRANSFERRED = "transferred"
    FAILED = "failed"
    ABANDONED = "abandoned"


class MessageRole(str, enum.Enum):
    """Message sender role."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class Conversation(BaseModel):
    """Conversation session model."""
    __tablename__ = "conversations"
    
    # Type and status
    conversation_type = Column(
        Enum(ConversationType),
        nullable=False
    )
    status = Column(
        Enum(ConversationStatus),
        default=ConversationStatus.ACTIVE,
        nullable=False
    )
    
    # Contact info
    contact_phone = Column(String(20), nullable=True)
    contact_email = Column(String(255), nullable=True)
    contact_name = Column(String(255), nullable=True)
    visitor_id = Column(String(255), nullable=True)  # For anonymous chat visitors
    
    # Call details (for voice)
    call_sid = Column(String(100), nullable=True)  # Twilio/Retell call ID
    from_number = Column(String(20), nullable=True)
    to_number = Column(String(20), nullable=True)
    recording_url = Column(String(500), nullable=True)
    transcript = Column(Text, nullable=True)
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(String(10), nullable=True)
    
    # Analytics
    message_count = Column(String(10), default="0")
    user_message_count = Column(String(10), default="0")
    ai_message_count = Column(String(10), default="0")
    
    # Sentiment & rating
    sentiment_score = Column(Float, nullable=True)  # -1 to 1
    user_rating = Column(String(5), nullable=True)  # 1-5
    user_feedback = Column(Text, nullable=True)
    
    # Resolution
    resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text, nullable=True)
    
    # Transfer
    transferred_to = Column(String(255), nullable=True)
    transfer_reason = Column(Text, nullable=True)
    
    # Extra data
    extra_data = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)
    
    # Relationships
    agent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="CASCADE"),
        nullable=False
    )
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False
    )
    
    agent = relationship("Agent", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )
    
    def __repr__(self):
        return f"<Conversation {self.id}>"


class Message(BaseModel):
    """Individual message in a conversation."""
    __tablename__ = "messages"
    
    # Content
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    
    # For function calls
    function_name = Column(String(100), nullable=True)
    function_args = Column(JSON, nullable=True)
    function_result = Column(JSON, nullable=True)
    
    # Audio (for voice)
    audio_url = Column(String(500), nullable=True)
    audio_duration = Column(Float, nullable=True)
    
    # Tokens
    prompt_tokens = Column(String(10), nullable=True)
    completion_tokens = Column(String(10), nullable=True)
    
    # Latency
    latency_ms = Column(String(10), nullable=True)
    
    # Extra data
    extra_data = Column(JSON, nullable=True)
    
    # Conversation
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False
    )
    
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message {self.role}: {self.content[:50]}...>"