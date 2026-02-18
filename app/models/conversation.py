"""
ConvoHubAI - Conversation Model
Conversations and messages between users and agents
Supports: Voice, Video, Chat, SMS, WhatsApp
"""
from sqlalchemy import Column, String, Boolean, Text, JSON, Enum, ForeignKey, Float, DateTime, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import enum
from datetime import datetime

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
        nullable=True,
        default=ConversationType.CHAT
    )
    status = Column(
        Enum(ConversationStatus),
        default=ConversationStatus.ACTIVE,
        nullable=False
    )
    
    # Channel (simplified)
    channel = Column(String(50), nullable=True)  # video, voice, chat, sms
    
    # Contact info
    contact_phone = Column(String(20), nullable=True)
    contact_email = Column(String(255), nullable=True)
    contact_name = Column(String(255), nullable=True)
    visitor_id = Column(String(255), nullable=True)  # For anonymous chat visitors
    
    # Participant info (for video calls)
    participant_identity = Column(String(255), nullable=True)
    room_name = Column(String(255), nullable=True)  # LiveKit room name
    
    # Call details (for voice)
    call_sid = Column(String(100), nullable=True)  # Twilio/Retell call ID
    from_number = Column(String(20), nullable=True)
    to_number = Column(String(20), nullable=True)
    recording_url = Column(String(500), nullable=True)
    
    # Transcript - stored as JSON array for video/voice calls
    transcript = Column(JSON, nullable=True)  # [{role, content, timestamp}]
    transcript_text = Column(Text, nullable=True)  # Plain text version
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    call_start = Column(DateTime, nullable=True)  # Alias for video calls
    call_end = Column(DateTime, nullable=True)    # Alias for video calls
    duration_seconds = Column(Integer, default=0)
    
    # Analytics
    message_count = Column(Integer, default=0)
    user_message_count = Column(Integer, default=0)
    ai_message_count = Column(Integer, default=0)
    
    # Sentiment & rating
    sentiment_score = Column(Float, nullable=True)  # -1 to 1
    sentiment = Column(String(50), nullable=True)  # positive, negative, neutral
    user_rating = Column(Integer, nullable=True)  # 1-5
    user_feedback = Column(Text, nullable=True)
    
    # Resolution
    resolved = Column(Boolean, default=False)
    resolution_notes = Column(Text, nullable=True)
    
    # Summary (AI-generated)
    summary = Column(Text, nullable=True)
    
    # Transfer
    transferred_to = Column(String(255), nullable=True)
    transfer_reason = Column(Text, nullable=True)
    
    # Collected data from conversation (name, email, phone, etc)
    collected_data = Column(JSON, nullable=True)
    
    # Extra data
    extra_data = Column(JSON, nullable=True)
    call_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata' (reserved)
    tags = Column(JSON, nullable=True)
    
    # Relationships
    agent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True
    )
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=True
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    
    agent = relationship("Agent", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at"
    )
    
    def __repr__(self):
        return f"<Conversation {self.id} - {self.channel}>"
    
    @property
    def duration_formatted(self) -> str:
        """Return duration in MM:SS format."""
        if not self.duration_seconds:
            return "00:00"
        mins = self.duration_seconds // 60
        secs = self.duration_seconds % 60
        return f"{mins:02d}:{secs:02d}"
    
    def get_transcript_text(self) -> str:
        """Convert JSON transcript to plain text."""
        if not self.transcript:
            return ""
        lines = []
        for msg in self.transcript:
            role = msg.get("role", "unknown").title()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)


class Message(BaseModel):
    """Individual message in a conversation."""
    __tablename__ = "messages"
    
    # Content
    role = Column(Enum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    
    # Timestamp
    message_timestamp = Column(DateTime, default=datetime.utcnow)
    
    # For function calls
    function_name = Column(String(100), nullable=True)
    function_args = Column(JSON, nullable=True)
    function_result = Column(JSON, nullable=True)
    
    # Audio (for voice)
    audio_url = Column(String(500), nullable=True)
    audio_duration = Column(Float, nullable=True)
    
    # Tokens
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    
    # Latency
    latency_ms = Column(Integer, nullable=True)
    
    # Extra data
    extra_data = Column(JSON, nullable=True)
    message_metadata = Column(JSON, nullable=True)  # Renamed from 'metadata' (reserved)
    
    # Conversation
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False
    )
    
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<Message {self.role}: {self.content[:50]}...>"