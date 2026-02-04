"""
ConvoHubAI - Call Model
Voice call records and history
"""
from sqlalchemy import Column, String, Boolean, Text, JSON, Enum, ForeignKey, Integer, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import enum

from app.models.base import BaseModel


class CallDirection(str, enum.Enum):
    """Call direction."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class CallStatus(str, enum.Enum):
    """Call status."""
    INITIATED = "initiated"
    RINGING = "ringing"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    BUSY = "busy"
    NO_ANSWER = "no-answer"
    FAILED = "failed"
    CANCELED = "canceled"


class Call(BaseModel):
    """Voice call record."""
    __tablename__ = "calls"
    
    # Call identifiers
    call_sid = Column(String(100), unique=True, nullable=False)  # Twilio Call SID
    
    # Direction and status
    direction = Column(Enum(CallDirection), nullable=False)
    status = Column(Enum(CallStatus), default=CallStatus.INITIATED)
    
    # Phone numbers
    from_number = Column(String(20), nullable=False)
    to_number = Column(String(20), nullable=False)
    
    # Timing
    started_at = Column(String(50), nullable=True)
    answered_at = Column(String(50), nullable=True)
    ended_at = Column(String(50), nullable=True)
    duration_seconds = Column(Integer, default=0)
    
    # Recording
    recording_url = Column(String(500), nullable=True)
    recording_sid = Column(String(100), nullable=True)
    recording_duration = Column(Integer, nullable=True)
    
    # Transcript
    transcript = Column(Text, nullable=True)
    transcript_segments = Column(JSON, nullable=True)  # [{role, text, timestamp}]
    
    # Cost tracking
    cost = Column(Float, default=0.0)
    currency = Column(String(10), default="USD")
    
    # AI Agent
    agent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True
    )
    
    # Phone number used
    phone_number_id = Column(
        UUID(as_uuid=True),
        ForeignKey("phone_numbers.id", ondelete="SET NULL"),
        nullable=True
    )
    
    # Workspace
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Metadata
    caller_info = Column(JSON, nullable=True)  # Caller details if available
    call_metadata = Column(JSON, nullable=True)  # Additional data
    
    # Sentiment/Analysis
    sentiment = Column(String(20), nullable=True)  # positive, negative, neutral
    summary = Column(Text, nullable=True)  # AI-generated summary
    
    # Error tracking
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    agent = relationship("Agent", foreign_keys=[agent_id])
    phone_number = relationship("PhoneNumber", foreign_keys=[phone_number_id])
    workspace = relationship("Workspace")
    
    def __repr__(self):
        return f"<Call {self.call_sid}>"


class CallEvent(BaseModel):
    """Call events/webhooks log."""
    __tablename__ = "call_events"
    
    # Event info
    event_type = Column(String(50), nullable=False)  # initiated, ringing, answered, completed, etc.
    event_data = Column(JSON, nullable=True)
    
    # Call reference
    call_id = Column(
        UUID(as_uuid=True),
        ForeignKey("calls.id", ondelete="CASCADE"),
        nullable=False
    )
    call_sid = Column(String(100), nullable=False)
    
    # Timestamp from Twilio
    twilio_timestamp = Column(String(50), nullable=True)
    
    # Relationships
    call = relationship("Call", foreign_keys=[call_id])
    
    def __repr__(self):
        return f"<CallEvent {self.event_type} for {self.call_sid}>"