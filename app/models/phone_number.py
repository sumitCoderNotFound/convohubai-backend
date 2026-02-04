"""
ConvoHubAI - Phone Number Model
Twilio phone numbers for voice and SMS
"""
from sqlalchemy import Column, String, Boolean, JSON, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import enum

from app.models.base import BaseModel


class PhoneNumberType(str, enum.Enum):
    """Phone number types."""
    LOCAL = "local"
    TOLL_FREE = "toll_free"
    MOBILE = "mobile"


class PhoneNumberStatus(str, enum.Enum):
    """Phone number status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class PhoneNumber(BaseModel):
    """Phone number model for voice and SMS."""
    __tablename__ = "phone_numbers"
    
    # Phone info
    phone_number = Column(String(20), unique=True, nullable=False)
    friendly_name = Column(String(255), nullable=True)
    
    # Twilio info
    twilio_sid = Column(String(100), nullable=True)

    provider = Column(String(50), default="twilio")
    phone_sid = Column(String(100), nullable=True)
    capabilities = Column(JSON, default={"voice": True, "sms": True})
    
    # Type and status
    number_type = Column(
        Enum(PhoneNumberType),
        default=PhoneNumberType.LOCAL,
        nullable=False
    )
    status = Column(
        Enum(PhoneNumberStatus),
        default=PhoneNumberStatus.PENDING,
        nullable=False
    )
    
    # Capabilities
    voice_enabled = Column(Boolean, default=True)
    sms_enabled = Column(Boolean, default=True)
    mms_enabled = Column(Boolean, default=False)
    
    # Location
    country_code = Column(String(5), default="US")
    region = Column(String(100), nullable=True)
    
    # Workspace
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Assigned agent
    agent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True
    )
    
    # Webhook URLs
    voice_webhook_url = Column(String(500), nullable=True)
    sms_webhook_url = Column(String(500), nullable=True)
    
    # Stats
    total_calls = Column(String(20), default="0")
    total_sms = Column(String(20), default="0")
    
    # Relationships
    workspace = relationship("Workspace", back_populates="phone_numbers")
    agent = relationship("Agent", back_populates="phone_numbers")
    
    def __repr__(self):
        return f"<PhoneNumber {self.phone_number}>"
