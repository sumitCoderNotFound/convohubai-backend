"""
ConvoHubAI - Agent Model
AI agents that handle conversations
Updated with Voice/TTS/STT settings
"""
from sqlalchemy import Column, String, Boolean, Text, JSON, Enum, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import enum

from app.models.base import BaseModel


class AgentType(str, enum.Enum):
    """Types of AI agents."""
    SINGLE_PROMPT = "single_prompt"
    CONVERSATION_FLOW = "conversation_flow"
    MULTI_PROMPT = "multi_prompt"
    CUSTOM_LLM = "custom_llm"


class AgentChannel(str, enum.Enum):
    """Communication channels."""
    VOICE = "voice"
    CHAT = "chat"
    SMS = "sms"
    WHATSAPP = "whatsapp"
    VIDEO = "video"


class AgentStatus(str, enum.Enum):
    """Agent status."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


class Agent(BaseModel):
    """AI Agent model."""
    __tablename__ = "agents"
    
    # Basic info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Configuration
    agent_type = Column(
        Enum(AgentType),
        default=AgentType.SINGLE_PROMPT,
        nullable=False
    )
    channels = Column(JSON, default=["chat"])  # List of channels
    status = Column(
        Enum(AgentStatus),
        default=AgentStatus.DRAFT,
        nullable=False
    )
    
    # LLM Configuration
    llm_provider = Column(String(50), default="groq")  # groq, openai, anthropic
    llm_model = Column(String(100), default="llama-3.3-70b-versatile")
    temperature = Column(Float, default=0.7)
    max_tokens = Column(String(10), default="1000")
    
    # ===========================================
    # VOICE/TTS/STT SETTINGS (NEW!)
    # ===========================================
    
    # TTS (Text-to-Speech) Settings
    tts_provider = Column(String(50), default="deepgram")  # deepgram, openai, elevenlabs
    voice_id = Column(String(100), default="aura-asteria-en")  # Voice ID for TTS
    
    # STT (Speech-to-Text) Settings  
    stt_provider = Column(String(50), default="groq")  # groq (free!), deepgram
    
    # Language Settings
    language = Column(String(10), default="en-GB")  # en-GB, en-US, es-ES, fr-FR, etc.
    
    # Legacy voice fields (for backward compatibility)
    voice_provider = Column(String(50), nullable=True)  # retell, elevenlabs, etc.
    
    # ===========================================
    # PROMPTS
    # ===========================================
    system_prompt = Column(Text, nullable=True)
    welcome_message = Column(Text, nullable=True)
    fallback_message = Column(Text, nullable=True)
    
    # Behavior
    conversation_flow = Column(JSON, nullable=True)  # Flow configuration
    functions = Column(JSON, nullable=True)  # Available functions/tools
    guardrails = Column(JSON, nullable=True)  # Safety rules
    
    # Chat Settings
    response_style = Column(String(50), default="conversational")  # conversational, formal, concise
    
    # Security Settings
    content_filter_enabled = Column(Boolean, default=True)
    
    # Publishing
    is_published = Column(Boolean, default=False)
    published_at = Column(String(50), nullable=True)
    
    # Workspace
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Creator
    created_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    
    # Knowledge base
    knowledge_base_id = Column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="SET NULL"),
        nullable=True
    )
    
    # Stats
    total_conversations = Column(String(20), default="0")
    total_messages = Column(String(20), default="0")
    avg_rating = Column(Float, nullable=True)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="agents")
    created_by = relationship("User", foreign_keys=[created_by_id])
    knowledge_base = relationship("KnowledgeBase", back_populates="agents")
    conversations = relationship(
        "Conversation",
        back_populates="agent",
        cascade="all, delete-orphan"
    )
    phone_numbers = relationship(
        "PhoneNumber",
        back_populates="agent"
    )
    
    def __repr__(self):
        return f"<Agent {self.name}>"
    
    def get_voice_config(self):
        """Get voice configuration for this agent."""
        return {
            "tts_provider": self.tts_provider or "deepgram",
            "voice_id": self.voice_id or "aura-asteria-en",
            "stt_provider": self.stt_provider or "groq",
            "language": self.language or "en-GB",
        }


class AgentTemplate(BaseModel):
    """Pre-built agent templates."""
    __tablename__ = "agent_templates"
    
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=True)  # education, hospitality, support, sales
    thumbnail_url = Column(String(500), nullable=True)
    
    # Template configuration
    agent_type = Column(Enum(AgentType), nullable=False)
    channels = Column(JSON, default=["chat"])
    
    # Template content
    system_prompt = Column(Text, nullable=True)
    welcome_message = Column(Text, nullable=True)
    conversation_flow = Column(JSON, nullable=True)
    functions = Column(JSON, nullable=True)
    
    # Voice settings
    tts_provider = Column(String(50), default="deepgram")
    voice_id = Column(String(100), default="aura-asteria-en")
    language = Column(String(10), default="en-GB")
    
    # Metadata
    is_featured = Column(Boolean, default=False)
    usage_count = Column(String(20), default="0")
    
    def __repr__(self):
        return f"<AgentTemplate {self.name}>"