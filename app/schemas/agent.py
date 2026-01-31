"""
ConvoHubAI - Agent Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from app.models.agent import AgentType, AgentStatus


# ============================================
# AGENT SCHEMAS
# ============================================

class AgentCreate(BaseModel):
    """Agent creation request."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    agent_type: AgentType = AgentType.SINGLE_PROMPT
    channels: List[str] = ["chat"]
    
    # LLM Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: str = "1000"
    
    # Prompts
    system_prompt: Optional[str] = None
    welcome_message: Optional[str] = None
    fallback_message: Optional[str] = None
    
    # Voice settings
    voice_id: Optional[str] = None
    voice_provider: Optional[str] = None
    language: str = "en"
    
    # Advanced
    conversation_flow: Optional[Dict[str, Any]] = None
    functions: Optional[List[Dict[str, Any]]] = None
    guardrails: Optional[Dict[str, Any]] = None
    
    # Knowledge base
    knowledge_base_id: Optional[UUID] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Admissions Assistant",
                "description": "Helps prospective students with inquiries",
                "agent_type": "single_prompt",
                "channels": ["chat", "voice"],
                "llm_provider": "openai",
                "llm_model": "gpt-4",
                "system_prompt": "You are a helpful admissions assistant...",
                "welcome_message": "Hello! I'm here to help with your admissions questions.",
                "language": "en"
            }
        }


class AgentUpdate(BaseModel):
    """Agent update request."""
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    status: Optional[AgentStatus] = None
    channels: Optional[List[str]] = None
    
    # LLM Configuration
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[str] = None
    
    # Prompts
    system_prompt: Optional[str] = None
    welcome_message: Optional[str] = None
    fallback_message: Optional[str] = None
    
    # Voice settings
    voice_id: Optional[str] = None
    voice_provider: Optional[str] = None
    language: Optional[str] = None
    
    # Advanced
    conversation_flow: Optional[Dict[str, Any]] = None
    functions: Optional[List[Dict[str, Any]]] = None
    guardrails: Optional[Dict[str, Any]] = None
    
    # Knowledge base
    knowledge_base_id: Optional[UUID] = None
    
    # Chat Settings (NEW)
    response_style: Optional[str] = None  # conversational, formal, concise
    
    # Security Settings (NEW)
    content_filter_enabled: Optional[bool] = None


class AgentResponse(BaseModel):
    """Agent response."""
    id: UUID
    name: str
    description: Optional[str]
    agent_type: AgentType
    channels: List[str]
    status: AgentStatus
    
    llm_provider: str
    llm_model: str
    temperature: float
    max_tokens: str
    
    system_prompt: Optional[str]
    welcome_message: Optional[str]
    fallback_message: Optional[str]
    
    voice_id: Optional[str]
    voice_provider: Optional[str]
    language: str
    
    conversation_flow: Optional[Dict[str, Any]]
    functions: Optional[List[Dict[str, Any]]]
    guardrails: Optional[Dict[str, Any]]
    
    # Chat Settings (NEW)
    response_style: Optional[str] = "conversational"
    
    # Security Settings (NEW)
    content_filter_enabled: Optional[bool] = True
    
    # Publishing (NEW)
    is_published: Optional[bool] = False
    published_at: Optional[str] = None
    
    workspace_id: UUID
    created_by_id: Optional[UUID]
    knowledge_base_id: Optional[UUID]
    
    total_conversations: str
    total_messages: str
    avg_rating: Optional[float]
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class AgentListResponse(BaseModel):
    """List of agents."""
    items: List[AgentResponse]
    total: int


# ============================================
# AGENT TEMPLATE SCHEMAS
# ============================================

class AgentTemplateResponse(BaseModel):
    """Agent template response."""
    id: UUID
    name: str
    description: Optional[str]
    category: Optional[str]
    thumbnail_url: Optional[str]
    agent_type: AgentType
    channels: List[str]
    system_prompt: Optional[str]
    welcome_message: Optional[str]
    conversation_flow: Optional[Dict[str, Any]]
    functions: Optional[List[Dict[str, Any]]]
    is_featured: bool
    usage_count: str
    
    class Config:
        from_attributes = True


class AgentTemplateListResponse(BaseModel):
    """List of agent templates."""
    items: List[AgentTemplateResponse]
    total: int


# ============================================
# AGENT ACTIONS
# ============================================

class AgentDuplicate(BaseModel):
    """Duplicate agent request."""
    name: str = Field(..., min_length=1, max_length=255)


class AgentFromTemplate(BaseModel):
    """Create agent from template request."""
    template_id: UUID
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None