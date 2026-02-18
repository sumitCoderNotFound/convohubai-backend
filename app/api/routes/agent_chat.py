"""
ConvoHubAI - Agent Chat Endpoint
For real-time voice/chat conversations with agents
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.agent import Agent
from app.services.llm_service import llm_service


router = APIRouter()


class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class AgentChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[ChatMessage]] = []
    session_id: Optional[str] = None


class AgentChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: datetime


@router.post("/agents/{agent_id}/chat", response_model=AgentChatResponse)
async def chat_with_agent(
    agent_id: UUID,
    request: AgentChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Chat with an AI agent. Used for both text chat and voice testing.
    """
    # Get agent
    result = await db.execute(
        select(Agent).where(
            and_(
                Agent.id == agent_id,
                Agent.workspace_id == current_user.current_workspace_id,
                Agent.is_deleted == False,
            )
        )
    )
    agent = result.scalar_one_or_none()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Build system prompt
    system_prompt = agent.system_prompt or f"""You are {agent.name}, an AI assistant.
{agent.description or ''}

Your role is to help users with their queries in a friendly and professional manner.
Keep your responses concise and conversational - suitable for voice interaction.
Respond naturally as if you're having a phone conversation."""

    # Build messages for LLM
    messages = []
    
    # Add conversation history
    for msg in request.conversation_history:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": request.message
    })
    
    try:
        # Get response from LLM
        response = await llm_service.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            model=agent.llm_model or "llama-3.3-70b-versatile",
            provider=agent.llm_provider or "groq",
            temperature=agent.temperature or 0.7,
            max_tokens=agent.max_tokens or 300,  # Keep short for voice
        )
        
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{agent_id}"
        
        return AgentChatResponse(
            response=response,
            session_id=session_id,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")


@router.post("/agents/{agent_id}/test-voice")
async def test_voice_call(
    agent_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Initialize a voice test session for an agent.
    Returns the agent's first message/greeting.
    """
    # Get agent
    result = await db.execute(
        select(Agent).where(
            and_(
                Agent.id == agent_id,
                Agent.workspace_id == current_user.current_workspace_id,
                Agent.is_deleted == False,
            )
        )
    )
    agent = result.scalar_one_or_none()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Get first message
    first_message = agent.welcome_message or f"Hello! Thank you for calling. I'm {agent.name}. How can I assist you today?"
    
    return {
        "agent_id": str(agent.id),
        "agent_name": agent.name,
        "first_message": first_message,
        "voice": agent.voice_id or "alloy",
        "language": agent.language or "en-US",
        "session_id": f"voice_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{agent_id}"
    }