"""
ConvoHubAI - Chat API Routes
Handles real-time chat with AI agents
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Optional, List
from uuid import UUID
from datetime import datetime
import json

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.agent import Agent, AgentStatus
from app.models.conversation import Conversation, Message, ConversationStatus, ConversationType, MessageRole
from app.services.llm_service import llm_service
from pydantic import BaseModel


router = APIRouter(prefix="/chat", tags=["Chat"])


# ============================================
# SCHEMAS
# ============================================

class ChatMessageRequest(BaseModel):
    """Request to send a chat message."""
    agent_id: UUID
    message: str
    conversation_id: Optional[UUID] = None


class ChatMessageResponse(BaseModel):
    """Response from chat."""
    conversation_id: UUID
    message_id: UUID
    response: str
    created_at: datetime


class ConversationResponse(BaseModel):
    """Conversation details."""
    id: UUID
    agent_id: UUID
    agent_name: str
    status: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    
    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    """Message details."""
    id: UUID
    role: str
    content: str
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============================================
# ROUTES
# ============================================

@router.post("/send", response_model=ChatMessageResponse)
async def send_message(
    request: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message to an AI agent and get a response.
    Creates a new conversation if conversation_id is not provided.
    """
    # Get the agent
    agent_result = await db.execute(
        select(Agent).where(
            and_(
                Agent.id == request.agent_id,
                Agent.workspace_id == current_user.current_workspace_id,
                Agent.is_deleted == False,
            )
        )
    )
    agent = agent_result.scalar_one_or_none()
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    
    # Get or create conversation
    conversation = None
    if request.conversation_id:
        conv_result = await db.execute(
            select(Conversation).where(
                and_(
                    Conversation.id == request.conversation_id,
                    Conversation.workspace_id == current_user.current_workspace_id,
                )
            )
        )
        conversation = conv_result.scalar_one_or_none()
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
    else:
        # Create new conversation
        conversation = Conversation(
            agent_id=agent.id,
            workspace_id=current_user.current_workspace_id,
            conversation_type=ConversationType.CHAT,
            status=ConversationStatus.ACTIVE,
            visitor_id=str(current_user.id),  # Store user ID as visitor_id
        )
        db.add(conversation)
        await db.flush()
    
    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content=request.message,
    )
    db.add(user_message)
    await db.flush()
    
    # Get conversation history
    messages_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
    )
    messages = messages_result.scalars().all()
    
    # Build messages for LLM
    llm_messages = [
        {"role": msg.role.value, "content": msg.content}
        for msg in messages
    ]
    
    # Generate AI response
    try:
        response_text = await llm_service.generate_response(
            messages=llm_messages,
            system_prompt=agent.system_prompt or "You are a helpful assistant.",
            model=agent.llm_model or "gpt-4o-mini",
            provider=agent.llm_provider or "openai",
            temperature=agent.temperature or 0.7,
            max_tokens=int(agent.max_tokens) if agent.max_tokens else 1000,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )
    
    # Save AI response
    ai_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT,
        content=response_text,
    )
    db.add(ai_message)
    
    # Update conversation
    conversation.updated_at = datetime.utcnow()
    conversation.message_count = str(int(conversation.message_count or "0") + 2)
    
    await db.commit()
    
    return ChatMessageResponse(
        conversation_id=conversation.id,
        message_id=ai_message.id,
        response=response_text,
        created_at=ai_message.created_at,
    )


@router.post("/send/stream")
async def send_message_stream(
    request: ChatMessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Send a message and stream the response back.
    Uses Server-Sent Events (SSE) for real-time streaming.
    """
    # Get the agent
    agent_result = await db.execute(
        select(Agent).where(
            and_(
                Agent.id == request.agent_id,
                Agent.workspace_id == current_user.current_workspace_id,
                Agent.is_deleted == False,
            )
        )
    )
    agent = agent_result.scalar_one_or_none()
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    
    # Get or create conversation
    conversation = None
    if request.conversation_id:
        conv_result = await db.execute(
            select(Conversation).where(
                and_(
                    Conversation.id == request.conversation_id,
                    Conversation.visitor_id == str(current_user.id),
                )
            )
        )
        conversation = conv_result.scalar_one_or_none()
    
    if not conversation:
        conversation = Conversation(
            agent_id=agent.id,
            workspace_id=current_user.current_workspace_id,
            conversation_type=ConversationType.CHAT,
            status=ConversationStatus.ACTIVE,
            visitor_id=str(current_user.id),
        )
        db.add(conversation)
        await db.flush()
    
    # Save user message
    user_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content=request.message,
    )
    db.add(user_message)
    await db.flush()
    
    # Get conversation history
    messages_result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
    )
    messages = messages_result.scalars().all()
    
    # Build messages for LLM
    llm_messages = [
        {"role": msg.role.value, "content": msg.content}
        for msg in messages
    ]
    
    conversation_id = conversation.id
    
    await db.commit()
    
    async def generate_stream():
        """Generate SSE stream."""
        full_response = ""
        
        # Send conversation ID first
        yield f"data: {json.dumps({'type': 'start', 'conversation_id': str(conversation_id)})}\n\n"
        
        try:
            async for chunk in llm_service.generate_response_stream(
                messages=llm_messages,
                system_prompt=agent.system_prompt or "You are a helpful assistant.",
                model=agent.llm_model or "gpt-4o-mini",
                provider=agent.llm_provider or "openai",
                temperature=agent.temperature or 0.7,
                max_tokens=int(agent.max_tokens) if agent.max_tokens else 1000,
            ):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            # Save complete response to database
            async with AsyncSession(db.get_bind()) as save_db:
                ai_message = Message(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT,
                    content=full_response,
                )
                save_db.add(ai_message)
                await save_db.commit()
            
            yield f"data: {json.dumps({'type': 'end', 'message_id': str(ai_message.id)})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(
    agent_id: Optional[UUID] = None,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List user's conversations."""
    query = select(Conversation).where(
        and_(
            Conversation.visitor_id == str(current_user.id),
            Conversation.workspace_id == current_user.current_workspace_id,
        )
    )
    
    if agent_id:
        query = query.where(Conversation.agent_id == agent_id)
    
    query = query.order_by(Conversation.updated_at.desc()).limit(limit)
    
    result = await db.execute(query)
    conversations = result.scalars().all()
    
    # Get agent names
    response = []
    for conv in conversations:
        agent_result = await db.execute(
            select(Agent).where(Agent.id == conv.agent_id)
        )
        agent = agent_result.scalar_one_or_none()
        
        response.append(ConversationResponse(
            id=conv.id,
            agent_id=conv.agent_id,
            agent_name=agent.name if agent else "Unknown",
            status=conv.status.value,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
            message_count=int(conv.message_count or "0"),
        ))
    
    return response


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get messages for a conversation."""
    # Verify conversation belongs to user
    conv_result = await db.execute(
        select(Conversation).where(
            and_(
                Conversation.id == conversation_id,
                Conversation.visitor_id == str(current_user.id),
            )
        )
    )
    conversation = conv_result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Get messages
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()
    
    return [
        MessageResponse(
            id=msg.id,
            role=msg.role.value,
            content=msg.content,
            created_at=msg.created_at,
        )
        for msg in messages
    ]


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation."""
    result = await db.execute(
        select(Conversation).where(
            and_(
                Conversation.id == conversation_id,
                Conversation.visitor_id == str(current_user.id),
            )
        )
    )
    conversation = result.scalar_one_or_none()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    await db.delete(conversation)
    await db.commit()
    
    return {"message": "Conversation deleted"}