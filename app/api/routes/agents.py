"""
ConvoHubAI - Agent API Routes
Updated with TTS/STT provider support
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel


from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User, Workspace, WorkspaceMember
from app.models.agent import Agent, AgentTemplate, AgentStatus
from app.schemas.agent import (
    AgentCreate,
    AgentUpdate,
    AgentResponse,
    AgentListResponse,
    AgentTemplateResponse,
    AgentTemplateListResponse,
    AgentDuplicate,
    AgentFromTemplate
)
from app.schemas.auth import MessageResponse

router = APIRouter(prefix="/agents", tags=["Agents"])


async def get_user_workspace(
    user: User,
    db: AsyncSession,
    workspace_id: Optional[UUID] = None
) -> Workspace:
    """Get user's current workspace or specified workspace."""
    if workspace_id:
        result = await db.execute(
            select(Workspace).where(
                Workspace.id == workspace_id,
                Workspace.is_deleted == False
            )
        )
    elif user.current_workspace_id:
        result = await db.execute(
            select(Workspace).where(
                Workspace.id == user.current_workspace_id,
                Workspace.is_deleted == False
            )
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No workspace selected"
        )
    
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    # Verify membership
    member_result = await db.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace.id,
            WorkspaceMember.user_id == user.id
        )
    )
    
    if not member_result.scalar_one_or_none() and not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this workspace"
        )
    
    return workspace


async def get_agent_or_404(
    agent_id: str,
    workspace: Workspace,
    db: AsyncSession
) -> Agent:
    """Get agent by ID or raise 404."""
    result = await db.execute(
        select(Agent).where(
            Agent.id == agent_id,
            Agent.workspace_id == workspace.id,
            Agent.is_deleted == False
        )
    )
    agent = result.scalar_one_or_none()
    
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    
    return agent


@router.get("", response_model=AgentListResponse)
async def list_agents(
    workspace_id: Optional[UUID] = Query(None),
    status_filter: Optional[AgentStatus] = Query(None, alias="status"),
    channel: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List agents in the workspace."""
    workspace = await get_user_workspace(current_user, db, workspace_id)
    
    # Build query
    query = select(Agent).where(
        Agent.workspace_id == workspace.id,
        Agent.is_deleted == False
    )
    
    if status_filter:
        query = query.where(Agent.status == status_filter)
    
    if search:
        query = query.where(
            Agent.name.ilike(f"%{search}%") |
            Agent.description.ilike(f"%{search}%")
        )
    
    # Get total count
    count_result = await db.execute(
        select(func.count()).select_from(query.subquery())
    )
    total = count_result.scalar() or 0
    
    # Paginate
    query = query.order_by(Agent.updated_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    result = await db.execute(query)
    agents = result.scalars().all()
    
    return AgentListResponse(
        items=[AgentResponse.model_validate(a) for a in agents],
        total=total
    )


@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_agent(
    agent_data: AgentCreate,
    workspace_id: Optional[UUID] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new agent."""
    workspace = await get_user_workspace(current_user, db, workspace_id)
    
    # Check agent limit
    agent_count_result = await db.execute(
        select(func.count(Agent.id)).where(
            Agent.workspace_id == workspace.id,
            Agent.is_deleted == False
        )
    )
    current_count = agent_count_result.scalar() or 0
    max_agents = int(workspace.max_agents or "1")
    
    if current_count >= max_agents:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Agent limit reached ({max_agents}). Upgrade your plan to create more agents."
        )
    
    # Create agent - UPDATED with tts_provider and stt_provider!
    agent = Agent(
        name=agent_data.name,
        description=agent_data.description,
        agent_type=agent_data.agent_type,
        channels=agent_data.channels,
        status=AgentStatus.DRAFT,
        llm_provider=agent_data.llm_provider,
        llm_model=agent_data.llm_model,
        temperature=agent_data.temperature,
        max_tokens=agent_data.max_tokens,
        system_prompt=agent_data.system_prompt,
        welcome_message=agent_data.welcome_message,
        fallback_message=agent_data.fallback_message,
        voice_id=agent_data.voice_id,
        voice_provider=agent_data.voice_provider,
        tts_provider=agent_data.tts_provider,  # NEW!
        stt_provider=agent_data.stt_provider,  # NEW!
        language=agent_data.language,
        conversation_flow=agent_data.conversation_flow,
        functions=agent_data.functions,
        guardrails=agent_data.guardrails,
        workspace_id=workspace.id,
        created_by_id=current_user.id,
        knowledge_base_id=agent_data.knowledge_base_id
    )
    
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    
    return AgentResponse.model_validate(agent)


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    workspace_id: Optional[UUID] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get agent details."""
    workspace = await get_user_workspace(current_user, db, workspace_id)
    agent = await get_agent_or_404(agent_id, workspace, db)
    
    return AgentResponse.model_validate(agent)


@router.patch("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    agent_data: AgentUpdate,
    workspace_id: Optional[UUID] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update agent settings."""
    workspace = await get_user_workspace(current_user, db, workspace_id)
    agent = await get_agent_or_404(agent_id, workspace, db)
    
    # Update fields - This automatically handles tts_provider and stt_provider
    update_data = agent_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(agent, field, value)
    
    await db.commit()
    await db.refresh(agent)
    
    return AgentResponse.model_validate(agent)


@router.delete("/{agent_id}", response_model=MessageResponse)
async def delete_agent(
    agent_id: str,
    workspace_id: Optional[UUID] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete (soft) an agent."""
    workspace = await get_user_workspace(current_user, db, workspace_id)
    agent = await get_agent_or_404(agent_id, workspace, db)
    
    agent.is_deleted = True
    agent.status = AgentStatus.ARCHIVED
    await db.commit()
    
    return MessageResponse(message="Agent deleted successfully")


@router.post("/{agent_id}/activate", response_model=AgentResponse)
async def activate_agent(
    agent_id: str,
    workspace_id: Optional[UUID] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Activate an agent."""
    workspace = await get_user_workspace(current_user, db, workspace_id)
    agent = await get_agent_or_404(agent_id, workspace, db)
    
    # Validate agent has required configuration
    if not agent.system_prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent must have a system prompt before activation"
        )
    
    agent.status = AgentStatus.ACTIVE
    await db.commit()
    await db.refresh(agent)
    
    return AgentResponse.model_validate(agent)


@router.post("/{agent_id}/pause", response_model=AgentResponse)
async def pause_agent(
    agent_id: str,
    workspace_id: Optional[UUID] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Pause an agent."""
    workspace = await get_user_workspace(current_user, db, workspace_id)
    agent = await get_agent_or_404(agent_id, workspace, db)
    
    agent.status = AgentStatus.PAUSED
    await db.commit()
    await db.refresh(agent)
    
    return AgentResponse.model_validate(agent)


@router.post("/{agent_id}/duplicate", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def duplicate_agent(
    agent_id: str,
    duplicate_data: AgentDuplicate,
    workspace_id: Optional[UUID] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Duplicate an existing agent."""
    workspace = await get_user_workspace(current_user, db, workspace_id)
    original = await get_agent_or_404(agent_id, workspace, db)
    
    # Create duplicate - UPDATED with tts_provider and stt_provider!
    agent = Agent(
        name=duplicate_data.name,
        description=original.description,
        agent_type=original.agent_type,
        channels=original.channels,
        status=AgentStatus.DRAFT,
        llm_provider=original.llm_provider,
        llm_model=original.llm_model,
        temperature=original.temperature,
        max_tokens=original.max_tokens,
        system_prompt=original.system_prompt,
        welcome_message=original.welcome_message,
        fallback_message=original.fallback_message,
        voice_id=original.voice_id,
        voice_provider=original.voice_provider,
        tts_provider=original.tts_provider,  # NEW!
        stt_provider=original.stt_provider,  # NEW!
        language=original.language,
        conversation_flow=original.conversation_flow,
        functions=original.functions,
        guardrails=original.guardrails,
        workspace_id=workspace.id,
        created_by_id=current_user.id,
        knowledge_base_id=original.knowledge_base_id
    )
    
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    
    return AgentResponse.model_validate(agent)


# ============================================
# AGENT TEMPLATES
# ============================================

@router.get("/templates/list", response_model=AgentTemplateListResponse)
async def list_templates(
    category: Optional[str] = Query(None),
    featured_only: bool = Query(False),
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List available agent templates."""
    query = select(AgentTemplate)
    
    if category:
        query = query.where(AgentTemplate.category == category)
    
    if featured_only:
        query = query.where(AgentTemplate.is_featured == True)
    
    if search:
        query = query.where(
            AgentTemplate.name.ilike(f"%{search}%") |
            AgentTemplate.description.ilike(f"%{search}%")
        )
    
    query = query.order_by(AgentTemplate.is_featured.desc(), AgentTemplate.usage_count.desc())
    
    result = await db.execute(query)
    templates = result.scalars().all()
    
    return AgentTemplateListResponse(
        items=[AgentTemplateResponse.model_validate(t) for t in templates],
        total=len(templates)
    )


@router.post("/from-template", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_from_template(
    template_data: AgentFromTemplate,
    workspace_id: Optional[UUID] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new agent from a template."""
    workspace = await get_user_workspace(current_user, db, workspace_id)
    
    # Get template
    result = await db.execute(
        select(AgentTemplate).where(AgentTemplate.id == template_data.template_id)
    )
    template = result.scalar_one_or_none()
    
    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found"
        )
    
    # Create agent from template - UPDATED with voice settings!
    agent = Agent(
        name=template_data.name,
        description=template_data.description or template.description,
        agent_type=template.agent_type,
        channels=template.channels,
        status=AgentStatus.DRAFT,
        llm_provider="groq",
        llm_model="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens="1000",
        system_prompt=template.system_prompt,
        welcome_message=template.welcome_message,
        conversation_flow=template.conversation_flow,
        functions=template.functions,
        tts_provider=getattr(template, 'tts_provider', 'deepgram'),  # NEW!
        stt_provider="groq",  # NEW!
        voice_id=getattr(template, 'voice_id', 'aura-asteria-en'),  # NEW!
        language=getattr(template, 'language', 'en-GB'),  # NEW!
        workspace_id=workspace.id,
        created_by_id=current_user.id
    )
    
    db.add(agent)
    
    # Increment template usage
    template.usage_count = str(int(template.usage_count or "0") + 1)
    
    await db.commit()
    await db.refresh(agent)
    
    return AgentResponse.model_validate(agent)

# ============================================
# CHAT ENDPOINT FOR TEST AUDIO
# ============================================

class AgentChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[dict]] = []
    session_id: Optional[str] = None


@router.post("/{agent_id}/chat")
async def chat_with_agent(
    agent_id: UUID,
    request: AgentChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Chat with an AI agent for voice/text testing."""
    from app.services.llm_service import llm_service
    from datetime import datetime
    
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

Keep responses concise and conversational - suitable for voice interaction."""

    # Build messages
    messages = []
    for msg in (request.conversation_history or []):
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })
    messages.append({"role": "user", "content": request.message})
    
    try:
        response = await llm_service.generate_response(
            messages=messages,
            system_prompt=system_prompt,
            model=agent.llm_model or "llama-3.3-70b-versatile",
            provider=agent.llm_provider or "groq",
            temperature=agent.temperature or 0.7,
            max_tokens=300,
        )
        
        return {
            "response": response,
            "session_id": request.session_id or f"session_{agent_id}",
            "timestamp": datetime.utcnow().isoformat(),
            # Return voice settings for frontend
            "voice_config": {
                "tts_provider": agent.tts_provider or "deepgram",
                "stt_provider": agent.stt_provider or "groq",
                "voice_id": agent.voice_id or "aura-asteria-en",
                "language": agent.language or "en-GB"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")