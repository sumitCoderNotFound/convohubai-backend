"""
ConvoHubAI - Conversation Flow API Routes
CRUD operations for visual workflow builder
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Optional, List, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.agent import Agent
from app.models.conversation_flow import ConversationFlow, FlowNode


router = APIRouter(prefix="/flows", tags=["Conversation Flows"])


# ============================================
# SCHEMAS
# ============================================

class NodeData(BaseModel):
    label: Optional[str] = None
    content: Optional[str] = None
    variable_name: Optional[str] = None
    expected_type: Optional[str] = None
    choices: Optional[List[str]] = None
    conditions: Optional[List[dict]] = None
    action_type: Optional[str] = None
    action_config: Optional[dict] = None


class FlowNodeSchema(BaseModel):
    id: str
    type: str
    position: dict
    data: NodeData


class FlowEdgeSchema(BaseModel):
    id: str
    source: str
    target: str
    sourceHandle: Optional[str] = None
    targetHandle: Optional[str] = None
    label: Optional[str] = None
    type: Optional[str] = "smoothstep"


class FlowCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    agent_id: UUID
    nodes: Optional[List[dict]] = []
    edges: Optional[List[dict]] = []


class FlowUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: Optional[List[dict]] = None
    edges: Optional[List[dict]] = None
    viewport: Optional[dict] = None
    is_active: Optional[bool] = None
    is_draft: Optional[bool] = None


class FlowResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    nodes: List[dict]
    edges: List[dict]
    viewport: dict
    is_active: bool
    is_draft: bool
    version: str
    agent_id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# ============================================
# FLOW CRUD OPERATIONS
# ============================================

@router.post("", response_model=FlowResponse)
async def create_flow(
    data: FlowCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new conversation flow."""
    # Verify agent exists and belongs to user's workspace
    agent_result = await db.execute(
        select(Agent).where(
            and_(
                Agent.id == data.agent_id,
                Agent.workspace_id == current_user.current_workspace_id,
                Agent.is_deleted == False,
            )
        )
    )
    agent = agent_result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Create default start node if no nodes provided
    default_nodes = data.nodes if data.nodes else [
        {
            "id": "start-1",
            "type": "start",
            "position": {"x": 250, "y": 50},
            "data": {"label": "Start"}
        }
    ]
    
    flow = ConversationFlow(
        name=data.name,
        description=data.description,
        nodes=default_nodes,
        edges=data.edges or [],
        agent_id=data.agent_id,
        workspace_id=current_user.current_workspace_id,
    )
    db.add(flow)
    await db.commit()
    await db.refresh(flow)
    
    return FlowResponse(
        id=flow.id,
        name=flow.name,
        description=flow.description,
        nodes=flow.nodes,
        edges=flow.edges,
        viewport=flow.viewport,
        is_active=flow.is_active,
        is_draft=flow.is_draft,
        version=flow.version,
        agent_id=flow.agent_id,
        created_at=flow.created_at,
        updated_at=flow.updated_at,
    )


@router.get("", response_model=List[FlowResponse])
async def list_flows(
    agent_id: Optional[UUID] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all conversation flows."""
    query = select(ConversationFlow).where(
        and_(
            ConversationFlow.workspace_id == current_user.current_workspace_id,
            ConversationFlow.is_deleted == False,
        )
    )
    
    if agent_id:
        query = query.where(ConversationFlow.agent_id == agent_id)
    
    query = query.order_by(ConversationFlow.created_at.desc())
    
    result = await db.execute(query)
    flows = result.scalars().all()
    
    return [
        FlowResponse(
            id=f.id,
            name=f.name,
            description=f.description,
            nodes=f.nodes,
            edges=f.edges,
            viewport=f.viewport,
            is_active=f.is_active,
            is_draft=f.is_draft,
            version=f.version,
            agent_id=f.agent_id,
            created_at=f.created_at,
            updated_at=f.updated_at,
        )
        for f in flows
    ]


@router.get("/{flow_id}", response_model=FlowResponse)
async def get_flow(
    flow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific conversation flow."""
    result = await db.execute(
        select(ConversationFlow).where(
            and_(
                ConversationFlow.id == flow_id,
                ConversationFlow.workspace_id == current_user.current_workspace_id,
                ConversationFlow.is_deleted == False,
            )
        )
    )
    flow = result.scalar_one_or_none()
    
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    
    return FlowResponse(
        id=flow.id,
        name=flow.name,
        description=flow.description,
        nodes=flow.nodes,
        edges=flow.edges,
        viewport=flow.viewport,
        is_active=flow.is_active,
        is_draft=flow.is_draft,
        version=flow.version,
        agent_id=flow.agent_id,
        created_at=flow.created_at,
        updated_at=flow.updated_at,
    )


@router.patch("/{flow_id}", response_model=FlowResponse)
async def update_flow(
    flow_id: UUID,
    data: FlowUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a conversation flow."""
    result = await db.execute(
        select(ConversationFlow).where(
            and_(
                ConversationFlow.id == flow_id,
                ConversationFlow.workspace_id == current_user.current_workspace_id,
                ConversationFlow.is_deleted == False,
            )
        )
    )
    flow = result.scalar_one_or_none()
    
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    
    # Update fields
    if data.name is not None:
        flow.name = data.name
    if data.description is not None:
        flow.description = data.description
    if data.nodes is not None:
        flow.nodes = data.nodes
    if data.edges is not None:
        flow.edges = data.edges
    if data.viewport is not None:
        flow.viewport = data.viewport
    if data.is_active is not None:
        flow.is_active = data.is_active
    if data.is_draft is not None:
        flow.is_draft = data.is_draft
    
    flow.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(flow)
    
    return FlowResponse(
        id=flow.id,
        name=flow.name,
        description=flow.description,
        nodes=flow.nodes,
        edges=flow.edges,
        viewport=flow.viewport,
        is_active=flow.is_active,
        is_draft=flow.is_draft,
        version=flow.version,
        agent_id=flow.agent_id,
        created_at=flow.created_at,
        updated_at=flow.updated_at,
    )


@router.delete("/{flow_id}")
async def delete_flow(
    flow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a conversation flow."""
    result = await db.execute(
        select(ConversationFlow).where(
            and_(
                ConversationFlow.id == flow_id,
                ConversationFlow.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    flow = result.scalar_one_or_none()
    
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    
    flow.is_deleted = True
    flow.updated_at = datetime.utcnow()
    await db.commit()
    
    return {"message": "Flow deleted successfully"}


@router.post("/{flow_id}/duplicate", response_model=FlowResponse)
async def duplicate_flow(
    flow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Duplicate a conversation flow."""
    result = await db.execute(
        select(ConversationFlow).where(
            and_(
                ConversationFlow.id == flow_id,
                ConversationFlow.workspace_id == current_user.current_workspace_id,
                ConversationFlow.is_deleted == False,
            )
        )
    )
    original = result.scalar_one_or_none()
    
    if not original:
        raise HTTPException(status_code=404, detail="Flow not found")
    
    # Create duplicate
    new_flow = ConversationFlow(
        name=f"{original.name} (Copy)",
        description=original.description,
        nodes=original.nodes,
        edges=original.edges,
        viewport=original.viewport,
        agent_id=original.agent_id,
        workspace_id=current_user.current_workspace_id,
        is_draft=True,
        is_active=False,
    )
    db.add(new_flow)
    await db.commit()
    await db.refresh(new_flow)
    
    return FlowResponse(
        id=new_flow.id,
        name=new_flow.name,
        description=new_flow.description,
        nodes=new_flow.nodes,
        edges=new_flow.edges,
        viewport=new_flow.viewport,
        is_active=new_flow.is_active,
        is_draft=new_flow.is_draft,
        version=new_flow.version,
        agent_id=new_flow.agent_id,
        created_at=new_flow.created_at,
        updated_at=new_flow.updated_at,
    )


@router.post("/{flow_id}/activate")
async def activate_flow(
    flow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Activate a flow and deactivate others for the same agent."""
    result = await db.execute(
        select(ConversationFlow).where(
            and_(
                ConversationFlow.id == flow_id,
                ConversationFlow.workspace_id == current_user.current_workspace_id,
                ConversationFlow.is_deleted == False,
            )
        )
    )
    flow = result.scalar_one_or_none()
    
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    
    # Deactivate other flows for the same agent
    other_flows = await db.execute(
        select(ConversationFlow).where(
            and_(
                ConversationFlow.agent_id == flow.agent_id,
                ConversationFlow.id != flow_id,
                ConversationFlow.is_active == True,
            )
        )
    )
    for other in other_flows.scalars().all():
        other.is_active = False
    
    # Activate this flow
    flow.is_active = True
    flow.is_draft = False
    flow.updated_at = datetime.utcnow()
    
    # Update agent's conversation_flow field
    agent_result = await db.execute(
        select(Agent).where(Agent.id == flow.agent_id)
    )
    agent = agent_result.scalar_one_or_none()
    if agent:
        agent.conversation_flow = {
            "flow_id": str(flow.id),
            "nodes": flow.nodes,
            "edges": flow.edges,
        }
    
    await db.commit()
    
    return {"message": "Flow activated successfully"}


# ============================================
# FLOW TEMPLATES
# ============================================

@router.get("/templates/list")
async def list_flow_templates(
    current_user: User = Depends(get_current_user),
):
    """Get pre-built flow templates."""
    templates = [
        {
            "id": "welcome-flow",
            "name": "Welcome Flow",
            "description": "Simple welcome and qualification flow",
            "category": "general",
            "nodes": [
                {"id": "start-1", "type": "start", "position": {"x": 250, "y": 50}, "data": {"label": "Start"}},
                {"id": "msg-1", "type": "message", "position": {"x": 250, "y": 150}, "data": {"label": "Welcome", "content": "Hello! Welcome to our service. How can I help you today?"}},
                {"id": "question-1", "type": "question", "position": {"x": 250, "y": 280}, "data": {"label": "Get Name", "content": "May I know your name?", "variable_name": "user_name", "expected_type": "text"}},
                {"id": "msg-2", "type": "message", "position": {"x": 250, "y": 410}, "data": {"label": "Personalized Greeting", "content": "Nice to meet you, {{user_name}}! What brings you here today?"}},
                {"id": "end-1", "type": "end", "position": {"x": 250, "y": 540}, "data": {"label": "End"}}
            ],
            "edges": [
                {"id": "e1", "source": "start-1", "target": "msg-1", "type": "smoothstep"},
                {"id": "e2", "source": "msg-1", "target": "question-1", "type": "smoothstep"},
                {"id": "e3", "source": "question-1", "target": "msg-2", "type": "smoothstep"},
                {"id": "e4", "source": "msg-2", "target": "end-1", "type": "smoothstep"}
            ]
        },
        {
            "id": "lead-qualification",
            "name": "Lead Qualification",
            "description": "Qualify leads with conditional branching",
            "category": "sales",
            "nodes": [
                {"id": "start-1", "type": "start", "position": {"x": 300, "y": 50}, "data": {"label": "Start"}},
                {"id": "msg-1", "type": "message", "position": {"x": 300, "y": 150}, "data": {"label": "Welcome", "content": "Hi! I'm here to help you find the right solution. Let me ask a few questions."}},
                {"id": "question-1", "type": "question", "position": {"x": 300, "y": 280}, "data": {"label": "Company Size", "content": "How many employees does your company have?", "variable_name": "company_size", "expected_type": "choice", "choices": ["1-10", "11-50", "51-200", "200+"]}},
                {"id": "condition-1", "type": "condition", "position": {"x": 300, "y": 410}, "data": {"label": "Check Size", "conditions": [{"variable": "company_size", "operator": "equals", "value": "200+", "label": "Enterprise"}]}},
                {"id": "msg-enterprise", "type": "message", "position": {"x": 100, "y": 540}, "data": {"label": "Enterprise Path", "content": "Great! For enterprise clients, we offer dedicated support. Let me connect you with our enterprise team."}},
                {"id": "msg-standard", "type": "message", "position": {"x": 500, "y": 540}, "data": {"label": "Standard Path", "content": "Perfect! Our standard plans would be great for your team size."}},
                {"id": "end-1", "type": "end", "position": {"x": 300, "y": 670}, "data": {"label": "End"}}
            ],
            "edges": [
                {"id": "e1", "source": "start-1", "target": "msg-1", "type": "smoothstep"},
                {"id": "e2", "source": "msg-1", "target": "question-1", "type": "smoothstep"},
                {"id": "e3", "source": "question-1", "target": "condition-1", "type": "smoothstep"},
                {"id": "e4", "source": "condition-1", "target": "msg-enterprise", "sourceHandle": "yes", "type": "smoothstep", "label": "Enterprise"},
                {"id": "e5", "source": "condition-1", "target": "msg-standard", "sourceHandle": "no", "type": "smoothstep", "label": "Other"},
                {"id": "e6", "source": "msg-enterprise", "target": "end-1", "type": "smoothstep"},
                {"id": "e7", "source": "msg-standard", "target": "end-1", "type": "smoothstep"}
            ]
        },
        {
            "id": "appointment-booking",
            "name": "Appointment Booking",
            "description": "Book appointments with availability check",
            "category": "booking",
            "nodes": [
                {"id": "start-1", "type": "start", "position": {"x": 250, "y": 50}, "data": {"label": "Start"}},
                {"id": "msg-1", "type": "message", "position": {"x": 250, "y": 150}, "data": {"label": "Greeting", "content": "Hello! I can help you book an appointment. What service are you interested in?"}},
                {"id": "question-1", "type": "question", "position": {"x": 250, "y": 280}, "data": {"label": "Service", "content": "Please select a service:", "variable_name": "service", "expected_type": "choice", "choices": ["Consultation", "Demo", "Support Call", "Other"]}},
                {"id": "question-2", "type": "question", "position": {"x": 250, "y": 410}, "data": {"label": "Preferred Date", "content": "What date works best for you?", "variable_name": "preferred_date", "expected_type": "text"}},
                {"id": "question-3", "type": "question", "position": {"x": 250, "y": 540}, "data": {"label": "Email", "content": "Please provide your email for confirmation:", "variable_name": "email", "expected_type": "text"}},
                {"id": "action-1", "type": "action", "position": {"x": 250, "y": 670}, "data": {"label": "Book Appointment", "action_type": "webhook", "action_config": {"url": "{{webhook_url}}", "method": "POST"}}},
                {"id": "msg-2", "type": "message", "position": {"x": 250, "y": 800}, "data": {"label": "Confirmation", "content": "Your {{service}} appointment has been requested for {{preferred_date}}. You'll receive a confirmation at {{email}}."}},
                {"id": "end-1", "type": "end", "position": {"x": 250, "y": 930}, "data": {"label": "End"}}
            ],
            "edges": [
                {"id": "e1", "source": "start-1", "target": "msg-1", "type": "smoothstep"},
                {"id": "e2", "source": "msg-1", "target": "question-1", "type": "smoothstep"},
                {"id": "e3", "source": "question-1", "target": "question-2", "type": "smoothstep"},
                {"id": "e4", "source": "question-2", "target": "question-3", "type": "smoothstep"},
                {"id": "e5", "source": "question-3", "target": "action-1", "type": "smoothstep"},
                {"id": "e6", "source": "action-1", "target": "msg-2", "type": "smoothstep"},
                {"id": "e7", "source": "msg-2", "target": "end-1", "type": "smoothstep"}
            ]
        },
        {
            "id": "faq-flow",
            "name": "FAQ Handler",
            "description": "Handle common questions with branching",
            "category": "support",
            "nodes": [
                {"id": "start-1", "type": "start", "position": {"x": 300, "y": 50}, "data": {"label": "Start"}},
                {"id": "question-1", "type": "question", "position": {"x": 300, "y": 150}, "data": {"label": "Topic", "content": "What would you like to know about?", "variable_name": "topic", "expected_type": "choice", "choices": ["Pricing", "Features", "Support", "Other"]}},
                {"id": "condition-1", "type": "condition", "position": {"x": 300, "y": 280}, "data": {"label": "Route Topic", "conditions": [{"variable": "topic", "operator": "equals", "value": "Pricing", "label": "Pricing"}, {"variable": "topic", "operator": "equals", "value": "Features", "label": "Features"}]}},
                {"id": "msg-pricing", "type": "message", "position": {"x": 50, "y": 410}, "data": {"label": "Pricing Info", "content": "Our plans start at $29/month. We offer Basic, Pro, and Enterprise tiers. Would you like more details?"}},
                {"id": "msg-features", "type": "message", "position": {"x": 300, "y": 410}, "data": {"label": "Features Info", "content": "Our key features include AI agents, voice calls, chat integration, and analytics. What specific feature interests you?"}},
                {"id": "msg-other", "type": "message", "position": {"x": 550, "y": 410}, "data": {"label": "Other", "content": "I'd be happy to help! Please describe your question and I'll do my best to assist."}},
                {"id": "end-1", "type": "end", "position": {"x": 300, "y": 540}, "data": {"label": "End"}}
            ],
            "edges": [
                {"id": "e1", "source": "start-1", "target": "question-1", "type": "smoothstep"},
                {"id": "e2", "source": "question-1", "target": "condition-1", "type": "smoothstep"},
                {"id": "e3", "source": "condition-1", "target": "msg-pricing", "sourceHandle": "condition-0", "type": "smoothstep", "label": "Pricing"},
                {"id": "e4", "source": "condition-1", "target": "msg-features", "sourceHandle": "condition-1", "type": "smoothstep", "label": "Features"},
                {"id": "e5", "source": "condition-1", "target": "msg-other", "sourceHandle": "default", "type": "smoothstep", "label": "Other"},
                {"id": "e6", "source": "msg-pricing", "target": "end-1", "type": "smoothstep"},
                {"id": "e7", "source": "msg-features", "target": "end-1", "type": "smoothstep"},
                {"id": "e8", "source": "msg-other", "target": "end-1", "type": "smoothstep"}
            ]
        }
    ]
    
    return {"templates": templates}


@router.post("/templates/{template_id}/use")
async def use_flow_template(
    template_id: str,
    agent_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new flow from a template."""
    # Get templates
    templates_response = await list_flow_templates(current_user)
    templates = templates_response["templates"]
    
    template = next((t for t in templates if t["id"] == template_id), None)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    # Verify agent exists
    agent_result = await db.execute(
        select(Agent).where(
            and_(
                Agent.id == agent_id,
                Agent.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    agent = agent_result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Create flow from template
    flow = ConversationFlow(
        name=template["name"],
        description=template["description"],
        nodes=template["nodes"],
        edges=template["edges"],
        agent_id=agent_id,
        workspace_id=current_user.current_workspace_id,
    )
    db.add(flow)
    await db.commit()
    await db.refresh(flow)
    
    return {
        "id": flow.id,
        "name": flow.name,
        "message": "Flow created from template"
    }