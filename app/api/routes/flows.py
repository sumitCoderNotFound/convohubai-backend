"""
ConvoHubAI - Conversation Flow API Routes
CRUD operations for visual workflow builder + AI Flow Generator
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel
import json

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.agent import Agent
from app.models.conversation_flow import ConversationFlow
from app.services.llm_service import llm_service


router = APIRouter(prefix="/flows", tags=["Conversation Flows"])


# ============================================
# SCHEMAS
# ============================================

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


class AIFlowGenerateRequest(BaseModel):
    """Request to generate a flow using AI."""
    prompt: str
    agent_id: UUID
    flow_name: Optional[str] = None


# ============================================
# AI FLOW GENERATOR SYSTEM PROMPT
# ============================================

AI_FLOW_SYSTEM_PROMPT = """You are an expert conversation flow designer. Convert natural language descriptions into structured conversation flows.

Generate a JSON object with:
1. "name": Flow name (string)
2. "description": Brief description (string)
3. "nodes": Array of node objects
4. "edges": Array of edge objects connecting nodes

NODE TYPES:
- "start": Entry point (always first, only one)
- "message": Display a message to user
- "question": Ask user a question and save response
- "condition": Branch based on user's answer
- "action": Perform an action (save_lead, send_email, book_appointment, webhook)
- "end": End the conversation (always last)

NODE STRUCTURE:
{
  "id": "unique-id",
  "type": "start|message|question|condition|action|end",
  "position": {"x": number, "y": number},
  "data": {
    "label": "Node label",
    "message": "For message nodes - the text to display",
    "question": "For question nodes - the question to ask",
    "options": ["Option 1", "Option 2"], // For multiple choice questions
    "inputType": "text|email|tel|date|textarea", // For open questions
    "saveAs": "variable_name", // Variable to save user's answer
    "condition": "variable_name", // For condition nodes - which variable to check
    "actionType": "save_lead|send_email|book_appointment|webhook", // For action nodes
    "description": "What this node does"
  }
}

EDGE STRUCTURE:
{
  "id": "edge-id",
  "source": "source-node-id",
  "target": "target-node-id",
  "sourceHandle": "branch-0|branch-1|default", // For condition branches
  "label": "Optional label for the edge"
}

POSITIONING RULES:
- Start node at y=50
- Each row increment y by 120
- Single path: x=400
- Two branches: x=200 and x=600
- Three branches: x=100, x=400, x=700
- Always center the flow

BEST PRACTICES:
- Always start with "start" node and end with "end" node
- Use emojis in messages to be friendly (🎉 ✅ 👋 📧 📞 🌍)
- Keep messages concise but helpful
- Use meaningful variable names (email, phone, name, budget, etc.)
- Every node must be connected via edges
- Conditions must have branches for different outcomes

Return ONLY valid JSON, no markdown code blocks, no explanations."""


# ============================================
# STATIC ROUTES FIRST (before dynamic {flow_id})
# ============================================

@router.post("/generate")
async def generate_flow_with_ai(
    data: AIFlowGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a conversation flow using AI from natural language prompt."""
    
    # Verify agent exists
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
    
    # Build prompt for AI
    user_prompt = f"""Create a conversation flow for:

{data.prompt}

Agent: {agent.name}
Agent Description: {agent.description or 'AI Assistant'}

Generate a complete flow with proper positioning. Return only valid JSON."""

    try:
        # Call LLM to generate flow
        response = await llm_service.generate_response(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=AI_FLOW_SYSTEM_PROMPT,
            model=agent.llm_model or "llama-3.3-70b-versatile",
            provider=agent.llm_provider or "groq",
            temperature=0.2,
            max_tokens=4000,
        )
        
        # Clean up response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        # Parse JSON
        try:
            flow_data = json.loads(response)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                flow_data = json.loads(json_match.group())
            else:
                raise HTTPException(status_code=500, detail="Failed to parse AI response")
        
        # Validate
        if "nodes" not in flow_data or "edges" not in flow_data:
            raise HTTPException(status_code=500, detail="Invalid flow structure")
        
        # Create flow name
        flow_name = data.flow_name or flow_data.get("name", f"AI Flow - {datetime.utcnow().strftime('%H:%M')}")
        
        # Save to database
        flow = ConversationFlow(
            name=flow_name,
            description=flow_data.get("description", data.prompt[:200]),
            nodes=flow_data["nodes"],
            edges=flow_data["edges"],
            agent_id=data.agent_id,
            workspace_id=current_user.current_workspace_id,
            is_draft=True,
        )
        db.add(flow)
        await db.commit()
        await db.refresh(flow)
        
        return {
            "id": flow.id,
            "name": flow.name,
            "description": flow.description,
            "nodes": flow.nodes,
            "edges": flow.edges,
            "message": "✨ Flow generated successfully!",
            "node_count": len(flow.nodes),
            "edge_count": len(flow.edges),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate flow: {str(e)}")


@router.post("/generate/preview")
async def preview_generated_flow(
    data: AIFlowGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Preview a flow without saving."""
    
    agent_result = await db.execute(
        select(Agent).where(
            and_(
                Agent.id == data.agent_id,
                Agent.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    agent = agent_result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    user_prompt = f"""Create a conversation flow for:

{data.prompt}

Agent: {agent.name}

Return only valid JSON."""

    try:
        response = await llm_service.generate_response(
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=AI_FLOW_SYSTEM_PROMPT,
            model=agent.llm_model or "llama-3.3-70b-versatile",
            provider=agent.llm_provider or "groq",
            temperature=0.2,
            max_tokens=4000,
        )
        
        response = response.strip()
        if "```" in response:
            response = response.replace("```json", "").replace("```", "").strip()
        
        flow_data = json.loads(response)
        
        return {
            "name": flow_data.get("name", "Generated Flow"),
            "description": flow_data.get("description", ""),
            "nodes": flow_data.get("nodes", []),
            "edges": flow_data.get("edges", []),
            "preview": True,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
                {"id": "start-1", "type": "start", "position": {"x": 400, "y": 50}, "data": {"label": "Start"}},
                {"id": "msg-1", "type": "message", "position": {"x": 400, "y": 170}, "data": {"label": "Welcome", "message": "Hello! 👋 Welcome to our service."}},
                {"id": "end-1", "type": "end", "position": {"x": 400, "y": 290}, "data": {"label": "End"}}
            ],
            "edges": [
                {"id": "e1", "source": "start-1", "target": "msg-1"},
                {"id": "e2", "source": "msg-1", "target": "end-1"}
            ]
        },
        {
            "id": "lead-capture",
            "name": "Lead Capture",
            "description": "Capture visitor information",
            "category": "sales",
            "nodes": [
                {"id": "start-1", "type": "start", "position": {"x": 400, "y": 50}, "data": {"label": "Start"}},
                {"id": "msg-1", "type": "message", "position": {"x": 400, "y": 170}, "data": {"label": "Welcome", "message": "Hi! 👋 Let me help you get started."}},
                {"id": "q-1", "type": "question", "position": {"x": 400, "y": 290}, "data": {"label": "Get Email", "question": "What's your email address?", "inputType": "email", "saveAs": "email"}},
                {"id": "q-2", "type": "question", "position": {"x": 400, "y": 410}, "data": {"label": "Get Phone", "question": "And your phone number?", "inputType": "tel", "saveAs": "phone"}},
                {"id": "action-1", "type": "action", "position": {"x": 400, "y": 530}, "data": {"label": "Save Lead", "actionType": "save_lead"}},
                {"id": "msg-2", "type": "message", "position": {"x": 400, "y": 650}, "data": {"label": "Thanks", "message": "Thanks! 🎉 We'll be in touch soon."}},
                {"id": "end-1", "type": "end", "position": {"x": 400, "y": 770}, "data": {"label": "End"}}
            ],
            "edges": [
                {"id": "e1", "source": "start-1", "target": "msg-1"},
                {"id": "e2", "source": "msg-1", "target": "q-1"},
                {"id": "e3", "source": "q-1", "target": "q-2"},
                {"id": "e4", "source": "q-2", "target": "action-1"},
                {"id": "e5", "source": "action-1", "target": "msg-2"},
                {"id": "e6", "source": "msg-2", "target": "end-1"}
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
    templates_response = await list_flow_templates(current_user)
    templates = templates_response["templates"]
    
    template = next((t for t in templates if t["id"] == template_id), None)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
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


# ============================================
# CRUD ROUTES (List and Create first)
# ============================================

@router.get("")
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
    
    result = await db.execute(query.order_by(ConversationFlow.created_at.desc()))
    flows = result.scalars().all()
    
    return [
        {
            "id": f.id,
            "name": f.name,
            "description": f.description,
            "nodes": f.nodes or [],
            "edges": f.edges or [],
            "viewport": f.viewport or {"x": 0, "y": 0, "zoom": 1},
            "is_active": f.is_active,
            "is_draft": f.is_draft,
            "version": f.version or "1.0",
            "agent_id": f.agent_id,
            "created_at": f.created_at,
            "updated_at": f.updated_at,
        }
        for f in flows
    ]


@router.post("")
async def create_flow(
    data: FlowCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new conversation flow."""
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
    
    default_nodes = data.nodes if data.nodes else [
        {"id": "start-1", "type": "start", "position": {"x": 400, "y": 50}, "data": {"label": "Start"}}
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
    
    return {
        "id": flow.id,
        "name": flow.name,
        "description": flow.description,
        "nodes": flow.nodes,
        "edges": flow.edges,
        "viewport": flow.viewport or {"x": 0, "y": 0, "zoom": 1},
        "is_active": flow.is_active,
        "is_draft": flow.is_draft,
        "version": flow.version or "1.0",
        "agent_id": flow.agent_id,
        "created_at": flow.created_at,
        "updated_at": flow.updated_at,
    }


# ============================================
# DYNAMIC ROUTES (with {flow_id}) - MUST BE LAST
# ============================================

@router.get("/{flow_id}")
async def get_flow(
    flow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific flow."""
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
    
    return {
        "id": flow.id,
        "name": flow.name,
        "description": flow.description,
        "nodes": flow.nodes or [],
        "edges": flow.edges or [],
        "viewport": flow.viewport or {"x": 0, "y": 0, "zoom": 1},
        "is_active": flow.is_active,
        "is_draft": flow.is_draft,
        "version": flow.version or "1.0",
        "agent_id": flow.agent_id,
        "created_at": flow.created_at,
        "updated_at": flow.updated_at,
    }


@router.patch("/{flow_id}")
async def update_flow(
    flow_id: UUID,
    data: FlowUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a flow."""
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
    
    return {
        "id": flow.id,
        "name": flow.name,
        "description": flow.description,
        "nodes": flow.nodes or [],
        "edges": flow.edges or [],
        "viewport": flow.viewport or {"x": 0, "y": 0, "zoom": 1},
        "is_active": flow.is_active,
        "is_draft": flow.is_draft,
        "version": flow.version or "1.0",
        "agent_id": flow.agent_id,
        "created_at": flow.created_at,
        "updated_at": flow.updated_at,
    }


@router.delete("/{flow_id}")
async def delete_flow(
    flow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a flow."""
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
    
    return {"message": "Flow deleted"}


@router.post("/{flow_id}/duplicate")
async def duplicate_flow(
    flow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Duplicate a flow."""
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
    
    new_flow = ConversationFlow(
        name=f"{original.name} (Copy)",
        description=original.description,
        nodes=original.nodes,
        edges=original.edges,
        viewport=original.viewport,
        agent_id=original.agent_id,
        workspace_id=current_user.current_workspace_id,
        is_draft=True,
    )
    db.add(new_flow)
    await db.commit()
    await db.refresh(new_flow)
    
    return {
        "id": new_flow.id,
        "name": new_flow.name,
        "description": new_flow.description,
        "nodes": new_flow.nodes or [],
        "edges": new_flow.edges or [],
        "viewport": new_flow.viewport or {"x": 0, "y": 0, "zoom": 1},
        "is_active": new_flow.is_active,
        "is_draft": new_flow.is_draft,
        "version": new_flow.version or "1.0",
        "agent_id": new_flow.agent_id,
        "created_at": new_flow.created_at,
        "updated_at": new_flow.updated_at,
    }


@router.post("/{flow_id}/activate")
async def activate_flow(
    flow_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Activate a flow."""
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
    
    # Deactivate other flows
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
    
    flow.is_active = True
    flow.is_draft = False
    flow.updated_at = datetime.utcnow()
    
    # Update agent
    agent_result = await db.execute(select(Agent).where(Agent.id == flow.agent_id))
    agent = agent_result.scalar_one_or_none()
    if agent:
        agent.conversation_flow = {"flow_id": str(flow.id), "nodes": flow.nodes, "edges": flow.edges}
    
    await db.commit()
    
    return {"message": "Flow activated"}