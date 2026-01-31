"""
ConvoHubAI - Simulation API Routes
Handles AI-simulated conversations for testing agents
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel
import time

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.agent import Agent
from app.services.llm_service import llm_service


router = APIRouter(prefix="/simulation", tags=["Simulation"])


# ============================================
# SCHEMAS
# ============================================

class SimulationRequest(BaseModel):
    agent_id: UUID
    user_persona: str = "You are a customer who wants to learn more about the product."
    num_turns: int = 5
    scenario: Optional[str] = None


class SimulationMessage(BaseModel):
    role: str
    content: str
    timestamp: str


class SimulationResponse(BaseModel):
    agent_id: UUID
    agent_name: str
    messages: List[SimulationMessage]
    total_turns: int
    duration_seconds: float


class SingleTurnSimulationRequest(BaseModel):
    agent_id: UUID
    user_persona: str = "You are a helpful customer."
    context: Optional[List[dict]] = None


# ============================================
# ROUTES
# ============================================

@router.post("/run", response_model=SimulationResponse)
async def run_simulation(
    data: SimulationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Run a full AI-simulated conversation."""
    start_time = time.time()
    
    # Get agent
    result = await db.execute(
        select(Agent).where(
            and_(
                Agent.id == data.agent_id,
                Agent.workspace_id == current_user.current_workspace_id,
                Agent.is_deleted == False,
            )
        )
    )
    agent = result.scalar_one_or_none()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    messages = []
    conversation_history = []
    
    user_simulation_prompt = f"""You are simulating a user in a conversation. 
Your persona: {data.user_persona}
{f'Scenario: {data.scenario}' if data.scenario else ''}

Guidelines:
- Stay in character as the user persona
- Ask realistic questions
- Respond naturally to the agent's messages
- Keep responses concise (1-3 sentences)
- If the conversation naturally concludes, say "[END]"

Respond ONLY as the user would. Do not include any explanations."""

    # Start with welcome message if available
    if agent.welcome_message:
        messages.append(SimulationMessage(
            role="assistant",
            content=agent.welcome_message,
            timestamp=datetime.utcnow().isoformat()
        ))
        conversation_history.append({
            "role": "assistant",
            "content": agent.welcome_message
        })
    
    # Run conversation turns
    for turn in range(data.num_turns):
        try:
            # Generate simulated user message
            user_prompt_messages = [
                {"role": "user", "content": f"Based on the conversation so far, generate the next user message.\n\nConversation:\n{_format_conversation(conversation_history)}"}
            ]
            
            simulated_user_message = await llm_service.generate_response(
                messages=user_prompt_messages,
                system_prompt=user_simulation_prompt,
                model="gpt-4o-mini",
                provider="openai",
                temperature=0.8,
                max_tokens=200,
            )
            
            if "[END]" in simulated_user_message:
                simulated_user_message = simulated_user_message.replace("[END]", "").strip()
                if simulated_user_message:
                    messages.append(SimulationMessage(
                        role="user",
                        content=simulated_user_message,
                        timestamp=datetime.utcnow().isoformat()
                    ))
                break
            
            messages.append(SimulationMessage(
                role="user",
                content=simulated_user_message,
                timestamp=datetime.utcnow().isoformat()
            ))
            conversation_history.append({
                "role": "user",
                "content": simulated_user_message
            })
            
            # Generate agent response
            agent_response = await llm_service.generate_response(
                messages=conversation_history,
                system_prompt=agent.system_prompt or "You are a helpful assistant.",
                model=agent.llm_model or "gpt-4o-mini",
                provider=agent.llm_provider or "openai",
                temperature=agent.temperature or 0.7,
                max_tokens=int(agent.max_tokens) if agent.max_tokens else 500,
            )
            
            messages.append(SimulationMessage(
                role="assistant",
                content=agent_response,
                timestamp=datetime.utcnow().isoformat()
            ))
            conversation_history.append({
                "role": "assistant",
                "content": agent_response
            })
            
        except Exception as e:
            print(f"Simulation turn {turn} error: {e}")
            break
    
    duration = time.time() - start_time
    
    return SimulationResponse(
        agent_id=agent.id,
        agent_name=agent.name,
        messages=messages,
        total_turns=len([m for m in messages if m.role == "user"]),
        duration_seconds=round(duration, 2)
    )


@router.post("/single-turn")
async def simulate_single_turn(
    data: SingleTurnSimulationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a single simulated user message."""
    result = await db.execute(
        select(Agent).where(
            and_(
                Agent.id == data.agent_id,
                Agent.workspace_id == current_user.current_workspace_id,
                Agent.is_deleted == False,
            )
        )
    )
    agent = result.scalar_one_or_none()
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    user_simulation_prompt = f"""You are simulating a user in a conversation. 
Your persona: {data.user_persona}

Generate a single realistic user message based on the conversation context.
Keep it concise (1-3 sentences).
Respond ONLY with the user's message, nothing else."""

    context = data.context or []
    context_str = _format_conversation(context) if context else "This is the start of the conversation."
    
    try:
        simulated_message = await llm_service.generate_response(
            messages=[{"role": "user", "content": f"Conversation so far:\n{context_str}\n\nGenerate the next user message:"}],
            system_prompt=user_simulation_prompt,
            model="gpt-4o-mini",
            provider="openai",
            temperature=0.8,
            max_tokens=200,
        )
        
        return {
            "message": simulated_message.strip(),
            "persona": data.user_persona
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate simulated message: {str(e)}"
        )


@router.get("/personas")
async def get_simulation_personas(
    current_user: User = Depends(get_current_user),
):
    """Get a list of pre-built user personas for simulation."""
    personas = [
        {
            "id": "curious_customer",
            "name": "Curious Customer",
            "description": "A customer who asks many questions to understand the product better",
            "prompt": "You are a curious potential customer who wants to learn everything about the product before making a decision. Ask detailed questions."
        },
        {
            "id": "frustrated_user",
            "name": "Frustrated User",
            "description": "A user experiencing issues and seeking support",
            "prompt": "You are a frustrated user who is having problems with the service. You want quick solutions and may express some impatience."
        },
        {
            "id": "quick_buyer",
            "name": "Quick Buyer",
            "description": "A customer ready to purchase with minimal questions",
            "prompt": "You are a customer who already knows what you want. You want to complete your purchase quickly with minimal back-and-forth."
        },
        {
            "id": "comparison_shopper",
            "name": "Comparison Shopper",
            "description": "A customer comparing options",
            "prompt": "You are comparing this product with competitors. Ask about pricing, features, and what makes this option better than alternatives."
        },
        {
            "id": "technical_user",
            "name": "Technical User",
            "description": "A tech-savvy user with detailed technical questions",
            "prompt": "You are a technical user who wants to understand the implementation details, APIs, and technical specifications."
        },
        {
            "id": "first_time_user",
            "name": "First-Time User",
            "description": "A new user unfamiliar with the product",
            "prompt": "You are using this product for the first time. Ask basic questions and request guidance on getting started."
        },
    ]
    
    return {"personas": personas}


@router.get("/scenarios")
async def get_simulation_scenarios(
    current_user: User = Depends(get_current_user),
):
    """Get a list of pre-built scenarios for simulation."""
    scenarios = [
        {"id": "product_inquiry", "name": "Product Inquiry", "description": "Customer asking about product features and pricing"},
        {"id": "support_request", "name": "Support Request", "description": "User needs help with a technical issue"},
        {"id": "booking_appointment", "name": "Booking Appointment", "description": "Customer wants to schedule an appointment"},
        {"id": "order_status", "name": "Order Status", "description": "Customer checking on their order"},
        {"id": "complaint", "name": "Complaint Handling", "description": "Customer has a complaint to resolve"},
        {"id": "cancellation", "name": "Cancellation Request", "description": "Customer wants to cancel their subscription"},
    ]
    
    return {"scenarios": scenarios}


def _format_conversation(messages: List[dict]) -> str:
    """Format conversation history as a string."""
    if not messages:
        return "No messages yet."
    
    formatted = []
    for msg in messages:
        role = "Agent" if msg["role"] == "assistant" else "User"
        formatted.append(f"{role}: {msg['content']}")
    
    return "\n".join(formatted)