"""
ConvoHubAI - Voice API Routes
Handles voice calls, Twilio webhooks, and phone number management
"""
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel
import json

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.agent import Agent
from app.models.phone_number import PhoneNumber
from app.models.call import Call, CallEvent, CallDirection, CallStatus
from app.services.voice_service import voice_service
from app.services.llm_service import llm_service


router = APIRouter(prefix="/voice", tags=["Voice"])


# ============================================
# SCHEMAS
# ============================================

class PhoneNumberSearch(BaseModel):
    country: str = "US"
    area_code: Optional[str] = None


class PhoneNumberPurchase(BaseModel):
    phone_number: str
    friendly_name: Optional[str] = None
    agent_id: Optional[UUID] = None


class PhoneNumberAssign(BaseModel):
    agent_id: UUID


class OutboundCallRequest(BaseModel):
    to_number: str
    agent_id: UUID
    phone_number_id: UUID


class CallResponse(BaseModel):
    id: UUID
    call_sid: str
    direction: str
    status: str
    from_number: str
    to_number: str
    duration_seconds: int
    started_at: Optional[str]
    ended_at: Optional[str]
    recording_url: Optional[str]
    transcript: Optional[str]
    agent_id: Optional[UUID]
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============================================
# PHONE NUMBER MANAGEMENT
# ============================================

@router.get("/phone-numbers/available")
async def search_available_numbers(
    country: str = "US",
    area_code: Optional[str] = None,
    current_user: User = Depends(get_current_user),
):
    """Search for available phone numbers to purchase."""
    try:
        numbers = await voice_service.list_available_numbers(country, area_code)
        return {
            "numbers": [
                {
                    "phone_number": n.get("phone_number"),
                    "friendly_name": n.get("friendly_name"),
                    "locality": n.get("locality"),
                    "region": n.get("region"),
                    "capabilities": n.get("capabilities", {}),
                }
                for n in numbers[:20]  # Limit to 20 results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/phone-numbers/purchase")
async def purchase_phone_number(
    data: PhoneNumberPurchase,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Purchase a phone number from Twilio."""
    # Get base URL for webhooks (you'll need to set this in env)
    import os
    base_url = os.getenv("APP_BASE_URL", "https://your-domain.com")
    webhook_url = f"{base_url}/api/v1/voice/webhook/incoming"
    
    try:
        # Purchase from Twilio
        twilio_response = await voice_service.buy_phone_number(
            data.phone_number, 
            webhook_url
        )
        
        # Save to database
        phone_number = PhoneNumber(
            phone_number=data.phone_number,
            friendly_name=data.friendly_name or twilio_response.get("friendly_name"),
            phone_sid=twilio_response.get("sid"),
            provider="twilio",
            capabilities={
                "voice": True,
                "sms": twilio_response.get("capabilities", {}).get("sms", False),
            },
            status="active",
            agent_id=data.agent_id,
            workspace_id=current_user.current_workspace_id,
        )
        db.add(phone_number)
        await db.commit()
        await db.refresh(phone_number)
        
        return {
            "id": phone_number.id,
            "phone_number": phone_number.phone_number,
            "friendly_name": phone_number.friendly_name,
            "status": phone_number.status,
            "message": "Phone number purchased successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AddExistingNumber(BaseModel):
    phone_number: str
    friendly_name: Optional[str] = None
    agent_id: Optional[UUID] = None


@router.post("/phone-numbers/add-existing")
async def add_existing_phone_number(
    data: AddExistingNumber,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Add an existing phone number (already owned in Twilio) to the database."""
    try:
        # Check if number already exists
        existing = await db.execute(
            select(PhoneNumber).where(
                and_(
                    PhoneNumber.phone_number == data.phone_number,
                    PhoneNumber.workspace_id == current_user.current_workspace_id,
                    PhoneNumber.is_deleted == False,
                )
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Phone number already exists")
        
        # Create phone number record
        phone_number = PhoneNumber(
            phone_number=data.phone_number,
            friendly_name=data.friendly_name or "Voice Line",
            provider="twilio",
            status="active",
            capabilities={"voice": True, "sms": True},
            agent_id=data.agent_id,
            workspace_id=current_user.current_workspace_id,
        )
        db.add(phone_number)
        await db.commit()
        await db.refresh(phone_number)
        
        return {
            "id": phone_number.id,
            "phone_number": phone_number.phone_number,
            "friendly_name": phone_number.friendly_name,
            "status": phone_number.status,
            "message": "Phone number added successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/phone-numbers")
async def list_phone_numbers(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all owned phone numbers."""
    result = await db.execute(
        select(PhoneNumber).where(
            and_(
                PhoneNumber.workspace_id == current_user.current_workspace_id,
                PhoneNumber.is_deleted == False,
            )
        ).order_by(PhoneNumber.created_at.desc())
    )
    phone_numbers = result.scalars().all()
    
    return {
        "phone_numbers": [
            {
                "id": pn.id,
                "phone_number": pn.phone_number,
                "friendly_name": pn.friendly_name,
                "status": pn.status,
                "agent_id": pn.agent_id,
                "capabilities": pn.capabilities,
                "total_calls": pn.total_calls,
                "created_at": pn.created_at,
            }
            for pn in phone_numbers
        ]
    }


@router.patch("/phone-numbers/{phone_number_id}/assign")
async def assign_phone_number_to_agent(
    phone_number_id: UUID,
    data: PhoneNumberAssign,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Assign a phone number to an agent."""
    # Get phone number
    result = await db.execute(
        select(PhoneNumber).where(
            and_(
                PhoneNumber.id == phone_number_id,
                PhoneNumber.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    phone_number = result.scalar_one_or_none()
    if not phone_number:
        raise HTTPException(status_code=404, detail="Phone number not found")
    
    # Verify agent exists
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
    
    phone_number.agent_id = data.agent_id
    phone_number.updated_at = datetime.utcnow()
    await db.commit()
    
    return {"message": f"Phone number assigned to agent '{agent.name}'"}


@router.delete("/phone-numbers/{phone_number_id}")
async def release_phone_number(
    phone_number_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Release a phone number."""
    result = await db.execute(
        select(PhoneNumber).where(
            and_(
                PhoneNumber.id == phone_number_id,
                PhoneNumber.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    phone_number = result.scalar_one_or_none()
    if not phone_number:
        raise HTTPException(status_code=404, detail="Phone number not found")
    
    try:
        # Release from Twilio
        if phone_number.phone_sid:
            await voice_service.release_phone_number(phone_number.phone_sid)
        
        # Mark as deleted
        phone_number.is_deleted = True
        phone_number.status = "released"
        phone_number.updated_at = datetime.utcnow()
        await db.commit()
        
        return {"message": "Phone number released successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# OUTBOUND CALLS
# ============================================

@router.post("/calls/outbound")
async def make_outbound_call(
    data: OutboundCallRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Initiate an outbound call."""
    # Get phone number
    pn_result = await db.execute(
        select(PhoneNumber).where(
            and_(
                PhoneNumber.id == data.phone_number_id,
                PhoneNumber.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    phone_number = pn_result.scalar_one_or_none()
    if not phone_number:
        raise HTTPException(status_code=404, detail="Phone number not found")
    
    # Get agent
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
    
    import os
    base_url = os.getenv("APP_BASE_URL", "https://your-domain.com")
    webhook_url = f"{base_url}/api/v1/voice/webhook/outbound/{agent.id}"
    
    try:
        # Make call via Twilio
        twilio_response = await voice_service.make_call(
            to_number=data.to_number,
            from_number=phone_number.phone_number,
            webhook_url=webhook_url
        )
        
        # Create call record
        call = Call(
            call_sid=twilio_response.get("sid"),
            direction=CallDirection.OUTBOUND,
            status=CallStatus.INITIATED,
            from_number=phone_number.phone_number,
            to_number=data.to_number,
            started_at=datetime.utcnow().isoformat(),
            agent_id=agent.id,
            phone_number_id=phone_number.id,
            workspace_id=current_user.current_workspace_id,
        )
        db.add(call)
        await db.commit()
        await db.refresh(call)
        
        return {
            "call_id": call.id,
            "call_sid": call.call_sid,
            "status": call.status.value,
            "message": "Call initiated successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calls/{call_id}/end")
async def end_call(
    call_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """End an active call."""
    result = await db.execute(
        select(Call).where(
            and_(
                Call.id == call_id,
                Call.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    call = result.scalar_one_or_none()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    try:
        await voice_service.end_call(call.call_sid)
        
        call.status = CallStatus.COMPLETED
        call.ended_at = datetime.utcnow().isoformat()
        await db.commit()
        
        return {"message": "Call ended successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# CALL HISTORY
# ============================================

@router.get("/calls", response_model=List[CallResponse])
async def list_calls(
    agent_id: Optional[UUID] = None,
    direction: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List call history."""
    query = select(Call).where(
        and_(
            Call.workspace_id == current_user.current_workspace_id,
            Call.is_deleted == False,
        )
    )
    
    if agent_id:
        query = query.where(Call.agent_id == agent_id)
    if direction:
        query = query.where(Call.direction == direction)
    if status:
        query = query.where(Call.status == status)
    
    query = query.order_by(desc(Call.created_at)).limit(limit).offset(offset)
    
    result = await db.execute(query)
    calls = result.scalars().all()
    
    return [
        CallResponse(
            id=c.id,
            call_sid=c.call_sid,
            direction=c.direction.value,
            status=c.status.value,
            from_number=c.from_number,
            to_number=c.to_number,
            duration_seconds=c.duration_seconds,
            started_at=c.started_at,
            ended_at=c.ended_at,
            recording_url=c.recording_url,
            transcript=c.transcript,
            agent_id=c.agent_id,
            created_at=c.created_at,
        )
        for c in calls
    ]


@router.get("/calls/{call_id}")
async def get_call(
    call_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get call details including transcript."""
    result = await db.execute(
        select(Call).where(
            and_(
                Call.id == call_id,
                Call.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    call = result.scalar_one_or_none()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    
    return {
        "id": call.id,
        "call_sid": call.call_sid,
        "direction": call.direction.value,
        "status": call.status.value,
        "from_number": call.from_number,
        "to_number": call.to_number,
        "duration_seconds": call.duration_seconds,
        "started_at": call.started_at,
        "answered_at": call.answered_at,
        "ended_at": call.ended_at,
        "recording_url": call.recording_url,
        "transcript": call.transcript,
        "transcript_segments": call.transcript_segments,
        "sentiment": call.sentiment,
        "summary": call.summary,
        "agent_id": call.agent_id,
        "cost": call.cost,
        "created_at": call.created_at,
    }


# ============================================
# TWILIO WEBHOOKS
# ============================================

@router.post("/webhook/incoming", response_class=PlainTextResponse)
async def handle_incoming_call(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Handle incoming call webhook from Twilio."""
    form_data = await request.form()
    
    call_sid = form_data.get("CallSid")
    from_number = form_data.get("From")
    to_number = form_data.get("To")
    call_status = form_data.get("CallStatus")
    
    # Find the phone number and associated agent
    result = await db.execute(
        select(PhoneNumber).where(PhoneNumber.phone_number == to_number)
    )
    phone_number = result.scalar_one_or_none()
    
    if not phone_number or not phone_number.agent_id:
        # No agent assigned, return basic message
        return """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Sorry, this number is not configured to receive calls. Goodbye.</Say>
    <Hangup/>
</Response>"""
    
    # Get agent
    agent_result = await db.execute(
        select(Agent).where(Agent.id == phone_number.agent_id)
    )
    agent = agent_result.scalar_one_or_none()
    
    if not agent:
        return """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Sorry, the agent is not available. Goodbye.</Say>
    <Hangup/>
</Response>"""
    
    # Create call record
    call = Call(
        call_sid=call_sid,
        direction=CallDirection.INBOUND,
        status=CallStatus.IN_PROGRESS,
        from_number=from_number,
        to_number=to_number,
        started_at=datetime.utcnow().isoformat(),
        agent_id=agent.id,
        phone_number_id=phone_number.id,
        workspace_id=phone_number.workspace_id,
    )
    db.add(call)
    
    # Update phone number stats
    phone_number.total_calls = str(int(phone_number.total_calls or "0") + 1)
    
    await db.commit()
    
    import os
    base_url = os.getenv("APP_BASE_URL", "https://your-domain.com")
    gather_url = f"{base_url}/api/v1/voice/webhook/gather/{agent.id}/{call_sid}"
    
    # Generate welcome TwiML
    welcome_message = agent.welcome_message or f"Hello! You've reached {agent.name}. How can I help you today?"
    
    return voice_service.generate_welcome_twiml(welcome_message, gather_url)


@router.post("/webhook/gather/{agent_id}/{call_sid}", response_class=PlainTextResponse)
async def handle_speech_input(
    agent_id: UUID,
    call_sid: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Handle speech input from caller and generate AI response."""
    form_data = await request.form()
    
    speech_result = form_data.get("SpeechResult", "")
    confidence = form_data.get("Confidence", "0")
    
    if not speech_result:
        import os
        base_url = os.getenv("APP_BASE_URL", "https://your-domain.com")
        gather_url = f"{base_url}/api/v1/voice/webhook/gather/{agent_id}/{call_sid}"
        return voice_service.generate_response_twiml(
            "I didn't catch that. Could you please repeat?",
            gather_url
        )
    
    # Get agent
    agent_result = await db.execute(
        select(Agent).where(Agent.id == agent_id)
    )
    agent = agent_result.scalar_one_or_none()
    
    if not agent:
        return """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Sorry, I'm having trouble. Goodbye.</Say>
    <Hangup/>
</Response>"""
    
    # Get call record
    call_result = await db.execute(
        select(Call).where(Call.call_sid == call_sid)
    )
    call = call_result.scalar_one_or_none()
    
    # Update transcript
    if call:
        segments = call.transcript_segments or []
        segments.append({
            "role": "user",
            "text": speech_result,
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": float(confidence)
        })
        call.transcript_segments = segments
    
    # Generate AI response
    try:
        # Build conversation history
        conversation_history = []
        if call and call.transcript_segments:
            for seg in call.transcript_segments:
                role = "user" if seg["role"] == "user" else "assistant"
                conversation_history.append({"role": role, "content": seg["text"]})
        else:
            conversation_history.append({"role": "user", "content": speech_result})
        
        # Get AI response
        ai_response = await llm_service.generate_response(
            messages=conversation_history,
            system_prompt=agent.system_prompt or "You are a helpful voice assistant. Keep responses concise and conversational.",
            model=agent.llm_model or "gpt-4o-mini",
            provider=agent.llm_provider or "openai",
            temperature=agent.temperature or 0.7,
            max_tokens=150,  # Keep responses short for voice
        )
        
        # Update transcript with AI response
        if call:
            segments = call.transcript_segments or []
            segments.append({
                "role": "assistant",
                "text": ai_response,
                "timestamp": datetime.utcnow().isoformat()
            })
            call.transcript_segments = segments
            call.transcript = "\n".join([
                f"{'User' if s['role'] == 'user' else 'Agent'}: {s['text']}"
                for s in segments
            ])
            await db.commit()
        
        # Check for conversation end signals
        end_signals = ["goodbye", "bye", "thank you, goodbye", "that's all", "hang up"]
        should_end = any(signal in speech_result.lower() for signal in end_signals)
        
        import os
        base_url = os.getenv("APP_BASE_URL", "https://your-domain.com")
        gather_url = f"{base_url}/api/v1/voice/webhook/gather/{agent_id}/{call_sid}"
        
        return voice_service.generate_response_twiml(ai_response, gather_url, end_call=should_end)
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">I'm having trouble processing your request. Please try again.</Say>
    <Hangup/>
</Response>"""


@router.post("/webhook/status/{call_sid}")
async def handle_call_status(
    call_sid: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Handle call status updates from Twilio."""
    form_data = await request.form()
    
    call_status = form_data.get("CallStatus")
    call_duration = form_data.get("CallDuration", "0")
    recording_url = form_data.get("RecordingUrl")
    recording_sid = form_data.get("RecordingSid")
    
    # Find call
    result = await db.execute(
        select(Call).where(Call.call_sid == call_sid)
    )
    call = result.scalar_one_or_none()
    
    if call:
        # Map Twilio status to our enum
        status_map = {
            "initiated": CallStatus.INITIATED,
            "ringing": CallStatus.RINGING,
            "in-progress": CallStatus.IN_PROGRESS,
            "completed": CallStatus.COMPLETED,
            "busy": CallStatus.BUSY,
            "no-answer": CallStatus.NO_ANSWER,
            "failed": CallStatus.FAILED,
            "canceled": CallStatus.CANCELED,
        }
        
        call.status = status_map.get(call_status, CallStatus.COMPLETED)
        call.duration_seconds = int(call_duration)
        
        if recording_url:
            call.recording_url = recording_url
            call.recording_sid = recording_sid
        
        if call_status in ["completed", "busy", "no-answer", "failed", "canceled"]:
            call.ended_at = datetime.utcnow().isoformat()
        
        # Log event
        event = CallEvent(
            event_type=call_status,
            event_data=dict(form_data),
            call_id=call.id,
            call_sid=call_sid,
            twilio_timestamp=form_data.get("Timestamp"),
        )
        db.add(event)
        
        await db.commit()
    
    return {"status": "ok"}


@router.post("/webhook/outbound/{agent_id}", response_class=PlainTextResponse)
async def handle_outbound_call_connected(
    agent_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Handle when outbound call is connected."""
    form_data = await request.form()
    call_sid = form_data.get("CallSid")
    
    # Get agent
    agent_result = await db.execute(
        select(Agent).where(Agent.id == agent_id)
    )
    agent = agent_result.scalar_one_or_none()
    
    if not agent:
        return """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Sorry, there was an error. Goodbye.</Say>
    <Hangup/>
</Response>"""
    
    # Update call status
    call_result = await db.execute(
        select(Call).where(Call.call_sid == call_sid)
    )
    call = call_result.scalar_one_or_none()
    if call:
        call.status = CallStatus.IN_PROGRESS
        call.answered_at = datetime.utcnow().isoformat()
        await db.commit()
    
    import os
    base_url = os.getenv("APP_BASE_URL", "https://your-domain.com")
    gather_url = f"{base_url}/api/v1/voice/webhook/gather/{agent_id}/{call_sid}"
    
    welcome_message = agent.welcome_message or f"Hello! This is {agent.name}. How can I help you today?"
    
    return voice_service.generate_welcome_twiml(welcome_message, gather_url)