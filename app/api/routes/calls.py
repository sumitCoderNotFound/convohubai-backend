"""
ConvoHubAI - Calls/Transcripts API Routes
Handles call transcripts and call history for video, voice, chat
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import uuid

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.conversation import Conversation, ConversationType, ConversationStatus

router = APIRouter(prefix="/calls", tags=["calls"])


# ============================================
# Pydantic Schemas
# ============================================

class TranscriptMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class TranscriptCreate(BaseModel):
    agent_id: str
    room_name: str
    participant_identity: str
    call_start: str
    call_end: str
    duration_seconds: int
    messages: List[TranscriptMessage]
    message_count: int
    channel: str = "video"


class TranscriptResponse(BaseModel):
    call_id: str
    message: str


class CallSummary(BaseModel):
    id: str
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    participant_identity: Optional[str] = None
    channel: Optional[str] = None
    duration_seconds: int = 0
    duration_formatted: Optional[str] = None
    message_count: int = 0
    call_start: Optional[datetime] = None
    call_end: Optional[datetime] = None
    status: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class CallDetail(BaseModel):
    id: str
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    participant_identity: Optional[str] = None
    channel: Optional[str] = None
    duration_seconds: int = 0
    duration_formatted: Optional[str] = None
    message_count: int = 0
    call_start: Optional[datetime] = None
    call_end: Optional[datetime] = None
    status: Optional[str] = None
    transcript: List[TranscriptMessage]
    summary: Optional[str] = None
    sentiment: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class CallListResponse(BaseModel):
    items: List[CallSummary]
    total: int
    page: int
    page_size: int


# ============================================
# API Endpoints
# ============================================

@router.post("/transcript", response_model=TranscriptResponse)
async def save_transcript(
    data: TranscriptCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Save a call transcript from the agent worker.
    This endpoint is called by the agent worker when a call ends.
    No authentication required - called internally by agent worker.
    """
    try:
        # Create the conversation record
        call_id = str(uuid.uuid4())
        
        # Parse timestamps
        try:
            call_start = datetime.fromisoformat(data.call_start.replace('Z', '+00:00'))
        except:
            call_start = datetime.utcnow()
        
        try:
            call_end = datetime.fromisoformat(data.call_end.replace('Z', '+00:00'))
        except:
            call_end = datetime.utcnow()
        
        # Determine conversation type
        channel_to_type = {
            "video": ConversationType.VIDEO,
            "voice": ConversationType.VOICE_INBOUND,
            "chat": ConversationType.CHAT,
            "sms": ConversationType.SMS,
        }
        conv_type = channel_to_type.get(data.channel, ConversationType.VIDEO)
        
        # Create conversation
        conversation = Conversation(
            id=call_id,
            agent_id=data.agent_id if data.agent_id and data.agent_id != "unknown" else None,
            room_name=data.room_name,
            participant_identity=data.participant_identity,
            channel=data.channel,
            conversation_type=conv_type,
            duration_seconds=data.duration_seconds,
            message_count=data.message_count,
            call_start=call_start,
            call_end=call_end,
            started_at=call_start,
            ended_at=call_end,
            transcript=[msg.dict() for msg in data.messages],
            status=ConversationStatus.COMPLETED
        )
        
        db.add(conversation)
        await db.commit()
        
        return TranscriptResponse(
            call_id=call_id,
            message="Transcript saved successfully"
        )
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save transcript: {str(e)}")


@router.get("", response_model=CallListResponse)
async def list_calls(
    agent_id: Optional[str] = Query(None),
    channel: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all calls/conversations with optional filters.
    """
    try:
        # Build query
        query = select(Conversation)
        count_query = select(func.count(Conversation.id))
        
        # Apply filters
        filters = []
        
        if agent_id:
            filters.append(Conversation.agent_id == agent_id)
        
        if channel:
            filters.append(Conversation.channel == channel)
        
        if status:
            filters.append(Conversation.status == status)
        
        if start_date:
            try:
                start = datetime.fromisoformat(start_date)
                filters.append(Conversation.call_start >= start)
            except:
                pass
        
        if end_date:
            try:
                end = datetime.fromisoformat(end_date)
                filters.append(Conversation.call_end <= end)
            except:
                pass
        
        if filters:
            query = query.where(and_(*filters))
            count_query = count_query.where(and_(*filters))
        
        # Get total count
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Apply pagination and ordering
        offset = (page - 1) * page_size
        query = query.order_by(desc(Conversation.created_at)).offset(offset).limit(page_size)
        
        # Execute query
        result = await db.execute(query)
        calls = result.scalars().all()
        
        # Format response
        items = []
        for call in calls:
            duration = call.duration_seconds or 0
            mins = duration // 60
            secs = duration % 60
            
            items.append(CallSummary(
                id=str(call.id),
                agent_id=str(call.agent_id) if call.agent_id else None,
                participant_identity=call.participant_identity,
                channel=call.channel,
                duration_seconds=duration,
                duration_formatted=f"{mins:02d}:{secs:02d}",
                message_count=call.message_count or 0,
                call_start=call.call_start,
                call_end=call.call_end,
                status=call.status.value if call.status else "completed",
                created_at=call.created_at
            ))
        
        return CallListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list calls: {str(e)}")


@router.get("/recent")
async def get_recent_calls(
    limit: int = Query(10, ge=1, le=50),
    channel: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get recent calls for dashboard display.
    """
    try:
        query = select(Conversation).order_by(desc(Conversation.created_at)).limit(limit)
        
        if channel:
            query = query.where(Conversation.channel == channel)
        
        result = await db.execute(query)
        calls = result.scalars().all()
        
        items = []
        for call in calls:
            duration = call.duration_seconds or 0
            mins = duration // 60
            secs = duration % 60
            
            items.append({
                "id": str(call.id),
                "agent_id": str(call.agent_id) if call.agent_id else None,
                "participant_identity": call.participant_identity,
                "channel": call.channel,
                "duration_formatted": f"{mins:02d}:{secs:02d}",
                "message_count": call.message_count or 0,
                "status": call.status.value if call.status else "completed",
                "created_at": call.created_at.isoformat() if call.created_at else None
            })
        
        return {"calls": items, "total": len(items)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent calls: {str(e)}")


@router.get("/stats/summary")
async def get_call_stats(
    agent_id: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get call statistics summary.
    """
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Build base filter
        filters = [Conversation.created_at >= start_date]
        if agent_id:
            filters.append(Conversation.agent_id == agent_id)
        
        # Total calls
        total_result = await db.execute(
            select(func.count(Conversation.id)).where(and_(*filters))
        )
        total_calls = total_result.scalar() or 0
        
        # Total duration
        duration_result = await db.execute(
            select(func.sum(Conversation.duration_seconds)).where(and_(*filters))
        )
        total_duration = duration_result.scalar() or 0
        
        # Average duration
        avg_result = await db.execute(
            select(func.avg(Conversation.duration_seconds)).where(and_(*filters))
        )
        avg_duration = avg_result.scalar() or 0
        
        # Total messages
        messages_result = await db.execute(
            select(func.sum(Conversation.message_count)).where(and_(*filters))
        )
        total_messages = messages_result.scalar() or 0
        
        # Calls by channel
        channel_result = await db.execute(
            select(Conversation.channel, func.count(Conversation.id))
            .where(and_(*filters))
            .group_by(Conversation.channel)
        )
        calls_by_channel = {row[0] or "unknown": row[1] for row in channel_result.all()}
        
        # Calls by status
        status_result = await db.execute(
            select(Conversation.status, func.count(Conversation.id))
            .where(and_(*filters))
            .group_by(Conversation.status)
        )
        calls_by_status = {row[0].value if row[0] else "unknown": row[1] for row in status_result.all()}
        
        return {
            "total_calls": total_calls,
            "total_duration_seconds": total_duration,
            "total_duration_minutes": round(total_duration / 60, 1) if total_duration else 0,
            "average_duration_seconds": round(float(avg_duration), 1) if avg_duration else 0,
            "total_messages": total_messages,
            "calls_by_channel": calls_by_channel,
            "calls_by_status": calls_by_status,
            "period_days": days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/{call_id}", response_model=CallDetail)
async def get_call(
    call_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get call details including full transcript.
    """
    try:
        result = await db.execute(
            select(Conversation).where(Conversation.id == call_id)
        )
        call = result.scalar_one_or_none()
        
        if not call:
            raise HTTPException(status_code=404, detail="Call not found")
        
        # Parse transcript
        transcript = []
        if call.transcript:
            for msg in call.transcript:
                transcript.append(TranscriptMessage(
                    role=msg.get("role", "unknown"),
                    content=msg.get("content", ""),
                    timestamp=msg.get("timestamp")
                ))
        
        duration = call.duration_seconds or 0
        mins = duration // 60
        secs = duration % 60
        
        return CallDetail(
            id=str(call.id),
            agent_id=str(call.agent_id) if call.agent_id else None,
            participant_identity=call.participant_identity,
            channel=call.channel,
            duration_seconds=duration,
            duration_formatted=f"{mins:02d}:{secs:02d}",
            message_count=call.message_count or 0,
            call_start=call.call_start,
            call_end=call.call_end,
            status=call.status.value if call.status else "completed",
            transcript=transcript,
            summary=call.summary,
            sentiment=call.sentiment,
            created_at=call.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get call: {str(e)}")


@router.delete("/{call_id}")
async def delete_call(
    call_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a call record.
    """
    try:
        result = await db.execute(
            select(Conversation).where(Conversation.id == call_id)
        )
        call = result.scalar_one_or_none()
        
        if not call:
            raise HTTPException(status_code=404, detail="Call not found")
        
        await db.delete(call)
        await db.commit()
        
        return {"message": "Call deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete call: {str(e)}")