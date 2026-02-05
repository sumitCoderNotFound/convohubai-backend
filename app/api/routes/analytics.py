"""
ConvoHubAI - Analytics API Routes
Provides detailed analytics and reporting
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, and_, select, case
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.agent import Agent, AgentStatus
from app.models.conversation import Conversation, ConversationStatus, ConversationType

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview")
async def get_analytics_overview(
    time_range: str = "7d",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get analytics overview with stats
    time_range: 24h, 7d, 30d, 90d
    """
    workspace_id = current_user.current_workspace_id
    
    # Calculate date range
    now = datetime.utcnow()
    if time_range == "24h":
        start_date = now - timedelta(hours=24)
        prev_start = now - timedelta(hours=48)
    elif time_range == "7d":
        start_date = now - timedelta(days=7)
        prev_start = now - timedelta(days=14)
    elif time_range == "30d":
        start_date = now - timedelta(days=30)
        prev_start = now - timedelta(days=60)
    else:  # 90d
        start_date = now - timedelta(days=90)
        prev_start = now - timedelta(days=180)
    
    # Current period stats
    result = await db.execute(
        select(func.count(Conversation.id)).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.created_at >= start_date
            )
        )
    )
    total_calls = result.scalar() or 0
    
    # Previous period stats
    result = await db.execute(
        select(func.count(Conversation.id)).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.created_at >= prev_start,
                Conversation.created_at < start_date
            )
        )
    )
    prev_total_calls = result.scalar() or 0
    
    # Calculate change
    calls_change = ((total_calls - prev_total_calls) / max(prev_total_calls, 1)) * 100
    
    # Success rate
    result = await db.execute(
        select(func.count(Conversation.id)).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.created_at >= start_date,
                Conversation.status == ConversationStatus.COMPLETED
            )
        )
    )
    completed_calls = result.scalar() or 0
    success_rate = (completed_calls / max(total_calls, 1)) * 100
    
    # Previous success rate
    result = await db.execute(
        select(func.count(Conversation.id)).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.created_at >= prev_start,
                Conversation.created_at < start_date,
                Conversation.status == ConversationStatus.COMPLETED
            )
        )
    )
    prev_completed = result.scalar() or 0
    prev_success_rate = (prev_completed / max(prev_total_calls, 1)) * 100
    success_change = success_rate - prev_success_rate
    
    # Average duration
    result = await db.execute(
        select(Conversation.duration_seconds).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.created_at >= start_date,
                Conversation.duration_seconds.isnot(None)
            )
        )
    )
    durations = result.scalars().all()
    
    total_duration = 0
    duration_count = 0
    for d in durations:
        try:
            dur = int(d) if d else 0
            if dur > 0:
                total_duration += dur
                duration_count += 1
        except:
            pass
    
    avg_duration = total_duration / max(duration_count, 1)
    mins = int(avg_duration // 60)
    secs = int(avg_duration % 60)
    avg_duration_formatted = f"{mins}m {secs}s"
    
    # Unique callers (unique visitor_ids)
    result = await db.execute(
        select(func.count(func.distinct(Conversation.visitor_id))).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.created_at >= start_date
            )
        )
    )
    unique_callers = result.scalar() or 0
    
    result = await db.execute(
        select(func.count(func.distinct(Conversation.visitor_id))).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.created_at >= prev_start,
                Conversation.created_at < start_date
            )
        )
    )
    prev_unique_callers = result.scalar() or 0
    callers_change = ((unique_callers - prev_unique_callers) / max(prev_unique_callers, 1)) * 100
    
    return {
        "stats": [
            {
                "label": "Total Calls",
                "value": f"{total_calls:,}",
                "change": f"{'+' if calls_change >= 0 else ''}{calls_change:.0f}%",
                "trend": "up" if calls_change >= 0 else "down"
            },
            {
                "label": "Avg. Call Duration",
                "value": avg_duration_formatted,
                "change": "-",
                "trend": "up"
            },
            {
                "label": "Success Rate",
                "value": f"{success_rate:.1f}%",
                "change": f"{'+' if success_change >= 0 else ''}{success_change:.1f}%",
                "trend": "up" if success_change >= 0 else "down"
            },
            {
                "label": "Unique Callers",
                "value": f"{unique_callers:,}",
                "change": f"{'+' if callers_change >= 0 else ''}{callers_change:.0f}%",
                "trend": "up" if callers_change >= 0 else "down"
            }
        ]
    }


@router.get("/daily-calls")
async def get_daily_calls(
    time_range: str = "7d",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get daily call volume data for charts"""
    workspace_id = current_user.current_workspace_id
    
    days = {"24h": 1, "7d": 7, "30d": 30, "90d": 90}.get(time_range, 7)
    
    data = []
    for i in range(days - 1, -1, -1):
        day = datetime.utcnow() - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        # Total calls
        result = await db.execute(
            select(func.count(Conversation.id)).where(
                and_(
                    Conversation.workspace_id == workspace_id,
                    Conversation.created_at >= day_start,
                    Conversation.created_at < day_end
                )
            )
        )
        total = result.scalar() or 0
        
        # Successful calls
        result = await db.execute(
            select(func.count(Conversation.id)).where(
                and_(
                    Conversation.workspace_id == workspace_id,
                    Conversation.created_at >= day_start,
                    Conversation.created_at < day_end,
                    Conversation.status == ConversationStatus.COMPLETED
                )
            )
        )
        success = result.scalar() or 0
        
        data.append({
            "day": day_start.strftime("%a"),
            "date": day_start.strftime("%Y-%m-%d"),
            "calls": total,
            "success": success
        })
    
    return {"data": data}


@router.get("/call-outcomes")
async def get_call_outcomes(
    time_range: str = "7d",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get call outcome distribution"""
    workspace_id = current_user.current_workspace_id
    
    days = {"24h": 1, "7d": 7, "30d": 30, "90d": 90}.get(time_range, 7)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Count by status
    statuses = [
        (ConversationStatus.COMPLETED, "Completed", "bg-green-500"),
        (ConversationStatus.TRANSFERRED, "Transferred", "bg-amber-500"),
        (ConversationStatus.FAILED, "Failed", "bg-red-500"),
        (ConversationStatus.ABANDONED, "Abandoned", "bg-neutral-400"),
    ]
    
    total = 0
    outcomes = []
    
    for status, name, color in statuses:
        result = await db.execute(
            select(func.count(Conversation.id)).where(
                and_(
                    Conversation.workspace_id == workspace_id,
                    Conversation.created_at >= start_date,
                    Conversation.status == status
                )
            )
        )
        count = result.scalar() or 0
        total += count
        outcomes.append({"name": name, "count": count, "color": color})
    
    # Add active/other
    result = await db.execute(
        select(func.count(Conversation.id)).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.created_at >= start_date,
                Conversation.status == ConversationStatus.ACTIVE
            )
        )
    )
    active_count = result.scalar() or 0
    if active_count > 0:
        total += active_count
        outcomes.append({"name": "Active", "count": active_count, "color": "bg-blue-500"})
    
    # Calculate percentages
    for outcome in outcomes:
        outcome["value"] = round((outcome["count"] / max(total, 1)) * 100)
    
    return {"outcomes": outcomes, "total": total}


@router.get("/agent-performance")
async def get_agent_performance(
    time_range: str = "7d",
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get agent performance metrics"""
    workspace_id = current_user.current_workspace_id
    
    days = {"24h": 1, "7d": 7, "30d": 30, "90d": 90}.get(time_range, 7)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get all agents
    result = await db.execute(
        select(Agent).where(Agent.workspace_id == workspace_id)
    )
    agents = result.scalars().all()
    
    performance = []
    for agent in agents:
        # Total calls
        result = await db.execute(
            select(func.count(Conversation.id)).where(
                and_(
                    Conversation.agent_id == agent.id,
                    Conversation.created_at >= start_date
                )
            )
        )
        total_calls = result.scalar() or 0
        
        if total_calls == 0:
            continue
        
        # Completed calls
        result = await db.execute(
            select(func.count(Conversation.id)).where(
                and_(
                    Conversation.agent_id == agent.id,
                    Conversation.created_at >= start_date,
                    Conversation.status == ConversationStatus.COMPLETED
                )
            )
        )
        completed = result.scalar() or 0
        
        # Average duration
        result = await db.execute(
            select(Conversation.duration_seconds).where(
                and_(
                    Conversation.agent_id == agent.id,
                    Conversation.created_at >= start_date,
                    Conversation.duration_seconds.isnot(None)
                )
            )
        )
        durations = result.scalars().all()
        
        total_dur = 0
        dur_count = 0
        for d in durations:
            try:
                dur = int(d) if d else 0
                if dur > 0:
                    total_dur += dur
                    dur_count += 1
            except:
                pass
        
        avg_dur = total_dur / max(dur_count, 1)
        mins = int(avg_dur // 60)
        secs = int(avg_dur % 60)
        
        success_rate = (completed / total_calls) * 100
        
        performance.append({
            "name": agent.name,
            "calls": total_calls,
            "success": round(success_rate),
            "avgDuration": f"{mins}:{str(secs).zfill(2)}"
        })
    
    # Sort by calls descending
    performance.sort(key=lambda x: x["calls"], reverse=True)
    
    return {"agents": performance[:limit]}


@router.get("/chat-history")
async def get_chat_history(
    limit: int = 50,
    status: Optional[str] = None,
    agent_id: Optional[UUID] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all chat conversations for the workspace (for Chat History page)"""
    workspace_id = current_user.current_workspace_id
    
    # Build query
    query = select(Conversation).where(
        and_(
            Conversation.workspace_id == workspace_id,
            Conversation.conversation_type == ConversationType.CHAT
        )
    )
    
    if status:
        try:
            status_enum = ConversationStatus(status)
            query = query.where(Conversation.status == status_enum)
        except:
            pass
    
    if agent_id:
        query = query.where(Conversation.agent_id == agent_id)
    
    query = query.order_by(Conversation.created_at.desc()).limit(limit)
    
    result = await db.execute(query)
    conversations = result.scalars().all()
    
    chats = []
    for conv in conversations:
        # Get agent name
        agent_result = await db.execute(
            select(Agent).where(Agent.id == conv.agent_id)
        )
        agent = agent_result.scalar_one_or_none()
        
        # Calculate duration
        duration_secs = 0
        try:
            duration_secs = int(conv.duration_seconds) if conv.duration_seconds else 0
        except:
            pass
        
        mins = duration_secs // 60
        secs = duration_secs % 60
        
        chats.append({
            "id": str(conv.id),
            "visitor": conv.visitor_id or conv.contact_name or "Anonymous",
            "agent": agent.name if agent else "Unknown Agent",
            "agent_id": str(conv.agent_id),
            "messages": int(conv.message_count or "0"),
            "duration": f"{mins}:{str(secs).zfill(2)}",
            "status": conv.status.value if conv.status else "unknown",
            "date": conv.created_at.strftime("%Y-%m-%d") if conv.created_at else "",
            "time": conv.created_at.strftime("%H:%M") if conv.created_at else "",
            "created_at": conv.created_at.isoformat() if conv.created_at else None
        })
    
    return {"chats": chats, "total": len(chats)}
