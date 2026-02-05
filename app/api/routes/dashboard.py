"""
ConvoHubAI - Dashboard API Routes
Provides statistics and analytics for the dashboard
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, case, and_, select, Integer, Float
from datetime import datetime, timedelta
from typing import Optional
import uuid

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.agent import Agent, AgentStatus
from app.models.conversation import Conversation, ConversationStatus, ConversationType
from app.models.call import Call, CallStatus

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats")
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get dashboard statistics:
    - Total Calls/Conversations
    - Success Rate
    - Average Duration
    - Active Agents
    """
    workspace_id = current_user.current_workspace_id
    
    # Get date range for comparison (current week vs last week)
    today = datetime.utcnow()
    week_start = today - timedelta(days=7)
    last_week_start = today - timedelta(days=14)
    
    # ============================================
    # TOTAL CALLS (from conversations table)
    # ============================================
    
    # Total calls
    result = await db.execute(
        select(func.count(Conversation.id)).where(
            Conversation.workspace_id == workspace_id
        )
    )
    total_calls = result.scalar() or 0
    
    # Current week calls
    result = await db.execute(
        select(func.count(Conversation.id)).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.created_at >= week_start
            )
        )
    )
    current_week_calls = result.scalar() or 0
    
    # Last week calls
    result = await db.execute(
        select(func.count(Conversation.id)).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.created_at >= last_week_start,
                Conversation.created_at < week_start
            )
        )
    )
    last_week_calls = result.scalar() or 0
    
    # Calculate change percentage
    if last_week_calls > 0:
        calls_change = ((current_week_calls - last_week_calls) / last_week_calls) * 100
    else:
        calls_change = 100 if current_week_calls > 0 else 0
    
    # ============================================
    # SUCCESS RATE
    # ============================================
    
    result = await db.execute(
        select(func.count(Conversation.id)).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.status == ConversationStatus.COMPLETED
            )
        )
    )
    completed_calls = result.scalar() or 0
    
    success_rate = (completed_calls / total_calls * 100) if total_calls > 0 else 0
    
    # Last week success rate
    result = await db.execute(
        select(func.count(Conversation.id)).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.created_at >= last_week_start,
                Conversation.created_at < week_start
            )
        )
    )
    last_week_total = result.scalar() or 0
    
    result = await db.execute(
        select(func.count(Conversation.id)).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.created_at >= last_week_start,
                Conversation.created_at < week_start,
                Conversation.status == ConversationStatus.COMPLETED
            )
        )
    )
    last_week_completed = result.scalar() or 0
    
    last_week_success_rate = (last_week_completed / last_week_total * 100) if last_week_total > 0 else 0
    success_rate_change = success_rate - last_week_success_rate
    
    # ============================================
    # AVERAGE DURATION
    # ============================================
    
    # Get all conversations with duration and calculate average manually
    result = await db.execute(
        select(Conversation.duration_seconds).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.duration_seconds.isnot(None)
            )
        )
    )
    conversations_with_duration = result.scalars().all()
    
    # Calculate average duration
    total_duration = 0
    count_with_duration = 0
    for duration in conversations_with_duration:
        try:
            dur = int(duration) if duration else 0
            if dur > 0:
                total_duration += dur
                count_with_duration += 1
        except (ValueError, TypeError):
            pass
    
    avg_duration_seconds = total_duration / count_with_duration if count_with_duration > 0 else 0
    
    # Format duration as "Xm Ys"
    minutes = int(avg_duration_seconds // 60)
    seconds = int(avg_duration_seconds % 60)
    avg_duration_formatted = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
    
    # ============================================
    # ACTIVE AGENTS
    # ============================================
    
    result = await db.execute(
        select(func.count(Agent.id)).where(
            and_(
                Agent.workspace_id == workspace_id,
                Agent.status == AgentStatus.ACTIVE
            )
        )
    )
    active_agents = result.scalar() or 0
    
    result = await db.execute(
        select(func.count(Agent.id)).where(
            Agent.workspace_id == workspace_id
        )
    )
    total_agents = result.scalar() or 0
    
    return {
        "total_calls": {
            "value": total_calls,
            "change": round(calls_change, 1),
            "change_type": "up" if calls_change >= 0 else "down"
        },
        "success_rate": {
            "value": round(success_rate, 1),
            "change": round(abs(success_rate_change), 1),
            "change_type": "up" if success_rate_change >= 0 else "down"
        },
        "avg_duration": {
            "value": avg_duration_formatted,
            "seconds": round(avg_duration_seconds, 0),
            "change": None,
            "change_type": None
        },
        "active_agents": {
            "value": active_agents,
            "total": total_agents,
            "change": None,
            "change_type": None
        }
    }


@router.get("/recent-calls")
async def get_recent_calls(
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get recent calls/conversations for the dashboard
    """
    workspace_id = current_user.current_workspace_id
    
    # Get recent conversations with agent info
    result = await db.execute(
        select(Conversation).where(
            Conversation.workspace_id == workspace_id
        ).order_by(
            Conversation.created_at.desc()
        ).limit(limit)
    )
    conversations = result.scalars().all()
    
    result_list = []
    for conv in conversations:
        # Get agent name
        agent_result = await db.execute(
            select(Agent).where(Agent.id == conv.agent_id)
        )
        agent = agent_result.scalar_one_or_none()
        agent_name = agent.name if agent else "Unknown Agent"
        
        # Calculate time ago
        time_ago = get_time_ago(conv.created_at)
        
        # Format duration
        duration_seconds = int(conv.duration_seconds) if conv.duration_seconds else 0
        minutes = duration_seconds // 60
        seconds = duration_seconds % 60
        duration_formatted = f"{minutes}:{str(seconds).zfill(2)}"
        
        # Determine phone number to display
        phone = conv.contact_phone or conv.from_number or "N/A"
        
        result_list.append({
            "id": str(conv.id),
            "phone": phone,
            "agent": agent_name,
            "agent_id": str(conv.agent_id),
            "duration": duration_formatted,
            "duration_seconds": duration_seconds,
            "status": conv.status.value if conv.status else "unknown",
            "type": conv.conversation_type.value if conv.conversation_type else "chat",
            "time": time_ago,
            "created_at": conv.created_at.isoformat() if conv.created_at else None
        })
    
    return {"calls": result_list, "total": len(result_list)}


@router.get("/top-agents")
async def get_top_agents(
    limit: int = 5,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get top performing agents based on conversations and success rate
    """
    workspace_id = current_user.current_workspace_id
    
    # Get agents
    result = await db.execute(
        select(Agent).where(
            Agent.workspace_id == workspace_id
        )
    )
    agents = result.scalars().all()
    
    agent_stats = []
    for agent in agents:
        # Count total conversations for this agent
        result = await db.execute(
            select(func.count(Conversation.id)).where(
                Conversation.agent_id == agent.id
            )
        )
        total_convs = result.scalar() or 0
        
        # Count completed conversations
        result = await db.execute(
            select(func.count(Conversation.id)).where(
                and_(
                    Conversation.agent_id == agent.id,
                    Conversation.status == ConversationStatus.COMPLETED
                )
            )
        )
        completed_convs = result.scalar() or 0
        
        # Calculate success rate
        success_rate = (completed_convs / total_convs * 100) if total_convs > 0 else 0
        
        # Get average rating manually
        result = await db.execute(
            select(Conversation.user_rating).where(
                and_(
                    Conversation.agent_id == agent.id,
                    Conversation.user_rating.isnot(None)
                )
            )
        )
        ratings = result.scalars().all()
        
        total_rating = 0
        rating_count = 0
        for rating in ratings:
            try:
                r = float(rating) if rating else 0
                if r > 0:
                    total_rating += r
                    rating_count += 1
            except (ValueError, TypeError):
                pass
        
        avg_rating = total_rating / rating_count if rating_count > 0 else None
        
        agent_stats.append({
            "id": str(agent.id),
            "name": agent.name,
            "status": agent.status.value if agent.status else "draft",
            "total_conversations": total_convs,
            "completed_conversations": completed_convs,
            "success_rate": round(success_rate, 1),
            "avg_rating": round(avg_rating, 1) if avg_rating else None,
            "channels": agent.channels or ["chat"]
        })
    
    # Sort by success rate and total conversations
    agent_stats.sort(key=lambda x: (x["success_rate"], x["total_conversations"]), reverse=True)
    
    return {"agents": agent_stats[:limit]}


@router.get("/activity-chart")
async def get_activity_chart(
    days: int = 7,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get conversation activity data for charts
    """
    workspace_id = current_user.current_workspace_id
    
    result_data = []
    for i in range(days - 1, -1, -1):
        day = datetime.utcnow() - timedelta(days=i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        
        # Count conversations for this day
        result = await db.execute(
            select(func.count(Conversation.id)).where(
                and_(
                    Conversation.workspace_id == workspace_id,
                    Conversation.created_at >= day_start,
                    Conversation.created_at < day_end
                )
            )
        )
        count = result.scalar() or 0
        
        result_data.append({
            "date": day_start.strftime("%Y-%m-%d"),
            "day": day_start.strftime("%a"),
            "conversations": count
        })
    
    return {"data": result_data}


def get_time_ago(dt):
    """Convert datetime to 'X time ago' format"""
    if not dt:
        return "Unknown"
    
    now = datetime.utcnow()
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return "Just now"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        return f"{minutes} min ago"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif seconds < 604800:
        days = int(seconds // 86400)
        return f"{days} day{'s' if days > 1 else ''} ago"
    else:
        return dt.strftime("%b %d, %Y")
