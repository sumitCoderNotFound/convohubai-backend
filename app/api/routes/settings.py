"""
ConvoHubAI - Settings API Routes
User profile and workspace settings
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr
from uuid import UUID

from app.core.database import get_db
from app.core.security import get_current_user, hash_password, verify_password
from app.models.user import User, Workspace
from app.models.agent import Agent
from app.models.conversation import Conversation

router = APIRouter(prefix="/settings", tags=["settings"])


# ============================================
# SCHEMAS
# ============================================

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    timezone: Optional[str] = None


class PasswordChange(BaseModel):
    current_password: str
    new_password: str


class NotificationSettings(BaseModel):
    email_notifications: bool = True
    sms_notifications: bool = False
    call_alerts: bool = True
    weekly_report: bool = True


# ============================================
# PROFILE ROUTES
# ============================================

@router.get("/profile")
async def get_profile(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user profile"""
    return {
        "id": str(current_user.id),
        "email": current_user.email,
        "full_name": current_user.full_name,
        "phone": current_user.phone,
        "avatar_url": current_user.avatar_url,
        "timezone": "America/New_York",  # Default, could add to user model
        "is_verified": current_user.is_verified,
        "created_at": current_user.created_at.isoformat() if current_user.created_at else None,
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None
    }


@router.patch("/profile")
async def update_profile(
    data: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update user profile"""
    if data.full_name is not None:
        current_user.full_name = data.full_name
    if data.phone is not None:
        current_user.phone = data.phone
    
    current_user.updated_at = datetime.utcnow()
    await db.commit()
    
    return {"message": "Profile updated successfully"}


@router.post("/password")
async def change_password(
    data: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Change user password"""
    # Verify current password
    if not verify_password(data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Update password
    current_user.hashed_password = hash_password(data.new_password)
    current_user.updated_at = datetime.utcnow()
    await db.commit()
    
    return {"message": "Password changed successfully"}


# ============================================
# WORKSPACE ROUTES
# ============================================

@router.get("/workspace")
async def get_workspace(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current workspace details"""
    workspace_id = current_user.current_workspace_id
    
    result = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    return {
        "id": str(workspace.id),
        "name": workspace.name,
        "slug": workspace.slug,
        "description": workspace.description,
        "logo_url": workspace.logo_url,
        "industry": workspace.industry,
        "timezone": workspace.timezone,
        "subscription_plan": workspace.subscription_plan,
        "subscription_status": workspace.subscription_status,
        "created_at": workspace.created_at.isoformat() if workspace.created_at else None
    }


# ============================================
# BILLING ROUTES
# ============================================

@router.get("/billing")
async def get_billing(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get billing information"""
    workspace_id = current_user.current_workspace_id
    
    result = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    # Get usage stats
    # Conversations this month
    now = datetime.utcnow()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    result = await db.execute(
        select(func.count(Conversation.id)).where(
            and_(
                Conversation.workspace_id == workspace_id,
                Conversation.created_at >= month_start
            )
        )
    )
    conversations_used = result.scalar() or 0
    
    # Agents count
    result = await db.execute(
        select(func.count(Agent.id)).where(
            and_(
                Agent.workspace_id == workspace_id,
                Agent.is_deleted == False
            )
        )
    )
    agents_used = result.scalar() or 0
    
    # Plan limits (these would come from a plans table in production)
    plan_limits = {
        "free": {"conversations": 500, "agents": 1, "price": 0},
        "starter": {"conversations": 2000, "agents": 3, "price": 49},
        "professional": {"conversations": 10000, "agents": 10, "price": 149},
        "enterprise": {"conversations": 999999, "agents": 999, "price": 0}
    }
    
    plan = workspace.subscription_plan or "free"
    limits = plan_limits.get(plan, plan_limits["free"])
    
    # Calculate days remaining in trial
    trial_days_remaining = 0
    if workspace.trial_ends_at:
        delta = workspace.trial_ends_at - now
        trial_days_remaining = max(0, delta.days)
    
    return {
        "plan": {
            "name": plan.capitalize(),
            "price": limits["price"],
            "billing_cycle": "monthly",
            "status": workspace.subscription_status or "active"
        },
        "trial": {
            "is_trial": plan == "free" or workspace.subscription_status == "trialing",
            "days_remaining": trial_days_remaining
        },
        "usage": {
            "conversations": {
                "used": conversations_used,
                "limit": limits["conversations"]
            },
            "agents": {
                "used": agents_used,
                "limit": limits["agents"]
            }
        },
        "payment_method": None  # Would come from Stripe in production
    }


@router.get("/billing/invoices")
async def get_invoices(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get billing invoices (mock data for now)"""
    # In production, this would fetch from Stripe
    return {
        "invoices": []
    }


# ============================================
# NOTIFICATION ROUTES  
# ============================================

@router.get("/notifications")
async def get_notification_settings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get notification settings"""
    # In production, these would be stored in user preferences
    return {
        "email_notifications": True,
        "sms_notifications": False,
        "call_alerts": True,
        "weekly_report": True
    }


@router.patch("/notifications")
async def update_notification_settings(
    data: NotificationSettings,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update notification settings"""
    # In production, save to user preferences
    return {"message": "Notification settings updated"}


# ============================================
# API KEYS ROUTES
# ============================================

@router.get("/api-keys")
async def get_api_keys(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get API keys"""
    # In production, fetch from api_keys table
    return {"keys": []}


@router.post("/api-keys")
async def create_api_key(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new API key"""
    import secrets
    
    # Generate a new API key
    key = f"sk_{secrets.token_urlsafe(32)}"
    
    # In production, save to api_keys table with hash
    return {
        "key": key,
        "message": "API key created. Please save it securely - it won't be shown again."
    }
