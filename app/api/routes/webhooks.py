"""
ConvoHubAI - Webhook API Routes
Handles webhook configuration and management
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel
import httpx
import hashlib
import hmac
import json
import secrets

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.agent import Agent
from app.models.webhook import Webhook


router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


# ============================================
# SCHEMAS
# ============================================

class WebhookCreate(BaseModel):
    name: str
    url: str
    description: Optional[str] = None
    events: List[str] = ["conversation_started", "message_received", "conversation_ended"]
    agent_id: UUID


class WebhookUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    events: Optional[List[str]] = None
    is_active: Optional[bool] = None


class WebhookResponse(BaseModel):
    id: UUID
    name: str
    url: str
    description: Optional[str]
    events: List[str]
    is_active: bool
    agent_id: UUID
    total_calls: str
    successful_calls: str
    failed_calls: str
    last_triggered_at: Optional[str]
    last_error: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class WebhookTestRequest(BaseModel):
    event_type: str = "test"
    payload: Optional[dict] = None


# ============================================
# WEBHOOK EVENT TYPES
# ============================================

WEBHOOK_EVENTS = [
    "conversation_started",
    "conversation_ended",
    "message_received",
    "message_sent",
    "agent_activated",
    "agent_paused",
    "error_occurred",
    "test",
]


# ============================================
# ROUTES
# ============================================

@router.get("", response_model=List[WebhookResponse])
async def list_webhooks(
    agent_id: Optional[UUID] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all webhooks in the workspace."""
    query = select(Webhook).where(
        and_(
            Webhook.workspace_id == current_user.current_workspace_id,
            Webhook.is_deleted == False,
        )
    )
    
    if agent_id:
        query = query.where(Webhook.agent_id == agent_id)
    
    result = await db.execute(query.order_by(Webhook.created_at.desc()))
    webhooks = result.scalars().all()
    
    return [
        WebhookResponse(
            id=wh.id,
            name=wh.name,
            url=wh.url,
            description=wh.description,
            events=wh.events or [],
            is_active=wh.is_active,
            agent_id=wh.agent_id,
            total_calls=wh.total_calls,
            successful_calls=wh.successful_calls,
            failed_calls=wh.failed_calls,
            last_triggered_at=wh.last_triggered_at,
            last_error=wh.last_error,
            created_at=wh.created_at,
        )
        for wh in webhooks
    ]


@router.post("", response_model=WebhookResponse)
async def create_webhook(
    data: WebhookCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new webhook."""
    # Verify agent access
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
    
    # Validate events
    for event in data.events:
        if event not in WEBHOOK_EVENTS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid event type: {event}. Valid types: {WEBHOOK_EVENTS}"
            )
    
    # Generate secret for signature verification
    webhook_secret = secrets.token_hex(32)
    
    webhook = Webhook(
        name=data.name,
        url=data.url,
        description=data.description,
        events=data.events,
        secret=webhook_secret,
        agent_id=data.agent_id,
        workspace_id=current_user.current_workspace_id,
    )
    db.add(webhook)
    await db.commit()
    await db.refresh(webhook)
    
    return WebhookResponse(
        id=webhook.id,
        name=webhook.name,
        url=webhook.url,
        description=webhook.description,
        events=webhook.events or [],
        is_active=webhook.is_active,
        agent_id=webhook.agent_id,
        total_calls=webhook.total_calls,
        successful_calls=webhook.successful_calls,
        failed_calls=webhook.failed_calls,
        last_triggered_at=webhook.last_triggered_at,
        last_error=webhook.last_error,
        created_at=webhook.created_at,
    )


@router.get("/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(
    webhook_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific webhook."""
    result = await db.execute(
        select(Webhook).where(
            and_(
                Webhook.id == webhook_id,
                Webhook.workspace_id == current_user.current_workspace_id,
                Webhook.is_deleted == False,
            )
        )
    )
    webhook = result.scalar_one_or_none()
    
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    return WebhookResponse(
        id=webhook.id,
        name=webhook.name,
        url=webhook.url,
        description=webhook.description,
        events=webhook.events or [],
        is_active=webhook.is_active,
        agent_id=webhook.agent_id,
        total_calls=webhook.total_calls,
        successful_calls=webhook.successful_calls,
        failed_calls=webhook.failed_calls,
        last_triggered_at=webhook.last_triggered_at,
        last_error=webhook.last_error,
        created_at=webhook.created_at,
    )


@router.patch("/{webhook_id}", response_model=WebhookResponse)
async def update_webhook(
    webhook_id: UUID,
    data: WebhookUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a webhook."""
    result = await db.execute(
        select(Webhook).where(
            and_(
                Webhook.id == webhook_id,
                Webhook.workspace_id == current_user.current_workspace_id,
                Webhook.is_deleted == False,
            )
        )
    )
    webhook = result.scalar_one_or_none()
    
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    if data.name is not None:
        webhook.name = data.name
    if data.url is not None:
        webhook.url = data.url
    if data.description is not None:
        webhook.description = data.description
    if data.events is not None:
        for event in data.events:
            if event not in WEBHOOK_EVENTS:
                raise HTTPException(status_code=400, detail=f"Invalid event type: {event}")
        webhook.events = data.events
    if data.is_active is not None:
        webhook.is_active = data.is_active
    
    webhook.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(webhook)
    
    return WebhookResponse(
        id=webhook.id,
        name=webhook.name,
        url=webhook.url,
        description=webhook.description,
        events=webhook.events or [],
        is_active=webhook.is_active,
        agent_id=webhook.agent_id,
        total_calls=webhook.total_calls,
        successful_calls=webhook.successful_calls,
        failed_calls=webhook.failed_calls,
        last_triggered_at=webhook.last_triggered_at,
        last_error=webhook.last_error,
        created_at=webhook.created_at,
    )


@router.delete("/{webhook_id}")
async def delete_webhook(
    webhook_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a webhook."""
    result = await db.execute(
        select(Webhook).where(
            and_(
                Webhook.id == webhook_id,
                Webhook.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    webhook = result.scalar_one_or_none()
    
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    webhook.is_deleted = True
    webhook.updated_at = datetime.utcnow()
    await db.commit()
    
    return {"message": "Webhook deleted"}


@router.post("/{webhook_id}/test")
async def test_webhook(
    webhook_id: UUID,
    data: WebhookTestRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Send a test request to a webhook."""
    result = await db.execute(
        select(Webhook).where(
            and_(
                Webhook.id == webhook_id,
                Webhook.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    webhook = result.scalar_one_or_none()
    
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    # Build test payload
    payload = {
        "event": data.event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "webhook_id": str(webhook.id),
        "agent_id": str(webhook.agent_id),
        "data": data.payload or {
            "test": True,
            "message": "This is a test webhook event from ConvoHubAI"
        }
    }
    
    # Generate signature
    payload_str = json.dumps(payload, sort_keys=True)
    signature = hmac.new(
        (webhook.secret or "").encode(),
        payload_str.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Send request
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                webhook.url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-ConvoHubAI-Signature": signature,
                    "X-ConvoHubAI-Event": data.event_type,
                }
            )
        
        webhook.total_calls = str(int(webhook.total_calls or "0") + 1)
        webhook.last_triggered_at = datetime.utcnow().isoformat()
        
        if response.is_success:
            webhook.successful_calls = str(int(webhook.successful_calls or "0") + 1)
            webhook.last_error = None
            await db.commit()
            
            return {
                "success": True,
                "status_code": response.status_code,
                "response": response.text[:500],
                "message": "Webhook test successful"
            }
        else:
            webhook.failed_calls = str(int(webhook.failed_calls or "0") + 1)
            webhook.last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            await db.commit()
            
            return {
                "success": False,
                "status_code": response.status_code,
                "error": response.text[:500],
                "message": "Webhook returned non-success status"
            }
            
    except httpx.RequestError as e:
        webhook.total_calls = str(int(webhook.total_calls or "0") + 1)
        webhook.failed_calls = str(int(webhook.failed_calls or "0") + 1)
        webhook.last_error = str(e)[:200]
        webhook.last_triggered_at = datetime.utcnow().isoformat()
        await db.commit()
        
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to connect to webhook URL"
        }


@router.get("/{webhook_id}/secret")
async def get_webhook_secret(
    webhook_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the webhook secret for signature verification."""
    result = await db.execute(
        select(Webhook).where(
            and_(
                Webhook.id == webhook_id,
                Webhook.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    webhook = result.scalar_one_or_none()
    
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    return {"secret": webhook.secret}


@router.post("/{webhook_id}/rotate-secret")
async def rotate_webhook_secret(
    webhook_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rotate the webhook secret."""
    result = await db.execute(
        select(Webhook).where(
            and_(
                Webhook.id == webhook_id,
                Webhook.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    webhook = result.scalar_one_or_none()
    
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    webhook.secret = secrets.token_hex(32)
    webhook.updated_at = datetime.utcnow()
    await db.commit()
    
    return {"secret": webhook.secret, "message": "Webhook secret rotated successfully"}