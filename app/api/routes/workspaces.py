"""
ConvoHubAI - Workspace API Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List
import secrets
import re

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User, Workspace, WorkspaceMember, UserRole
from app.models.agent import Agent
from app.models.conversation import Conversation
from app.schemas.user import (
    WorkspaceCreate,
    WorkspaceUpdate,
    WorkspaceResponse,
    WorkspaceWithStats,
    WorkspaceMemberInvite,
    WorkspaceMemberUpdate,
    WorkspaceMemberResponse,
    WorkspaceListResponse,
    UserResponse
)
from app.schemas.auth import MessageResponse

router = APIRouter(prefix="/workspaces", tags=["Workspaces"])


def generate_slug(name: str) -> str:
    """Generate a URL-friendly slug from a name."""
    slug = name.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    return f"{slug}-{secrets.token_hex(4)}"


async def get_workspace_or_404(
    workspace_id: str,
    db: AsyncSession,
    user: User
) -> Workspace:
    """Get workspace by ID or raise 404."""
    result = await db.execute(
        select(Workspace).where(
            Workspace.id == workspace_id,
            Workspace.is_deleted == False
        )
    )
    workspace = result.scalar_one_or_none()
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    # Check if user is a member
    member_result = await db.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace.id,
            WorkspaceMember.user_id == user.id
        )
    )
    member = member_result.scalar_one_or_none()
    
    if not member and not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this workspace"
        )
    
    return workspace


@router.get("", response_model=WorkspaceListResponse)
async def list_workspaces(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all workspaces the user is a member of."""
    result = await db.execute(
        select(Workspace)
        .join(WorkspaceMember, WorkspaceMember.workspace_id == Workspace.id)
        .where(
            WorkspaceMember.user_id == current_user.id,
            Workspace.is_deleted == False
        )
        .order_by(Workspace.created_at.desc())
    )
    workspaces = result.scalars().all()
    
    return WorkspaceListResponse(
        items=[WorkspaceResponse.model_validate(w) for w in workspaces],
        total=len(workspaces)
    )


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    workspace_data: WorkspaceCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new workspace."""
    workspace = Workspace(
        name=workspace_data.name,
        slug=generate_slug(workspace_data.name),
        description=workspace_data.description,
        industry=workspace_data.industry,
        timezone=workspace_data.timezone,
        owner_id=current_user.id,
        subscription_plan="free"
    )
    db.add(workspace)
    await db.flush()
    
    # Add creator as owner
    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=current_user.id,
        role="owner",
        joined_at=workspace.created_at
    )
    db.add(member)
    
    await db.commit()
    await db.refresh(workspace)
    
    return WorkspaceResponse.model_validate(workspace)


@router.get("/{workspace_id}", response_model=WorkspaceWithStats)
async def get_workspace(
    workspace_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get workspace details with statistics."""
    workspace = await get_workspace_or_404(workspace_id, db, current_user)
    
    # Get stats
    member_count = await db.execute(
        select(func.count(WorkspaceMember.id))
        .where(WorkspaceMember.workspace_id == workspace.id)
    )
    
    agent_count = await db.execute(
        select(func.count(Agent.id))
        .where(Agent.workspace_id == workspace.id, Agent.is_deleted == False)
    )
    
    conversation_count = await db.execute(
        select(func.count(Conversation.id))
        .where(Conversation.workspace_id == workspace.id)
    )
    
    response = WorkspaceWithStats.model_validate(workspace)
    response.member_count = member_count.scalar() or 0
    response.agent_count = agent_count.scalar() or 0
    response.conversation_count = conversation_count.scalar() or 0
    
    return response


@router.patch("/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    workspace_id: str,
    workspace_data: WorkspaceUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update workspace settings."""
    workspace = await get_workspace_or_404(workspace_id, db, current_user)
    
    # Check if user is owner or admin
    member_result = await db.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace.id,
            WorkspaceMember.user_id == current_user.id
        )
    )
    member = member_result.scalar_one_or_none()
    
    if not member or member.role not in ["owner", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only workspace owners and admins can update settings"
        )
    
    # Update fields
    update_data = workspace_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(workspace, field, value)
    
    await db.commit()
    await db.refresh(workspace)
    
    return WorkspaceResponse.model_validate(workspace)


@router.delete("/{workspace_id}", response_model=MessageResponse)
async def delete_workspace(
    workspace_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete (soft) a workspace."""
    workspace = await get_workspace_or_404(workspace_id, db, current_user)
    
    # Only owner can delete
    if workspace.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only workspace owner can delete the workspace"
        )
    
    workspace.is_deleted = True
    await db.commit()
    
    return MessageResponse(message="Workspace deleted successfully")


# ============================================
# WORKSPACE MEMBERS
# ============================================

@router.get("/{workspace_id}/members", response_model=List[WorkspaceMemberResponse])
async def list_members(
    workspace_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List workspace members."""
    workspace = await get_workspace_or_404(workspace_id, db, current_user)
    
    result = await db.execute(
        select(WorkspaceMember)
        .where(WorkspaceMember.workspace_id == workspace.id)
        .order_by(WorkspaceMember.joined_at.desc())
    )
    members = result.scalars().all()
    
    # Load user data for each member
    responses = []
    for member in members:
        user_result = await db.execute(
            select(User).where(User.id == member.user_id)
        )
        user = user_result.scalar_one_or_none()
        
        member_response = WorkspaceMemberResponse.model_validate(member)
        if user:
            member_response.user = UserResponse.model_validate(user)
        responses.append(member_response)
    
    return responses


@router.post("/{workspace_id}/members", response_model=WorkspaceMemberResponse, status_code=status.HTTP_201_CREATED)
async def invite_member(
    workspace_id: str,
    invite_data: WorkspaceMemberInvite,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Invite a user to the workspace."""
    workspace = await get_workspace_or_404(workspace_id, db, current_user)
    
    # Check if current user can invite (owner or admin)
    member_result = await db.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace.id,
            WorkspaceMember.user_id == current_user.id
        )
    )
    current_member = member_result.scalar_one_or_none()
    
    if not current_member or current_member.role not in ["owner", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only owners and admins can invite members"
        )
    
    # Find user by email
    user_result = await db.execute(
        select(User).where(User.email == invite_data.email)
    )
    user = user_result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User with this email not found"
        )
    
    # Check if already a member
    existing_member = await db.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace.id,
            WorkspaceMember.user_id == user.id
        )
    )
    if existing_member.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is already a member of this workspace"
        )
    
    # Create membership
    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=user.id,
        role=invite_data.role.value if hasattr(invite_data.role, 'value') else invite_data.role
    )
    db.add(member)
    await db.commit()
    await db.refresh(member)
    
    # TODO: Send invite email
    
    response = WorkspaceMemberResponse.model_validate(member)
    response.user = UserResponse.model_validate(user)
    
    return response


@router.patch("/{workspace_id}/members/{member_id}", response_model=WorkspaceMemberResponse)
async def update_member(
    workspace_id: str,
    member_id: str,
    update_data: WorkspaceMemberUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a member's role."""
    workspace = await get_workspace_or_404(workspace_id, db, current_user)
    
    # Only owner can change roles
    if workspace.owner_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only workspace owner can change member roles"
        )
    
    # Get member
    result = await db.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.id == member_id,
            WorkspaceMember.workspace_id == workspace.id
        )
    )
    member = result.scalar_one_or_none()
    
    if not member:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )
    
    # Can't change owner's role
    if member.role == "owner":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change the owner's role"
        )
    
    member.role = update_data.role.value if hasattr(update_data.role, 'value') else update_data.role
    await db.commit()
    await db.refresh(member)
    
    return WorkspaceMemberResponse.model_validate(member)


@router.delete("/{workspace_id}/members/{member_id}", response_model=MessageResponse)
async def remove_member(
    workspace_id: str,
    member_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Remove a member from the workspace."""
    workspace = await get_workspace_or_404(workspace_id, db, current_user)
    
    # Get current user's membership
    current_member_result = await db.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace.id,
            WorkspaceMember.user_id == current_user.id
        )
    )
    current_member = current_member_result.scalar_one_or_none()
    
    # Get target member
    result = await db.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.id == member_id,
            WorkspaceMember.workspace_id == workspace.id
        )
    )
    member = result.scalar_one_or_none()
    
    if not member:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )
    
    # Can't remove owner
    if member.role == "owner":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove the workspace owner"
        )
    
    # Check permissions (owner/admin can remove, or user can leave)
    can_remove = (
        workspace.owner_id == current_user.id or
        (current_member and current_member.role == "admin") or
        member.user_id == current_user.id
    )
    
    if not can_remove:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to remove this member"
        )
    
    await db.delete(member)
    await db.commit()
    
    return MessageResponse(message="Member removed successfully")