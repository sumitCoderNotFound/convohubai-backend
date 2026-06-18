"""
Shared dependencies for recruitment routers:
tenant resolution, role-based edit checks, and common 404/immutability helpers.
"""
from typing import Optional, Tuple
from uuid import UUID
from fastapi import Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User, Workspace, WorkspaceMember

# Roles permitted to create/edit recruitment data. "viewer"/"read_only" are read-only.
EDIT_ROLES = {"owner", "admin", "member", "recruiter", "hiring_manager"}


async def resolve_workspace(
    user: User,
    db: AsyncSession,
    workspace_id: Optional[UUID] = None,
) -> Tuple[Workspace, str]:
    """Return (workspace, role) for the user, enforcing membership."""
    target_id = workspace_id or user.current_workspace_id
    if not target_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No workspace selected")

    result = await db.execute(
        select(Workspace).where(Workspace.id == target_id, Workspace.is_deleted == False)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workspace not found")

    if workspace.owner_id == user.id:
        return workspace, "owner"

    member_result = await db.execute(
        select(WorkspaceMember).where(
            WorkspaceMember.workspace_id == workspace.id,
            WorkspaceMember.user_id == user.id,
        )
    )
    member = member_result.scalar_one_or_none()
    if member:
        return workspace, (member.role or "member")
    if user.is_superuser:
        return workspace, "admin"

    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You don't have access to this workspace")


class WorkspaceContext:
    def __init__(self, workspace: Workspace, role: str, user: User):
        self.workspace = workspace
        self.role = role
        self.user = user

    @property
    def id(self) -> UUID:
        return self.workspace.id

    def require_edit(self):
        if self.role not in EDIT_ROLES:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your role is read-only for this action",
            )


async def get_ctx(
    workspace_id: Optional[UUID] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> WorkspaceContext:
    workspace, role = await resolve_workspace(current_user, db, workspace_id)
    return WorkspaceContext(workspace, role, current_user)


def ensure_draft(version):
    """Block mutations on published/immutable interview versions."""
    if version.status != "draft" or version.is_immutable:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This interview version is published and immutable. Create a new draft to edit.",
        )


async def get_or_create_settings(workspace_id, db):
    """Fetch (or lazily create) the workspace's recruitment settings row."""
    from sqlalchemy import select
    from app.recruitment.models.settings import RecruitmentSettings
    res = await db.execute(select(RecruitmentSettings).where(RecruitmentSettings.workspace_id == workspace_id))
    s = res.scalar_one_or_none()
    if not s:
        s = RecruitmentSettings(workspace_id=workspace_id)
        db.add(s)
        await db.commit()
        await db.refresh(s)
    return s
