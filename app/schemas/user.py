"""
ConvoHubAI - User and Workspace Schemas
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from uuid import UUID
from app.models.user import UserRole


# ============================================
# USER SCHEMAS
# ============================================

class UserUpdate(BaseModel):
    """User update request."""
    full_name: Optional[str] = Field(None, max_length=255)
    phone: Optional[str] = Field(None, max_length=20)
    avatar_url: Optional[str] = Field(None, max_length=500)


class UserResponse(BaseModel):
    """User response."""
    id: UUID
    email: str
    full_name: Optional[str]
    avatar_url: Optional[str]
    phone: Optional[str]
    is_active: bool
    is_verified: bool
    current_workspace_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# ============================================
# WORKSPACE SCHEMAS
# ============================================

class WorkspaceCreate(BaseModel):
    """Workspace creation request."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    industry: Optional[str] = Field(None, max_length=100)
    timezone: str = "UTC"
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "State University",
                "description": "Admissions department AI assistant",
                "industry": "education",
                "timezone": "America/New_York"
            }
        }


class WorkspaceUpdate(BaseModel):
    """Workspace update request."""
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    logo_url: Optional[str] = Field(None, max_length=500)
    industry: Optional[str] = Field(None, max_length=100)
    timezone: Optional[str] = Field(None, max_length=50)


class WorkspaceResponse(BaseModel):
    """Workspace response."""
    id: UUID
    name: str
    slug: str
    description: Optional[str]
    logo_url: Optional[str]
    industry: Optional[str]
    timezone: str
    owner_id: UUID
    subscription_plan: str
    subscription_status: str
    trial_ends_at: Optional[datetime]
    max_agents: str
    max_conversations_per_month: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class WorkspaceWithStats(WorkspaceResponse):
    """Workspace response with statistics."""
    member_count: int = 0
    agent_count: int = 0
    conversation_count: int = 0


# ============================================
# WORKSPACE MEMBER SCHEMAS
# ============================================

class WorkspaceMemberInvite(BaseModel):
    """Invite member to workspace."""
    email: EmailStr
    role: UserRole = UserRole.MEMBER


class WorkspaceMemberUpdate(BaseModel):
    """Update workspace member."""
    role: UserRole


class WorkspaceMemberResponse(BaseModel):
    """Workspace member response."""
    id: UUID
    workspace_id: UUID
    user_id: UUID
    role: UserRole
    invited_at: datetime
    joined_at: Optional[datetime]
    user: Optional[UserResponse] = None
    
    class Config:
        from_attributes = True


# ============================================
# LIST RESPONSES
# ============================================

class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    items: List
    total: int
    page: int
    page_size: int
    total_pages: int


class WorkspaceListResponse(BaseModel):
    """List of workspaces."""
    items: List[WorkspaceResponse]
    total: int
