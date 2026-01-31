"""
ConvoHubAI - Pydantic Schemas
"""
from app.schemas.auth import (
    UserRegister,
    UserLogin,
    TokenRefresh,
    PasswordReset,
    PasswordResetConfirm,
    PasswordChange,
    Token,
    AuthResponse,
    MessageResponse
)
from app.schemas.user import (
    UserUpdate,
    UserResponse,
    WorkspaceCreate,
    WorkspaceUpdate,
    WorkspaceResponse,
    WorkspaceWithStats,
    WorkspaceMemberInvite,
    WorkspaceMemberUpdate,
    WorkspaceMemberResponse,
    PaginatedResponse,
    WorkspaceListResponse
)
from app.schemas.agent import (
    AgentCreate,
    AgentUpdate,
    AgentResponse,
    AgentListResponse,
    AgentTemplateResponse,
    AgentTemplateListResponse,
    AgentDuplicate,
    AgentFromTemplate
)

__all__ = [
    # Auth
    "UserRegister",
    "UserLogin",
    "TokenRefresh",
    "PasswordReset",
    "PasswordResetConfirm",
    "PasswordChange",
    "Token",
    "AuthResponse",
    "MessageResponse",
    
    # User
    "UserUpdate",
    "UserResponse",
    "WorkspaceCreate",
    "WorkspaceUpdate",
    "WorkspaceResponse",
    "WorkspaceWithStats",
    "WorkspaceMemberInvite",
    "WorkspaceMemberUpdate",
    "WorkspaceMemberResponse",
    "PaginatedResponse",
    "WorkspaceListResponse",
    
    # Agent
    "AgentCreate",
    "AgentUpdate",
    "AgentResponse",
    "AgentListResponse",
    "AgentTemplateResponse",
    "AgentTemplateListResponse",
    "AgentDuplicate",
    "AgentFromTemplate",
]
