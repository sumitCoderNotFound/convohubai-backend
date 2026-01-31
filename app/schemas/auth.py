"""
ConvoHubAI - Authentication Schemas
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from uuid import UUID


# ============================================
# REQUEST SCHEMAS
# ============================================

class UserRegister(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: str = Field(..., min_length=1, max_length=255)
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123!",
                "full_name": "John Doe"
            }
        }


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123!"
            }
        }


class TokenRefresh(BaseModel):
    """Token refresh request."""
    refresh_token: str


class PasswordReset(BaseModel):
    """Password reset request."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation."""
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)


class PasswordChange(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)


# ============================================
# RESPONSE SCHEMAS
# ============================================

class Token(BaseModel):
    """Token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


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
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    """Authentication response with user and tokens."""
    user: UserResponse
    tokens: Token


class MessageResponse(BaseModel):
    """Simple message response."""
    message: str
    success: bool = True
