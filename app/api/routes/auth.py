"""
ConvoHubAI - Authentication API Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timedelta
from pydantic import BaseModel
import secrets
import os

# Google OAuth imports
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from app.core.database import get_db
from app.core.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user
)
from app.core.config import settings
from app.models.user import User, Workspace, WorkspaceMember, UserRole
from app.schemas.auth import (
    UserRegister,
    UserLogin,
    TokenRefresh,
    PasswordReset,
    PasswordResetConfirm,
    PasswordChange,
    Token,
    AuthResponse,
    MessageResponse,
    UserResponse
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# Schema for Google OAuth
class GoogleAuthRequest(BaseModel):
    token: str


def generate_slug(name: str) -> str:
    """Generate a URL-friendly slug from a name."""
    import re
    slug = name.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    return f"{slug}-{secrets.token_hex(4)}"


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserRegister,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user."""
    # Check if email already exists
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
        is_active=True,
        is_verified=False,
        verification_token=secrets.token_urlsafe(32),
        verification_token_expires=datetime.utcnow() + timedelta(days=7)
    )
    db.add(user)
    await db.flush()
    
    # Create default workspace
    workspace = Workspace(
        name=f"{user_data.full_name}'s Workspace",
        slug=generate_slug(user_data.full_name),
        owner_id=user.id,
        subscription_plan="free",
        trial_ends_at=datetime.utcnow() + timedelta(days=14)
    )
    db.add(workspace)
    await db.flush()
    
    # Add user as workspace owner - using string "owner" instead of UserRole.OWNER
    member = WorkspaceMember(
        workspace_id=workspace.id,
        user_id=user.id,
        role="owner",
        joined_at=datetime.utcnow()
    )
    db.add(member)
    
    # Set current workspace
    user.current_workspace_id = workspace.id
    
    await db.commit()
    await db.refresh(user)
    
    # Create tokens
    access_token = create_access_token(subject=str(user.id))
    refresh_token = create_refresh_token(subject=str(user.id))
    
    return AuthResponse(
        user=UserResponse.model_validate(user),
        tokens=Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.access_token_expire_minutes * 60
        )
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """Login with email and password."""
    # Find user
    result = await db.execute(select(User).where(User.email == credentials.email))
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled"
        )
    
    # Update login stats
    user.last_login = datetime.utcnow()
    user.login_count = str(int(user.login_count or "0") + 1)
    await db.commit()
    await db.refresh(user)
    
    # Create tokens
    access_token = create_access_token(subject=str(user.id))
    refresh_token = create_refresh_token(subject=str(user.id))
    
    return AuthResponse(
        user=UserResponse.model_validate(user),
        tokens=Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.access_token_expire_minutes * 60
        )
    )


@router.post("/google", response_model=AuthResponse)
async def google_auth(
    request: GoogleAuthRequest,
    db: AsyncSession = Depends(get_db)
):
    """Authenticate with Google OAuth."""
    try:
        # Get Google Client ID from environment
        google_client_id = settings.google_client_id
        if not google_client_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google OAuth not configured"
            )
        
        # Verify token with Google
        idinfo = id_token.verify_oauth2_token(
            request.token, 
            google_requests.Request(), 
            google_client_id
        )
        
        # Extract user info from Google token
        google_id = idinfo.get("sub")
        email = idinfo.get("email")
        full_name = idinfo.get("name")
        avatar_url = idinfo.get("picture")
        email_verified = idinfo.get("email_verified", False)
        
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email not provided by Google"
            )
        
        # Check if user exists
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if user:
            # Update existing user with Google info
            if not user.avatar_url and avatar_url:
                user.avatar_url = avatar_url
            if not user.is_verified and email_verified:
                user.is_verified = True
            user.last_login = datetime.utcnow()
            user.login_count = str(int(user.login_count or "0") + 1)
            await db.commit()
            await db.refresh(user)
        else:
            # Create new user
            user = User(
                email=email,
                hashed_password=hash_password(secrets.token_urlsafe(32)),  # Random password for OAuth users
                full_name=full_name or email.split('@')[0],
                avatar_url=avatar_url,
                is_active=True,
                is_verified=email_verified,
            )
            db.add(user)
            await db.flush()
            
            # Create default workspace
            workspace_name = full_name or email.split('@')[0]
            workspace = Workspace(
                name=f"{workspace_name}'s Workspace",
                slug=generate_slug(workspace_name),
                owner_id=user.id,
                subscription_plan="free",
                trial_ends_at=datetime.utcnow() + timedelta(days=14)
            )
            db.add(workspace)
            await db.flush()
            
            # Add user as workspace owner
            member = WorkspaceMember(
                workspace_id=workspace.id,
                user_id=user.id,
                role="owner",
                joined_at=datetime.utcnow()
            )
            db.add(member)
            
            # Set current workspace
            user.current_workspace_id = workspace.id
            
            await db.commit()
            await db.refresh(user)
        
        # Create tokens
        access_token = create_access_token(subject=str(user.id))
        refresh_token = create_refresh_token(subject=str(user.id))
        
        return AuthResponse(
            user=UserResponse.model_validate(user),
            tokens=Token(
                access_token=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                expires_in=settings.access_token_expire_minutes * 60
            )
        )
        
    except ValueError as e:
        # Invalid token
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token"
        )
    except Exception as e:
        print(f"Google auth error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    token_data: TokenRefresh,
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token using refresh token."""
    payload = decode_token(token_data.refresh_token)
    
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user_id = payload.get("sub")
    
    # Verify user exists and is active
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new tokens
    access_token = create_access_token(subject=str(user.id))
    new_refresh_token = create_refresh_token(subject=str(user.id))
    
    return Token(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60
    )


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_user)
):
    """Get current authenticated user."""
    return UserResponse.model_validate(current_user)


@router.post("/logout", response_model=MessageResponse)
async def logout(
    current_user: User = Depends(get_current_user)
):
    """Logout current user (client should discard tokens)."""
    # In a more complete implementation, we would:
    # 1. Add the token to a blacklist (Redis)
    # 2. Invalidate all refresh tokens for this user
    return MessageResponse(message="Successfully logged out")


@router.post("/password/change", response_model=MessageResponse)
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Change password for authenticated user."""
    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Update password
    current_user.hashed_password = hash_password(password_data.new_password)
    await db.commit()
    
    return MessageResponse(message="Password changed successfully")


@router.post("/password/reset", response_model=MessageResponse)
async def request_password_reset(
    reset_data: PasswordReset,
    db: AsyncSession = Depends(get_db)
):
    """Request password reset email."""
    result = await db.execute(select(User).where(User.email == reset_data.email))
    user = result.scalar_one_or_none()
    
    if user:
        # Generate reset token
        user.password_reset_token = secrets.token_urlsafe(32)
        user.password_reset_expires = datetime.utcnow() + timedelta(hours=24)
        await db.commit()
        
        # TODO: Send email with reset link
        # email_service.send_password_reset(user.email, user.password_reset_token)
    
    # Always return success to prevent email enumeration
    return MessageResponse(
        message="If the email exists, a password reset link has been sent"
    )


@router.post("/password/reset/confirm", response_model=MessageResponse)
async def confirm_password_reset(
    reset_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db)
):
    """Confirm password reset with token."""
    result = await db.execute(
        select(User).where(
            User.password_reset_token == reset_data.token,
            User.password_reset_expires > datetime.utcnow()
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    # Update password
    user.hashed_password = hash_password(reset_data.new_password)
    user.password_reset_token = None
    user.password_reset_expires = None
    await db.commit()
    
    return MessageResponse(message="Password has been reset successfully")

