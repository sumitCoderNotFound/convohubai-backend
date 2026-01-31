"""
ConvoHubAI - User Model
"""

from sqlalchemy import Column, String, Boolean, DateTime, Enum, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import enum
from datetime import datetime

from app.models.base import BaseModel


class UserRole(str, enum.Enum):
    """User roles."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"

    def __str__(self):
        return self.value


class User(BaseModel):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    # Basic info
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    phone = Column(String(20), nullable=True)

    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    # Verification
    verification_token = Column(String(255), nullable=True)
    verification_token_expires = Column(DateTime, nullable=True)

    # Password reset
    password_reset_token = Column(String(255), nullable=True)
    password_reset_expires = Column(DateTime, nullable=True)

    # Tracking
    last_login = Column(DateTime, nullable=True)
    login_count = Column(String(10), default="0")

    # Current workspace
    current_workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relationships
    owned_workspaces = relationship(
        "Workspace", back_populates="owner", foreign_keys="Workspace.owner_id"
    )
    workspace_memberships = relationship(
        "WorkspaceMember", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<User {self.email}>"


class Workspace(BaseModel):
    """Workspace/Organization model."""

    __tablename__ = "workspaces"

    # Basic info
    name = Column(String(255), nullable=False)
    slug = Column(String(255), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    logo_url = Column(String(500), nullable=True)

    # Settings
    industry = Column(String(100), nullable=True)  # education, hospitality, etc.
    timezone = Column(String(50), default="UTC")

    # Owner
    owner_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    # Billing
    stripe_customer_id = Column(String(255), nullable=True)
    subscription_plan = Column(
        String(50), default="free"
    )  # free, starter, professional, enterprise
    subscription_status = Column(
        String(50), default="active"
    )  # active, canceled, past_due
    trial_ends_at = Column(DateTime, nullable=True)

    # Usage limits
    max_agents = Column(String(10), default="1")
    max_conversations_per_month = Column(String(10), default="500")

    # Relationships
    owner = relationship(
        "User", back_populates="owned_workspaces", foreign_keys=[owner_id]
    )
    members = relationship(
        "WorkspaceMember", back_populates="workspace", cascade="all, delete-orphan"
    )
    agents = relationship(
        "Agent", back_populates="workspace", cascade="all, delete-orphan"
    )
    knowledge_bases = relationship(
        "KnowledgeBase", back_populates="workspace", cascade="all, delete-orphan"
    )
    phone_numbers = relationship(
        "PhoneNumber", back_populates="workspace", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Workspace {self.name}>"


class WorkspaceMember(BaseModel):
    """Workspace membership model."""

    __tablename__ = "workspace_members"

    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    role = Column(String(20), default="member", nullable=False)
    invited_at = Column(DateTime, default=datetime.utcnow)
    joined_at = Column(DateTime, nullable=True)

    # Relationships
    workspace = relationship("Workspace", back_populates="members")
    user = relationship("User", back_populates="workspace_memberships")

    def __repr__(self):
        return f"<WorkspaceMember {self.user_id} in {self.workspace_id}>"
