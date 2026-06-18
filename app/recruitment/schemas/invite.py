"""Invite schemas (Phase 2)."""
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import Optional, List
from datetime import datetime
from uuid import UUID


class InviteCreate(BaseModel):
    version_id: Optional[UUID] = None  # defaults to latest published version
    email: Optional[EmailStr] = None
    candidate_name: Optional[str] = None
    max_attempts: int = Field(1, ge=1, le=10)
    expires_at: Optional[datetime] = None


class InviteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    workspace_id: UUID
    template_id: UUID
    version_id: UUID
    job_position_id: Optional[UUID] = None
    token: str
    email: Optional[str] = None
    candidate_name: Optional[str] = None
    status: str
    max_attempts: int
    expires_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    candidate_id: Optional[UUID] = None
    application_id: Optional[UUID] = None
    created_at: datetime


class InviteWithLink(InviteResponse):
    invite_url: str


class InviteListResponse(BaseModel):
    items: List[InviteResponse]
    total: int


# ---------------- Bulk + email (Phase 5) ----------------
class BulkInviteRequest(BaseModel):
    emails: List[EmailStr] = Field(default_factory=list)
    send_email: bool = False
    base_url: Optional[str] = None
    version_id: Optional[UUID] = None


class BulkInviteItem(BaseModel):
    email: str
    token: str
    invite_url: str
    email_sent: bool = False


class BulkInviteResponse(BaseModel):
    items: List[BulkInviteItem] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class SendEmailRequest(BaseModel):
    base_url: Optional[str] = None


class SendEmailResponse(BaseModel):
    sent: bool
    message: str
