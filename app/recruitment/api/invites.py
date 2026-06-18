"""Recruiter invites API (Phase 2 + Phase 5 bulk/email)."""
import asyncio
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.recruitment.models.interview import InterviewTemplate, InterviewVersion
from app.recruitment.models.invite import InterviewInvite
from app.recruitment.schemas.invite import (
    InviteCreate, InviteResponse, InviteWithLink, InviteListResponse,
    BulkInviteRequest, BulkInviteItem, BulkInviteResponse, SendEmailRequest, SendEmailResponse,
)
from app.recruitment.schemas.common import MessageResponse
from app.recruitment.services.tokens import new_token
from app.recruitment.services.email import send_email, invite_email_html, email_configured
from app.recruitment.api.deps import get_ctx, WorkspaceContext, get_or_create_settings

router = APIRouter(prefix="/recruitment", tags=["Recruitment - Invites"])


def _link(token: str) -> str:
    # Relative path; the recruiter UI prepends the public origin.
    return f"/candidate/interview/{token}"


async def _resolve_published_version(template_id: UUID, version_id, ctx, db):
    """Return (template, version) ensuring the version is published, or raise."""
    tpl = (await db.execute(select(InterviewTemplate).where(
        InterviewTemplate.id == template_id, InterviewTemplate.workspace_id == ctx.id,
        InterviewTemplate.is_deleted == False))).scalar_one_or_none()
    if not tpl:
        raise HTTPException(status_code=404, detail="Interview not found")
    vid = version_id or tpl.latest_published_version_id
    if not vid:
        raise HTTPException(status_code=400, detail="Publish the interview before inviting candidates")
    version = (await db.execute(select(InterviewVersion).where(
        InterviewVersion.id == vid, InterviewVersion.workspace_id == ctx.id))).scalar_one_or_none()
    if not version or version.status != "published":
        raise HTTPException(status_code=400, detail="Invites must point to a published version")
    return tpl, version


@router.post("/interviews/{template_id}/invites", response_model=InviteWithLink, status_code=201)
async def create_invite(template_id: UUID, payload: InviteCreate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    tpl = (await db.execute(select(InterviewTemplate).where(
        InterviewTemplate.id == template_id, InterviewTemplate.workspace_id == ctx.id,
        InterviewTemplate.is_deleted == False))).scalar_one_or_none()
    if not tpl:
        raise HTTPException(status_code=404, detail="Interview not found")

    version_id = payload.version_id or tpl.latest_published_version_id
    if not version_id:
        raise HTTPException(status_code=400, detail="Publish the interview before inviting candidates")
    version = (await db.execute(select(InterviewVersion).where(
        InterviewVersion.id == version_id, InterviewVersion.workspace_id == ctx.id))).scalar_one_or_none()
    if not version or version.status != "published":
        raise HTTPException(status_code=400, detail="Invites must point to a published version")

    invite = InterviewInvite(
        workspace_id=ctx.id, template_id=tpl.id, version_id=version.id,
        job_position_id=tpl.job_position_id, created_by_id=ctx.user.id,
        token=new_token(), email=payload.email, candidate_name=payload.candidate_name,
        max_attempts=payload.max_attempts, expires_at=payload.expires_at, status="pending",
    )
    db.add(invite)
    await db.commit()
    await db.refresh(invite)
    return InviteWithLink(**InviteResponse.model_validate(invite).model_dump(), invite_url=_link(invite.token))


@router.get("/interviews/{template_id}/invites", response_model=InviteListResponse)
async def list_invites(template_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    base = select(InterviewInvite).where(
        InterviewInvite.template_id == template_id, InterviewInvite.workspace_id == ctx.id,
        InterviewInvite.is_deleted == False)
    total = await db.scalar(select(func.count()).select_from(base.subquery()))
    rows = await db.execute(base.order_by(InterviewInvite.created_at.desc()))
    return InviteListResponse(items=list(rows.scalars().all()), total=total or 0)


@router.post("/invites/{invite_id}/revoke", response_model=MessageResponse)
async def revoke_invite(invite_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    inv = (await db.execute(select(InterviewInvite).where(
        InterviewInvite.id == invite_id, InterviewInvite.workspace_id == ctx.id))).scalar_one_or_none()
    if not inv:
        raise HTTPException(status_code=404, detail="Invite not found")
    inv.status = "revoked"
    await db.commit()
    return MessageResponse(message="Invite revoked")


@router.post("/interviews/{template_id}/invites/bulk", response_model=BulkInviteResponse)
async def bulk_create_invites(template_id: UUID, payload: BulkInviteRequest, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Generate invites for many emails at once, optionally emailing each link."""
    ctx.require_edit()
    tpl, version = await _resolve_published_version(template_id, payload.version_id, ctx, db)
    settings_row = await get_or_create_settings(ctx.id, db)

    out = BulkInviteResponse()
    base = (payload.base_url or "").rstrip("/")
    for email in payload.emails:
        try:
            inv = InterviewInvite(
                workspace_id=ctx.id, template_id=tpl.id, version_id=version.id,
                job_position_id=tpl.job_position_id, created_by_id=ctx.user.id,
                token=new_token(), email=str(email), max_attempts=1, status="pending",
            )
            db.add(inv)
            await db.flush()
            url = f"{base}{_link(inv.token)}" if base else _link(inv.token)
            sent = False
            if payload.send_email:
                html = invite_email_html(tpl.name, url, settings_row.brand_name or "")
                sent = await asyncio.to_thread(send_email, str(email), f"Interview invite: {tpl.name}", html, f"You're invited to interview for {tpl.name}: {url}")
                if sent:
                    from datetime import datetime
                    inv.sent_at = datetime.utcnow()
            out.items.append(BulkInviteItem(email=str(email), token=inv.token, invite_url=url, email_sent=sent))
        except Exception as e:
            out.errors.append(f"{email}: {e}")
    await db.commit()
    return out


@router.post("/invites/{invite_id}/send-email", response_model=SendEmailResponse)
async def send_invite_email(invite_id: UUID, payload: SendEmailRequest, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Email an existing invite's link to its candidate."""
    ctx.require_edit()
    inv = (await db.execute(select(InterviewInvite).where(
        InterviewInvite.id == invite_id, InterviewInvite.workspace_id == ctx.id))).scalar_one_or_none()
    if not inv:
        raise HTTPException(status_code=404, detail="Invite not found")
    if not inv.email:
        return SendEmailResponse(sent=False, message="This invite has no email address attached.")
    if not email_configured():
        return SendEmailResponse(sent=False, message="Email is not configured. Copy the link and share it manually.")
    tpl = (await db.execute(select(InterviewTemplate).where(InterviewTemplate.id == inv.template_id))).scalar_one_or_none()
    settings_row = await get_or_create_settings(ctx.id, db)
    base = (payload.base_url or "").rstrip("/")
    url = f"{base}{_link(inv.token)}" if base else _link(inv.token)
    html = invite_email_html(tpl.name if tpl else "your interview", url, settings_row.brand_name or "")
    sent = await asyncio.to_thread(send_email, inv.email, f"Interview invite: {tpl.name if tpl else 'Interview'}", html, f"Start your interview: {url}")
    if sent:
        from datetime import datetime
        inv.sent_at = datetime.utcnow()
        await db.commit()
        return SendEmailResponse(sent=True, message=f"Invite emailed to {inv.email}")
    return SendEmailResponse(sent=False, message="Could not send the email. Check SMTP settings or share the link manually.")
