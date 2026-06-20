"""Email template editor API (Phase 13)."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.recruitment.models.email_template import EmailTemplate
from app.recruitment.schemas.email_template import EmailTemplateUpsert, EmailTemplateItem, EmailTemplatesView
from app.recruitment.schemas.common import MessageResponse
from app.recruitment.services.email import TEMPLATE_KINDS, TEMPLATE_VARIABLES, DEFAULT_TEMPLATES
from app.recruitment.api.deps import get_ctx, WorkspaceContext

router = APIRouter(prefix="/recruitment/email-templates", tags=["Recruitment - Email templates"])


@router.get("", response_model=EmailTemplatesView)
async def list_templates(ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    rows = await db.execute(select(EmailTemplate).where(
        EmailTemplate.workspace_id == ctx.id, EmailTemplate.is_deleted == False))
    custom = {t.kind: t for t in rows.scalars().all()}
    items = []
    for kind in TEMPLATE_KINDS:
        if kind in custom:
            t = custom[kind]
            items.append(EmailTemplateItem(kind=kind, subject=t.subject, body_html=t.body_html, enabled=t.enabled, is_custom=True))
        else:
            d = DEFAULT_TEMPLATES[kind]
            items.append(EmailTemplateItem(kind=kind, subject=d["subject"], body_html=d["body_html"], enabled=True, is_custom=False))
    return EmailTemplatesView(variables=TEMPLATE_VARIABLES, templates=items)


@router.put("/{kind}", response_model=EmailTemplateItem)
async def upsert_template(kind: str, payload: EmailTemplateUpsert, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    if kind not in TEMPLATE_KINDS:
        raise HTTPException(status_code=400, detail=f"Unknown template kind. Allowed: {', '.join(TEMPLATE_KINDS)}")
    t = (await db.execute(select(EmailTemplate).where(
        EmailTemplate.workspace_id == ctx.id, EmailTemplate.kind == kind, EmailTemplate.is_deleted == False))).scalar_one_or_none()
    if t:
        t.subject = payload.subject
        t.body_html = payload.body_html
        t.enabled = payload.enabled
    else:
        t = EmailTemplate(workspace_id=ctx.id, kind=kind, subject=payload.subject, body_html=payload.body_html, enabled=payload.enabled)
        db.add(t)
    await db.commit()
    await db.refresh(t)
    return EmailTemplateItem(kind=t.kind, subject=t.subject, body_html=t.body_html, enabled=t.enabled, is_custom=True)


@router.delete("/{kind}", response_model=MessageResponse)
async def reset_template(kind: str, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Revert to the built-in default."""
    ctx.require_edit()
    t = (await db.execute(select(EmailTemplate).where(
        EmailTemplate.workspace_id == ctx.id, EmailTemplate.kind == kind, EmailTemplate.is_deleted == False))).scalar_one_or_none()
    if t:
        await db.delete(t)
        await db.commit()
    return MessageResponse(message="Reverted to default")
