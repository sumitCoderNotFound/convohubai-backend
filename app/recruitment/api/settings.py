"""Recruiter settings API (the three product decisions; Phase 2)."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.recruitment.schemas.settings import SettingsResponse, SettingsUpdate
from app.recruitment.api.deps import get_ctx, WorkspaceContext, get_or_create_settings

router = APIRouter(prefix="/recruitment/settings", tags=["Recruitment - Settings"])


@router.get("", response_model=SettingsResponse)
async def get_settings(ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    return await get_or_create_settings(ctx.id, db)


@router.patch("", response_model=SettingsResponse)
async def update_settings(payload: SettingsUpdate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    s = await get_or_create_settings(ctx.id, db)
    data = payload.model_dump(exclude_unset=True)
    if "jurisdiction" in data and data["jurisdiction"] is not None:
        data["jurisdiction"] = data["jurisdiction"].value
    for k, v in data.items():
        setattr(s, k, v)
    await db.commit()
    await db.refresh(s)
    return s
