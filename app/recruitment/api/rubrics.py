"""Feature 5 - Rubric criteria + anchors, scoped to a draft interview version."""
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.recruitment.models.interview import InterviewVersion
from app.recruitment.models.rubric import RubricCriterion, ScoreAnchor
from app.recruitment.schemas.rubric import (
    CriterionCreate, CriterionUpdate, CriterionResponse, RubricResponse,
)
from app.recruitment.schemas.common import MessageResponse
from app.recruitment.api.deps import get_ctx, WorkspaceContext, ensure_draft

router = APIRouter(prefix="/recruitment/versions", tags=["Recruitment - Rubrics"])

SENSITIVE_TERMS = ["age", "gender", "sex", "race", "ethnic", "religion", "disab",
                   "nationality", "marital", "pregnan", "sexual orientation", "accent"]


async def _version_or_404(vid: UUID, ws_id: UUID, db: AsyncSession) -> InterviewVersion:
    res = await db.execute(select(InterviewVersion).where(
        InterviewVersion.id == vid, InterviewVersion.workspace_id == ws_id, InterviewVersion.is_deleted == False))
    v = res.scalar_one_or_none()
    if not v:
        raise HTTPException(status_code=404, detail="Interview version not found")
    return v


async def _criterion_or_404(cid: UUID, vid: UUID, db: AsyncSession) -> RubricCriterion:
    res = await db.execute(select(RubricCriterion).options(selectinload(RubricCriterion.anchors)).where(
        RubricCriterion.id == cid, RubricCriterion.version_id == vid))
    c = res.scalar_one_or_none()
    if not c:
        raise HTTPException(status_code=404, detail="Criterion not found")
    return c


def _is_sensitive(name: str) -> bool:
    return any(term in (name or "").lower() for term in SENSITIVE_TERMS)


@router.get("/{version_id}/rubric", response_model=RubricResponse)
async def get_rubric(version_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    await _version_or_404(version_id, ctx.id, db)
    rows = await db.execute(select(RubricCriterion).options(selectinload(RubricCriterion.anchors)).where(
        RubricCriterion.version_id == version_id).order_by(RubricCriterion.order_index))
    criteria = list(rows.scalars().all())
    total = round(sum(c.weight or 0 for c in criteria), 2)
    return RubricResponse(
        version_id=version_id, criteria=criteria, total_weight=total,
        weights_valid=(bool(criteria) and abs(total - 100.0) <= 0.5))


@router.post("/{version_id}/criteria", response_model=CriterionResponse, status_code=201)
async def add_criterion(version_id: UUID, payload: CriterionCreate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    version = await _version_or_404(version_id, ctx.id, db)
    ensure_draft(version)
    order_index = payload.order_index
    if order_index is None:
        count = await db.scalar(select(func.count()).select_from(
            select(RubricCriterion).where(RubricCriterion.version_id == version_id).subquery()))
        order_index = count or 0
    crit = RubricCriterion(
        workspace_id=ctx.id, version_id=version_id, name=payload.name, description=payload.description,
        weight=payload.weight, evidence_instructions=payload.evidence_instructions,
        order_index=order_index, is_blocked_sensitive=_is_sensitive(payload.name))
    db.add(crit)
    await db.flush()
    for a in payload.anchors:
        db.add(ScoreAnchor(workspace_id=ctx.id, criterion_id=crit.id, level=a.level.value, descriptor=a.descriptor))
    await db.commit()
    return await _criterion_or_404(crit.id, version_id, db)


@router.patch("/{version_id}/criteria/{criterion_id}", response_model=CriterionResponse)
async def update_criterion(version_id: UUID, criterion_id: UUID, payload: CriterionUpdate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    version = await _version_or_404(version_id, ctx.id, db)
    ensure_draft(version)
    crit = await _criterion_or_404(criterion_id, version_id, db)
    data = payload.model_dump(exclude_unset=True)
    anchors = data.pop("anchors", None)
    for k, v in data.items():
        setattr(crit, k, v)
    if "name" in data:
        crit.is_blocked_sensitive = _is_sensitive(crit.name)
    if anchors is not None:
        existing = await db.execute(select(ScoreAnchor).where(ScoreAnchor.criterion_id == crit.id))
        for old in existing.scalars().all():
            await db.delete(old)
        for a in anchors:
            lvl = a["level"].value if hasattr(a["level"], "value") else a["level"]
            db.add(ScoreAnchor(workspace_id=ctx.id, criterion_id=crit.id, level=lvl, descriptor=a["descriptor"]))
    await db.commit()
    return await _criterion_or_404(criterion_id, version_id, db)


@router.delete("/{version_id}/criteria/{criterion_id}", response_model=MessageResponse)
async def delete_criterion(version_id: UUID, criterion_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    version = await _version_or_404(version_id, ctx.id, db)
    ensure_draft(version)
    crit = await _criterion_or_404(criterion_id, version_id, db)
    await db.delete(crit)
    await db.commit()
    return MessageResponse(message="Criterion deleted")
