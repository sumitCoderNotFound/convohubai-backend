"""Feature 3 - Interview templates + versioning (creator, publish, clone, generate)."""
from datetime import datetime
from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.recruitment.models.interview import InterviewTemplate, InterviewVersion
from app.recruitment.models.question import InterviewQuestion
from app.recruitment.models.rubric import RubricCriterion, ScoreAnchor
from app.recruitment.models.job import JobPosition
from app.recruitment.schemas.interview import (
    InterviewCreate, InterviewVersionUpdate, InterviewResponse, InterviewListResponse,
    InterviewVersionResponse, PublishResult, GenerateRequest,
)
from app.recruitment.schemas.common import MessageResponse
from app.recruitment.services.interview_generation import generate_interview_content
from app.recruitment.api.deps import get_ctx, WorkspaceContext, ensure_draft

router = APIRouter(prefix="/recruitment/interviews", tags=["Recruitment - Interviews"])

# Heuristic block-list for sensitive/protected scoring criteria (FR-RUB-006).
SENSITIVE_TERMS = [
    "age", "gender", "sex", "race", "ethnic", "religion", "disab",
    "nationality", "marital", "pregnan", "sexual orientation", "accent",
]


async def _template_or_404(tid: UUID, ws_id: UUID, db: AsyncSession) -> InterviewTemplate:
    res = await db.execute(
        select(InterviewTemplate)
        .options(selectinload(InterviewTemplate.versions))
        .where(InterviewTemplate.id == tid, InterviewTemplate.workspace_id == ws_id, InterviewTemplate.is_deleted == False)
    )
    t = res.scalar_one_or_none()
    if not t:
        raise HTTPException(status_code=404, detail="Interview not found")
    return t


async def _version_or_404(vid: UUID, ws_id: UUID, db: AsyncSession) -> InterviewVersion:
    res = await db.execute(select(InterviewVersion).where(
        InterviewVersion.id == vid, InterviewVersion.workspace_id == ws_id, InterviewVersion.is_deleted == False))
    v = res.scalar_one_or_none()
    if not v:
        raise HTTPException(status_code=404, detail="Interview version not found")
    return v


async def _current_draft(template: InterviewTemplate, db: AsyncSession) -> Optional[InterviewVersion]:
    res = await db.execute(select(InterviewVersion).where(
        InterviewVersion.template_id == template.id, InterviewVersion.status == "draft",
        InterviewVersion.is_deleted == False).order_by(InterviewVersion.version_number.desc()))
    return res.scalars().first()


async def _clone_content(src_version_id: UUID, dst_version: InterviewVersion, ws_id: UUID, db: AsyncSession):
    """Copy questions, criteria and anchors from a source version into a draft."""
    qres = await db.execute(select(InterviewQuestion).where(
        InterviewQuestion.version_id == src_version_id).order_by(InterviewQuestion.order_index))
    for q in qres.scalars().all():
        db.add(InterviewQuestion(
            workspace_id=ws_id, version_id=dst_version.id, order_index=q.order_index,
            question_type=q.question_type, prompt_text=q.prompt_text, config=q.config,
            is_knockout=q.is_knockout))
    cres = await db.execute(select(RubricCriterion).options(selectinload(RubricCriterion.anchors)).where(
        RubricCriterion.version_id == src_version_id).order_by(RubricCriterion.order_index))
    for c in cres.scalars().all():
        new_c = RubricCriterion(
            workspace_id=ws_id, version_id=dst_version.id, name=c.name, description=c.description,
            weight=c.weight, evidence_instructions=c.evidence_instructions, order_index=c.order_index)
        db.add(new_c)
        await db.flush()
        for a in c.anchors:
            db.add(ScoreAnchor(workspace_id=ws_id, criterion_id=new_c.id, level=a.level, descriptor=a.descriptor))


@router.post("", response_model=InterviewResponse, status_code=201)
async def create_interview(payload: InterviewCreate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Create a template and its first draft version (optionally cloned)."""
    ctx.require_edit()
    if payload.job_position_id:
        j = await db.execute(select(JobPosition).where(
            JobPosition.id == payload.job_position_id, JobPosition.workspace_id == ctx.id, JobPosition.is_deleted == False))
        if not j.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Job not found")

    template = InterviewTemplate(
        workspace_id=ctx.id, job_position_id=payload.job_position_id,
        created_by_id=ctx.user.id, name=payload.name, description=payload.description)
    db.add(template)
    await db.flush()

    version = InterviewVersion(
        workspace_id=ctx.id, template_id=template.id, version_number=1,
        status="draft", mode=payload.mode.value, language=payload.language)
    db.add(version)
    await db.flush()

    if payload.clone_from_version_id:
        src = await _version_or_404(payload.clone_from_version_id, ctx.id, db)
        await _clone_content(src.id, version, ctx.id, db)

    await db.commit()
    return await _template_or_404(template.id, ctx.id, db)


@router.get("", response_model=InterviewListResponse)
async def list_interviews(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
    ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db),
):
    base = select(InterviewTemplate).where(
        InterviewTemplate.workspace_id == ctx.id, InterviewTemplate.is_deleted == False)
    total = await db.scalar(select(func.count()).select_from(base.subquery()))
    rows = await db.execute(base.options(selectinload(InterviewTemplate.versions)).order_by(
        InterviewTemplate.created_at.desc()).offset((page - 1) * page_size).limit(page_size))
    return InterviewListResponse(items=list(rows.scalars().all()), total=total or 0, page=page, page_size=page_size)


@router.get("/{template_id}", response_model=InterviewResponse)
async def get_interview(template_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    return await _template_or_404(template_id, ctx.id, db)


@router.get("/{template_id}/draft", response_model=InterviewVersionResponse)
async def get_draft(template_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    template = await _template_or_404(template_id, ctx.id, db)
    draft = await _current_draft(template, db)
    if not draft:
        raise HTTPException(status_code=404, detail="No draft version; create a new draft to edit")
    return draft


@router.patch("/{template_id}/draft", response_model=InterviewVersionResponse)
async def update_draft(template_id: UUID, payload: InterviewVersionUpdate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    template = await _template_or_404(template_id, ctx.id, db)
    draft = await _current_draft(template, db)
    if not draft:
        raise HTTPException(status_code=404, detail="No draft version to edit")
    ensure_draft(draft)
    data = payload.model_dump(exclude_unset=True)
    if "mode" in data and data["mode"] is not None:
        data["mode"] = data["mode"].value
    for k, v in data.items():
        setattr(draft, k, v)
    await db.commit()
    await db.refresh(draft)
    return draft


@router.post("/{template_id}/new-draft", response_model=InterviewVersionResponse, status_code=201)
async def create_new_draft(template_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Clone the latest version into a fresh editable draft."""
    ctx.require_edit()
    template = await _template_or_404(template_id, ctx.id, db)
    if await _current_draft(template, db):
        raise HTTPException(status_code=409, detail="A draft already exists for this interview")
    latest = sorted(template.versions, key=lambda v: v.version_number)[-1] if template.versions else None
    new_num = (latest.version_number + 1) if latest else 1
    draft = InterviewVersion(
        workspace_id=ctx.id, template_id=template.id, version_number=new_num, status="draft",
        mode=latest.mode if latest else "voice_only", language=latest.language if latest else "en",
        introduction=latest.introduction if latest else None,
        instructions=latest.instructions if latest else None,
        recording_enabled=latest.recording_enabled if latest else False)
    db.add(draft)
    await db.flush()
    if latest:
        await _clone_content(latest.id, draft, ctx.id, db)
    await db.commit()
    await db.refresh(draft)
    return draft


def _validate_publish(version, questions, criteria) -> List[str]:
    errors: List[str] = []
    if not questions:
        errors.append("Add at least one question before publishing.")
    if not criteria:
        errors.append("Add at least one rubric criterion before publishing.")
    total_weight = round(sum(c.weight or 0 for c in criteria), 2)
    if criteria and abs(total_weight - 100.0) > 0.5:
        errors.append(f"Rubric weights must total 100 (currently {total_weight}).")
    for c in criteria:
        if any(term in (c.name or "").lower() for term in SENSITIVE_TERMS):
            errors.append(f"Criterion '{c.name}' looks like a protected characteristic and is blocked.")
        if not c.anchors:
            errors.append(f"Criterion '{c.name}' needs weak/moderate/strong anchors.")
    return errors


@router.post("/{template_id}/publish", response_model=PublishResult)
async def publish_interview(template_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Validate and publish the draft into an immutable version (FR-INT-005)."""
    ctx.require_edit()
    template = await _template_or_404(template_id, ctx.id, db)
    draft = await _current_draft(template, db)
    if not draft:
        raise HTTPException(status_code=404, detail="No draft version to publish")

    qres = await db.execute(select(InterviewQuestion).where(InterviewQuestion.version_id == draft.id))
    questions = list(qres.scalars().all())
    cres = await db.execute(select(RubricCriterion).options(selectinload(RubricCriterion.anchors)).where(
        RubricCriterion.version_id == draft.id))
    criteria = list(cres.scalars().all())

    errors = _validate_publish(draft, questions, criteria)
    if errors:
        return PublishResult(published=False, version_id=draft.id, version_number=draft.version_number, errors=errors)

    draft.status = "published"
    draft.is_immutable = True
    draft.published_at = datetime.utcnow()
    draft.published_by_id = ctx.user.id
    template.latest_published_version_id = draft.id
    await db.commit()
    await db.refresh(draft)
    return PublishResult(published=True, version_id=draft.id, version_number=draft.version_number, errors=[])


@router.post("/{template_id}/generate", response_model=InterviewVersionResponse)
async def generate_content(template_id: UUID, payload: GenerateRequest, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Populate the draft with AI-generated questions + rubric from job context."""
    ctx.require_edit()
    template = await _template_or_404(template_id, ctx.id, db)
    draft = await _current_draft(template, db)
    if not draft:
        raise HTTPException(status_code=404, detail="No draft version to generate into")
    ensure_draft(draft)

    context = payload.context or ""
    if not context and template.job_position_id:
        j = await db.execute(select(JobPosition).where(JobPosition.id == template.job_position_id))
        job = j.scalar_one_or_none()
        if job:
            context = f"{job.title}\n{job.description or ''}"
    if not context:
        raise HTTPException(status_code=400, detail="Provide context or link a job with a description")

    result = await generate_interview_content(context, payload.num_questions, language=getattr(draft, "language", "en") or "en")

    start_idx = await db.scalar(select(func.count()).select_from(
        select(InterviewQuestion).where(InterviewQuestion.version_id == draft.id).subquery())) or 0
    for i, q in enumerate(result["questions"]):
        db.add(InterviewQuestion(
            workspace_id=ctx.id, version_id=draft.id, order_index=start_idx + i,
            question_type=q.get("question_type", "open_response"),
            prompt_text=q.get("prompt_text", ""), config={"required": True, "probing_depth": 1}))
    for i, c in enumerate(result["criteria"]):
        new_c = RubricCriterion(
            workspace_id=ctx.id, version_id=draft.id, name=c.get("name", f"Criterion {i+1}"),
            description=c.get("description"), weight=float(c.get("weight", 0) or 0), order_index=i)
        db.add(new_c)
        await db.flush()
        anchors = c.get("anchors", {}) or {}
        for level in ("weak", "moderate", "strong"):
            if anchors.get(level):
                db.add(ScoreAnchor(workspace_id=ctx.id, criterion_id=new_c.id, level=level, descriptor=anchors[level]))
    await db.commit()
    await db.refresh(draft)
    return draft


@router.delete("/{template_id}", response_model=MessageResponse)
async def archive_interview(template_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    template = await _template_or_404(template_id, ctx.id, db)
    template.is_deleted = True
    template.archived = True
    await db.commit()
    return MessageResponse(message="Interview archived")
