"""Feature 2 - Candidates + Applications API."""
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_

from app.core.database import get_db
from app.recruitment.models.candidate import Candidate, Application, ApplicationHistory
from app.recruitment.models.job import JobPosition
from app.recruitment.schemas.candidate import (
    CandidateCreate, CandidateUpdate, CandidateResponse, CandidateListResponse,
    ApplicationCreate, ApplicationDecision, ApplicationResponse, ApplicationListResponse,
    ApplicationHistoryResponse, BulkImportRequest, BulkImportResult,
)
from app.recruitment.schemas.common import MessageResponse
from app.recruitment.api.deps import get_ctx, WorkspaceContext

router = APIRouter(prefix="/recruitment", tags=["Recruitment - Candidates"])


async def _candidate_or_404(cid: UUID, ws_id: UUID, db: AsyncSession) -> Candidate:
    res = await db.execute(select(Candidate).where(
        Candidate.id == cid, Candidate.workspace_id == ws_id, Candidate.is_deleted == False))
    c = res.scalar_one_or_none()
    if not c:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return c


async def _application_or_404(aid: UUID, ws_id: UUID, db: AsyncSession) -> Application:
    res = await db.execute(select(Application).where(
        Application.id == aid, Application.workspace_id == ws_id, Application.is_deleted == False))
    a = res.scalar_one_or_none()
    if not a:
        raise HTTPException(status_code=404, detail="Application not found")
    return a


# ---------- Candidates ----------

@router.post("/candidates", response_model=CandidateResponse, status_code=201)
async def create_candidate(payload: CandidateCreate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Create, or upsert by email within the workspace (FR-CAN-001)."""
    ctx.require_edit()
    if payload.email:
        existing = await db.execute(select(Candidate).where(
            Candidate.workspace_id == ctx.id,
            func.lower(Candidate.email) == payload.email.lower(),
            Candidate.is_deleted == False))
        found = existing.scalar_one_or_none()
        if found:
            for k, v in payload.model_dump(exclude_unset=True).items():
                if k == "source":
                    continue
                setattr(found, k, v)
            await db.commit()
            await db.refresh(found)
            return found
    cand = Candidate(
        workspace_id=ctx.id, full_name=payload.full_name,
        email=payload.email, phone=payload.phone, language=payload.language,
        source=payload.source.value, tags=payload.tags, notes=payload.notes,
    )
    db.add(cand)
    await db.commit()
    await db.refresh(cand)
    return cand


@router.get("/candidates", response_model=CandidateListResponse)
async def list_candidates(
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
    ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db),
):
    base = select(Candidate).where(Candidate.workspace_id == ctx.id, Candidate.is_deleted == False)
    if search:
        like = f"%{search}%"
        base = base.where(or_(Candidate.full_name.ilike(like), Candidate.email.ilike(like)))
    total = await db.scalar(select(func.count()).select_from(base.subquery()))
    rows = await db.execute(base.order_by(Candidate.created_at.desc()).offset((page - 1) * page_size).limit(page_size))
    return CandidateListResponse(items=list(rows.scalars().all()), total=total or 0, page=page, page_size=page_size)


@router.get("/candidates/{candidate_id}", response_model=CandidateResponse)
async def get_candidate(candidate_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    return await _candidate_or_404(candidate_id, ctx.id, db)


@router.patch("/candidates/{candidate_id}", response_model=CandidateResponse)
async def update_candidate(candidate_id: UUID, payload: CandidateUpdate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    cand = await _candidate_or_404(candidate_id, ctx.id, db)
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(cand, k, v)
    await db.commit()
    await db.refresh(cand)
    return cand


@router.delete("/candidates/{candidate_id}", response_model=MessageResponse)
async def delete_candidate(candidate_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    cand = await _candidate_or_404(candidate_id, ctx.id, db)
    cand.is_deleted = True
    await db.commit()
    return MessageResponse(message="Candidate deleted")


# ---------- Applications ----------

@router.post("/applications", response_model=ApplicationResponse, status_code=201)
async def create_application(payload: ApplicationCreate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    await _candidate_or_404(payload.candidate_id, ctx.id, db)
    if payload.job_position_id:
        job = await db.execute(select(JobPosition).where(
            JobPosition.id == payload.job_position_id, JobPosition.workspace_id == ctx.id, JobPosition.is_deleted == False))
        if not job.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Job not found")
    app = Application(
        workspace_id=ctx.id, candidate_id=payload.candidate_id,
        job_position_id=payload.job_position_id, stage=payload.stage.value, status="active",
    )
    db.add(app)
    await db.flush()
    db.add(ApplicationHistory(
        workspace_id=ctx.id, application_id=app.id,
        from_stage=None, to_stage=app.stage, actor_user_id=ctx.user.id, reason="Application created",
    ))
    await db.commit()
    await db.refresh(app)
    return app


@router.get("/applications", response_model=ApplicationListResponse)
async def list_applications(
    job_position_id: Optional[UUID] = Query(None),
    stage: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
    ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db),
):
    base = select(Application).where(Application.workspace_id == ctx.id, Application.is_deleted == False)
    if job_position_id:
        base = base.where(Application.job_position_id == job_position_id)
    if stage:
        base = base.where(Application.stage == stage)
    if status_filter:
        base = base.where(Application.status == status_filter)
    total = await db.scalar(select(func.count()).select_from(base.subquery()))
    rows = await db.execute(base.order_by(Application.created_at.desc()).offset((page - 1) * page_size).limit(page_size))
    return ApplicationListResponse(items=list(rows.scalars().all()), total=total or 0, page=page, page_size=page_size)


@router.get("/applications/{application_id}", response_model=ApplicationResponse)
async def get_application(application_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    return await _application_or_404(application_id, ctx.id, db)


@router.post("/applications/{application_id}/decisions", response_model=ApplicationResponse)
async def decide_application(application_id: UUID, payload: ApplicationDecision, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Human stage transition with an audited reason (FR-REV-004)."""
    ctx.require_edit()
    app = await _application_or_404(application_id, ctx.id, db)
    prev = app.stage
    app.stage = payload.to_stage.value
    if payload.to_stage.value in ("rejected", "advanced", "withdrawn"):
        app.status = "closed" if payload.to_stage.value in ("rejected", "withdrawn") else app.status
    db.add(ApplicationHistory(
        workspace_id=ctx.id, application_id=app.id,
        from_stage=prev, to_stage=app.stage, actor_user_id=ctx.user.id, reason=payload.reason,
    ))
    await db.commit()
    await db.refresh(app)
    return app


@router.get("/applications/{application_id}/history", response_model=list[ApplicationHistoryResponse])
async def application_history(application_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    await _application_or_404(application_id, ctx.id, db)
    rows = await db.execute(select(ApplicationHistory).where(
        ApplicationHistory.application_id == application_id,
        ApplicationHistory.workspace_id == ctx.id,
    ).order_by(ApplicationHistory.created_at.asc()))
    return list(rows.scalars().all())


@router.post("/candidates/bulk-import", response_model=BulkImportResult)
async def bulk_import(payload: BulkImportRequest, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Create candidates in bulk (dedupe by email). Optionally attach them to a job."""
    ctx.require_edit()
    result = BulkImportResult()

    job = None
    if payload.job_position_id:
        job = (await db.execute(select(JobPosition).where(
            JobPosition.id == payload.job_position_id, JobPosition.workspace_id == ctx.id,
            JobPosition.is_deleted == False))).scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

    for i, row in enumerate(payload.rows):
        try:
            email = row.email.lower().strip()
            cand = (await db.execute(select(Candidate).where(
                Candidate.workspace_id == ctx.id, func.lower(Candidate.email) == email,
                Candidate.is_deleted == False))).scalar_one_or_none()
            if cand:
                result.matched += 1
                if row.full_name and not cand.full_name:
                    cand.full_name = row.full_name
            else:
                cand = Candidate(workspace_id=ctx.id, full_name=row.full_name, email=row.email,
                                 phone=row.phone, source="bulk_import")
                db.add(cand)
                await db.flush()
                result.created += 1
            result.candidate_ids.append(cand.id)

            if job:
                exists = (await db.execute(select(Application).where(
                    Application.workspace_id == ctx.id, Application.candidate_id == cand.id,
                    Application.job_position_id == job.id, Application.is_deleted == False))).scalar_one_or_none()
                if not exists:
                    app = Application(workspace_id=ctx.id, candidate_id=cand.id,
                                      job_position_id=job.id, stage="applied", status="active")
                    db.add(app)
                    await db.flush()
                    db.add(ApplicationHistory(workspace_id=ctx.id, application_id=app.id,
                                              from_stage=None, to_stage="applied",
                                              actor_user_id=ctx.user.id, reason="Bulk import"))
                    result.applications_created += 1
        except Exception as e:
            result.skipped += 1
            result.errors.append(f"Row {i + 1} ({row.email}): {e}")

    await db.commit()
    return result
