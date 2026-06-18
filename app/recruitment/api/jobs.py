"""Feature 1 - Jobs API."""
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_

from app.core.database import get_db
from app.recruitment.models.job import JobPosition
from app.recruitment.schemas.job import (
    JobCreate, JobUpdate, JobResponse, JobListResponse,
    ParseJobDescriptionRequest, ParseJobDescriptionResponse,
)
from app.recruitment.schemas.common import MessageResponse
from app.recruitment.services.job_parsing import parse_job_description
from app.recruitment.api.deps import get_ctx, WorkspaceContext

router = APIRouter(prefix="/recruitment/jobs", tags=["Recruitment - Jobs"])


async def _get_job_or_404(job_id: UUID, ws_id: UUID, db: AsyncSession) -> JobPosition:
    result = await db.execute(
        select(JobPosition).where(
            JobPosition.id == job_id,
            JobPosition.workspace_id == ws_id,
            JobPosition.is_deleted == False,
        )
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


@router.post("", response_model=JobResponse, status_code=201)
async def create_job(payload: JobCreate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    job = JobPosition(
        workspace_id=ctx.id,
        created_by_id=ctx.user.id,
        title=payload.title,
        description=payload.description,
        department=payload.department,
        location=payload.location,
        employment_type=payload.employment_type.value,
        competency_profile=payload.competency_profile,
        required_criteria=payload.required_criteria,
        preferred_criteria=payload.preferred_criteria,
        disqualifying_criteria=payload.disqualifying_criteria,
        is_general_assessment=payload.is_general_assessment,
        status="draft",
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status_filter: Optional[str] = Query(None, alias="status"),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    ctx: WorkspaceContext = Depends(get_ctx),
    db: AsyncSession = Depends(get_db),
):
    base = select(JobPosition).where(JobPosition.workspace_id == ctx.id, JobPosition.is_deleted == False)
    if status_filter:
        base = base.where(JobPosition.status == status_filter)
    if search:
        like = f"%{search}%"
        base = base.where(or_(JobPosition.title.ilike(like), JobPosition.department.ilike(like)))

    total = await db.scalar(select(func.count()).select_from(base.subquery()))
    rows = await db.execute(
        base.order_by(JobPosition.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    )
    return JobListResponse(items=list(rows.scalars().all()), total=total or 0, page=page, page_size=page_size)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    return await _get_job_or_404(job_id, ctx.id, db)


@router.patch("/{job_id}", response_model=JobResponse)
async def update_job(job_id: UUID, payload: JobUpdate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    job = await _get_job_or_404(job_id, ctx.id, db)
    data = payload.model_dump(exclude_unset=True)
    if "employment_type" in data and data["employment_type"] is not None:
        data["employment_type"] = data["employment_type"].value
    if "status" in data and data["status"] is not None:
        data["status"] = data["status"].value
    for k, v in data.items():
        setattr(job, k, v)
    await db.commit()
    await db.refresh(job)
    return job


@router.post("/{job_id}/duplicate", response_model=JobResponse, status_code=201)
async def duplicate_job(job_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    src = await _get_job_or_404(job_id, ctx.id, db)
    copy = JobPosition(
        workspace_id=ctx.id, created_by_id=ctx.user.id,
        title=f"{src.title} (copy)", description=src.description,
        department=src.department, location=src.location,
        employment_type=src.employment_type, competency_profile=src.competency_profile,
        required_criteria=src.required_criteria, preferred_criteria=src.preferred_criteria,
        disqualifying_criteria=src.disqualifying_criteria, status="draft",
    )
    db.add(copy)
    await db.commit()
    await db.refresh(copy)
    return copy


@router.post("/{job_id}/close", response_model=JobResponse)
async def close_job(job_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    job = await _get_job_or_404(job_id, ctx.id, db)
    job.status = "closed"
    await db.commit()
    await db.refresh(job)
    return job


@router.delete("/{job_id}", response_model=MessageResponse)
async def archive_job(job_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    job = await _get_job_or_404(job_id, ctx.id, db)
    job.is_deleted = True
    job.status = "archived"
    await db.commit()
    return MessageResponse(message="Job archived")


@router.post("/parse-description", response_model=ParseJobDescriptionResponse)
async def parse_description(payload: ParseJobDescriptionRequest, ctx: WorkspaceContext = Depends(get_ctx)):
    """FR-JOB-002: draft an editable competency profile from a pasted job description."""
    ctx.require_edit()
    result = await parse_job_description(payload.description)
    return ParseJobDescriptionResponse(**result)
