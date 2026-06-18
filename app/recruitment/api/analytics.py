"""Recruiter dashboard + per-job shortlist/ranking API (Phase 4)."""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.recruitment.models.job import JobPosition
from app.recruitment.models.candidate import Candidate, Application
from app.recruitment.models.interview import InterviewTemplate
from app.recruitment.models.session import InterviewSession
from app.recruitment.models.score import InterviewScore
from app.recruitment.schemas.analytics import (
    DashboardResponse, RecentSession, ShortlistResponse, ShortlistItem,
)
from app.recruitment.api.deps import get_ctx, WorkspaceContext

router = APIRouter(prefix="/recruitment", tags=["Recruitment - Dashboard & Shortlist"])


@router.get("/dashboard", response_model=DashboardResponse)
async def dashboard(ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ws = ctx.id

    async def count(model, *conds):
        q = select(func.count()).select_from(model).where(model.workspace_id == ws, model.is_deleted == False, *conds)
        return (await db.scalar(q)) or 0

    jobs_open = await count(JobPosition, JobPosition.status == "open")
    candidates_total = await count(Candidate)
    applications_total = await count(Application)
    sessions_total = await count(InterviewSession)
    sessions_completed = await count(InterviewSession, InterviewSession.status == "completed")
    interviews_published = await count(InterviewTemplate, InterviewTemplate.latest_published_version_id.isnot(None))

    # applications by stage
    stage_rows = await db.execute(
        select(Application.stage, func.count()).where(
            Application.workspace_id == ws, Application.is_deleted == False
        ).group_by(Application.stage)
    )
    applications_by_stage = {stage: n for stage, n in stage_rows.all()}

    # scoring stats
    scored_count = (await db.scalar(
        select(func.count()).select_from(InterviewScore).where(
            InterviewScore.workspace_id == ws, InterviewScore.is_deleted == False,
            InterviewScore.status == "completed")
    )) or 0
    avg_score = await db.scalar(
        select(func.avg(InterviewScore.overall_score)).where(
            InterviewScore.workspace_id == ws, InterviewScore.is_deleted == False,
            InterviewScore.status == "completed")
    )
    needs_review_count = (await db.scalar(
        select(func.count()).select_from(InterviewScore).where(
            InterviewScore.workspace_id == ws, InterviewScore.is_deleted == False,
            InterviewScore.needs_human_review == True)
    )) or 0

    # recent completed sessions (last 5) with candidate + job title
    recent_rows = await db.execute(
        select(InterviewSession, Candidate.full_name, JobPosition.title, InterviewScore.overall_score, InterviewScore.recommendation)
        .outerjoin(Candidate, Candidate.id == InterviewSession.candidate_id)
        .outerjoin(Application, Application.id == InterviewSession.application_id)
        .outerjoin(JobPosition, JobPosition.id == Application.job_position_id)
        .outerjoin(InterviewScore, InterviewScore.session_id == InterviewSession.id)
        .where(InterviewSession.workspace_id == ws, InterviewSession.status == "completed",
               InterviewSession.is_deleted == False)
        .order_by(InterviewSession.completed_at.desc().nullslast())
        .limit(5)
    )
    recent = [
        RecentSession(
            session_id=s.id, candidate_name=name, job_title=title,
            overall_score=score, recommendation=rec, completed_at=s.completed_at,
        )
        for s, name, title, score, rec in recent_rows.all()
    ]

    return DashboardResponse(
        jobs_open=jobs_open, candidates_total=candidates_total, applications_total=applications_total,
        applications_by_stage=applications_by_stage, interviews_published=interviews_published,
        sessions_total=sessions_total, sessions_completed=sessions_completed,
        scored_count=scored_count, avg_score=round(avg_score, 1) if avg_score is not None else None,
        needs_review_count=needs_review_count, recent=recent,
    )


@router.get("/jobs/{job_id}/shortlist", response_model=ShortlistResponse)
async def shortlist(job_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ws = ctx.id
    job = (await db.execute(select(JobPosition).where(
        JobPosition.id == job_id, JobPosition.workspace_id == ws, JobPosition.is_deleted == False))).scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Ranked: applications on this job that have a completed, scored session.
    rows = await db.execute(
        select(Application, Candidate, InterviewSession, InterviewScore)
        .join(InterviewSession, InterviewSession.application_id == Application.id)
        .join(InterviewScore, InterviewScore.session_id == InterviewSession.id)
        .outerjoin(Candidate, Candidate.id == Application.candidate_id)
        .where(Application.workspace_id == ws, Application.job_position_id == job_id,
               Application.is_deleted == False, InterviewScore.status == "completed")
        .order_by(InterviewScore.overall_score.desc().nullslast())
    )
    items = []
    seen_apps = set()
    for app, cand, sess, score in rows.all():
        if app.id in seen_apps:  # one row per application (best/most-recent score)
            continue
        seen_apps.add(app.id)
        items.append(ShortlistItem(
            application_id=app.id, candidate_id=cand.id if cand else None,
            candidate_name=cand.full_name if cand else None,
            candidate_email=cand.email if cand else None,
            session_id=sess.id, overall_score=score.overall_score, recommendation=score.recommendation,
            needs_human_review=score.needs_human_review, risk_level=score.risk_level,
            quality_flag=score.quality_flag, stage=app.stage, completed_at=sess.completed_at,
        ))

    total_apps = (await db.scalar(
        select(func.count()).select_from(Application).where(
            Application.workspace_id == ws, Application.job_position_id == job_id, Application.is_deleted == False)
    )) or 0

    return ShortlistResponse(
        job_id=job_id, job_title=job.title, items=items,
        not_interviewed=max(total_apps - len(items), 0),
    )
