"""Recruiter sessions + results API (Phase 2)."""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.recruitment.models.session import InterviewSession
from app.recruitment.models.score import InterviewScore
from app.recruitment.schemas.session import SessionResponse, SessionDetailResponse
from app.recruitment.schemas.score import ScoreResponse, ApplicationResultResponse
from app.recruitment.api.deps import get_ctx, WorkspaceContext
from app.recruitment.api.scoring_ops import run_and_store_score

router = APIRouter(prefix="/recruitment", tags=["Recruitment - Sessions & Results"])


async def _session_or_404(sid: UUID, ws_id: UUID, db: AsyncSession) -> InterviewSession:
    res = await db.execute(select(InterviewSession).options(selectinload(InterviewSession.answers)).where(
        InterviewSession.id == sid, InterviewSession.workspace_id == ws_id, InterviewSession.is_deleted == False))
    s = res.scalar_one_or_none()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s


@router.get("/sessions", response_model=list[SessionResponse])
async def list_sessions(
    version_id: Optional[UUID] = Query(None),
    application_id: Optional[UUID] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db),
):
    q = select(InterviewSession).where(InterviewSession.workspace_id == ctx.id, InterviewSession.is_deleted == False)
    if version_id:
        q = q.where(InterviewSession.version_id == version_id)
    if application_id:
        q = q.where(InterviewSession.application_id == application_id)
    if status_filter:
        q = q.where(InterviewSession.status == status_filter)
    rows = await db.execute(q.order_by(InterviewSession.created_at.desc()))
    return list(rows.scalars().all())


@router.get("/sessions/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    return await _session_or_404(session_id, ctx.id, db)


@router.get("/sessions/{session_id}/score", response_model=ScoreResponse)
async def get_session_score(session_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    await _session_or_404(session_id, ctx.id, db)
    res = await db.execute(select(InterviewScore).options(selectinload(InterviewScore.criterion_scores)).where(
        InterviewScore.session_id == session_id, InterviewScore.workspace_id == ctx.id))
    score = res.scalar_one_or_none()
    if not score:
        raise HTTPException(status_code=404, detail="No score yet for this session")
    return score


@router.post("/sessions/{session_id}/score", response_model=ScoreResponse)
async def rescore_session(session_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Re-run scoring for a completed session (e.g. after configuring an AI key)."""
    ctx.require_edit()
    session = await _session_or_404(session_id, ctx.id, db)
    if session.status not in ("completed", "abandoned"):
        raise HTTPException(status_code=409, detail="Session is not finished yet")
    score = await run_and_store_score(session, db)
    res = await db.execute(select(InterviewScore).options(selectinload(InterviewScore.criterion_scores)).where(
        InterviewScore.id == score.id))
    return res.scalar_one()


@router.get("/applications/{application_id}/result", response_model=ApplicationResultResponse)
async def application_result(application_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    sres = await db.execute(select(InterviewSession).where(
        InterviewSession.application_id == application_id, InterviewSession.workspace_id == ctx.id,
        InterviewSession.is_deleted == False).order_by(InterviewSession.created_at.desc()))
    session = sres.scalars().first()
    if not session:
        return ApplicationResultResponse(application_id=application_id, has_session=False)
    scres = await db.execute(select(InterviewScore).options(selectinload(InterviewScore.criterion_scores)).where(
        InterviewScore.session_id == session.id))
    score = scres.scalar_one_or_none()
    return ApplicationResultResponse(
        application_id=application_id, session_id=session.id, has_session=True,
        score=ScoreResponse.model_validate(score) if score else None,
    )
