"""Recruiter sessions + results API (Phase 2)."""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from io import BytesIO
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.recruitment.models.session import InterviewSession
from app.recruitment.models.score import InterviewScore
from app.recruitment.models.score_review import ScoreReview
from app.recruitment.models.candidate import Candidate, Application
from app.recruitment.models.job import JobPosition
from app.recruitment.models.interview import InterviewVersion, InterviewTemplate
from app.recruitment.schemas.session import SessionResponse, SessionDetailResponse
from app.recruitment.schemas.score import ScoreResponse, ApplicationResultResponse, ScoreReviewCreate, ScoreReviewResponse, SpeechAnalytics
from app.recruitment.services.report import build_score_report
from app.recruitment.services.speech_analytics import analyze as analyze_speech
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


@router.get("/sessions/{session_id}/report")
async def session_report(session_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Download a PDF score report for a session."""
    session = await _session_or_404(session_id, ctx.id, db)

    score = (await db.execute(select(InterviewScore).options(selectinload(InterviewScore.criterion_scores)).where(
        InterviewScore.session_id == session_id, InterviewScore.workspace_id == ctx.id))).scalar_one_or_none()

    candidate = None
    if session.candidate_id:
        candidate = (await db.execute(select(Candidate).where(Candidate.id == session.candidate_id))).scalar_one_or_none()
    job_title = None
    if session.application_id:
        app = (await db.execute(select(Application).where(Application.id == session.application_id))).scalar_one_or_none()
        if app and app.job_position_id:
            job = (await db.execute(select(JobPosition).where(JobPosition.id == app.job_position_id))).scalar_one_or_none()
            job_title = job.title if job else None
    interview_name = None
    version = (await db.execute(select(InterviewVersion).where(InterviewVersion.id == session.version_id))).scalar_one_or_none()
    if version:
        tpl = (await db.execute(select(InterviewTemplate).where(InterviewTemplate.id == version.template_id))).scalar_one_or_none()
        interview_name = tpl.name if tpl else None

    data = {
        "candidate_name": candidate.full_name if candidate else "Candidate",
        "candidate_email": candidate.email if candidate else None,
        "job_title": job_title,
        "interview_name": interview_name,
        "status": session.status,
        "overall_score": score.overall_score if score else None,
        "recommendation": score.recommendation if score else None,
        "summary": score.summary if score else None,
        "needs_human_review": score.needs_human_review if score else False,
        "risk_level": score.risk_level if score else None,
        "risk_signals": score.risk_signals if score else {},
        "criteria": [
            {"name": cs.criterion_name, "weight": cs.weight, "raw_score": cs.raw_score,
             "evidence": cs.evidence, "reasoning": cs.reasoning}
            for cs in (score.criterion_scores if score else [])
        ],
        "transcript": [
            {"question": a.question_text_snapshot, "answer": a.transcript_text}
            for a in session.answers
        ],
        "delivery": analyze_speech([{"transcript_text": a.transcript_text, "duration_seconds": a.duration_seconds} for a in session.answers]),
    }
    pdf = build_score_report(data)
    name = (candidate.full_name if candidate else "candidate").replace(" ", "_")
    return StreamingResponse(
        BytesIO(pdf), media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="interview_report_{name}.pdf"'},
    )


@router.post("/sessions/{session_id}/review", response_model=ScoreReviewResponse, status_code=201)
async def add_review(session_id: UUID, payload: ScoreReviewCreate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Record a human review / score override. Each call is a new audit-trail entry."""
    ctx.require_edit()
    session = await _session_or_404(session_id, ctx.id, db)
    score = (await db.execute(select(InterviewScore).where(
        InterviewScore.session_id == session_id, InterviewScore.workspace_id == ctx.id))).scalar_one_or_none()
    review = ScoreReview(
        workspace_id=ctx.id, score_id=(score.id if score else None), session_id=session_id,
        application_id=session.application_id, reviewer_user_id=ctx.user.id,
        override_recommendation=payload.override_recommendation, override_score=payload.override_score,
        note=payload.note,
    )
    db.add(review)
    await db.commit()
    await db.refresh(review)
    return review


@router.get("/sessions/{session_id}/reviews", response_model=list[ScoreReviewResponse])
async def list_reviews(session_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    await _session_or_404(session_id, ctx.id, db)
    rows = await db.execute(select(ScoreReview).where(
        ScoreReview.session_id == session_id, ScoreReview.workspace_id == ctx.id).order_by(ScoreReview.created_at.desc()))
    return list(rows.scalars().all())


@router.get("/sessions/{session_id}/speech-analytics", response_model=SpeechAnalytics)
async def speech_analytics(session_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Advisory delivery (cadence) and sentiment metrics derived from the answers."""
    session = await _session_or_404(session_id, ctx.id, db)
    answers = [{"transcript_text": a.transcript_text, "duration_seconds": a.duration_seconds} for a in session.answers]
    return SpeechAnalytics(**analyze_speech(answers))
