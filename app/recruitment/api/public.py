"""
Public, candidate-facing API (Phase 2). NO login — addressed by opaque tokens.
Flow: view invite -> register (+consent) -> answer loop -> complete -> result.
"""
from datetime import datetime
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.recruitment.models.candidate import Candidate, Application, ApplicationHistory
from app.recruitment.models.interview import InterviewVersion, InterviewTemplate
from app.recruitment.models.question import InterviewQuestion, BranchRule
from app.recruitment.models.invite import InterviewInvite
from app.recruitment.models.session import InterviewSession, SessionAnswer
from app.recruitment.models.score import InterviewScore
from app.recruitment.schemas.public import (
    InvitePublicView, RegisterRequest, SessionStateResponse, PublicQuestion,
    AnswerSubmit, PublicResult, RiskSignalsUpdate,
)
from app.recruitment.services.tokens import new_token, default_consent_text, CONSENT_VERSION
from app.recruitment.services.session_flow import next_index_after
from app.recruitment.api.deps import get_or_create_settings
from app.recruitment.api.scoring_ops import run_and_store_score

router = APIRouter(prefix="/recruitment/public", tags=["Recruitment - Public (Candidate)"])


async def _invite_by_token(token: str, db: AsyncSession) -> InterviewInvite:
    inv = (await db.execute(select(InterviewInvite).where(InterviewInvite.token == token))).scalar_one_or_none()
    if not inv:
        raise HTTPException(status_code=404, detail="Invite not found")
    if inv.status == "revoked":
        raise HTTPException(status_code=410, detail="This invite has been revoked")
    if inv.expires_at and inv.expires_at < datetime.utcnow():
        if inv.status != "expired":
            inv.status = "expired"; await db.commit()
        raise HTTPException(status_code=410, detail="This invite has expired")
    return inv


async def _session_by_token(token: str, db: AsyncSession) -> InterviewSession:
    s = (await db.execute(select(InterviewSession).where(InterviewSession.session_token == token))).scalar_one_or_none()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s


async def _ordered_questions(version_id, db):
    rows = await db.execute(select(InterviewQuestion).where(
        InterviewQuestion.version_id == version_id).order_by(InterviewQuestion.order_index))
    return list(rows.scalars().all())


def _public_q(q) -> PublicQuestion:
    return PublicQuestion(id=q.id, order_index=q.order_index, question_type=q.question_type, prompt_text=q.prompt_text)


async def _state(session, db) -> SessionStateResponse:
    questions = await _ordered_questions(session.version_id, db)
    idx = session.current_question_index
    finished = session.status in ("completed", "abandoned") or idx >= len(questions)
    cur = None if finished else _public_q(questions[idx])
    return SessionStateResponse(
        session_token=session.session_token, status=session.status,
        current_question_index=idx, total_questions=len(questions),
        current_question=cur, ai_identity_disclosed=None, finished=finished,
    )


@router.get("/invites/{token}", response_model=InvitePublicView)
async def view_invite(token: str, db: AsyncSession = Depends(get_db)):
    inv = await _invite_by_token(token, db)
    version = (await db.execute(select(InterviewVersion).where(InterviewVersion.id == inv.version_id))).scalar_one_or_none()
    tpl = (await db.execute(select(InterviewTemplate).where(InterviewTemplate.id == inv.template_id))).scalar_one_or_none()
    if not version or not tpl:
        raise HTTPException(status_code=404, detail="Interview unavailable")
    settings = await get_or_create_settings(inv.workspace_id, db)
    consent = settings.consent_text or default_consent_text(settings.jurisdiction)
    return InvitePublicView(
        token=token, status=inv.status, interview_name=tpl.name,
        introduction=version.introduction, instructions=version.instructions,
        ai_identity_disclosure=version.ai_identity_disclosure, consent_text=consent,
        mode=version.mode, language=version.language,
        brand_name=settings.brand_name, brand_logo_url=settings.brand_logo_url,
        expected_duration_minutes=version.expected_duration_minutes,
        already_completed=(inv.status == "completed"),
    )


@router.post("/invites/{token}/register", response_model=SessionStateResponse, status_code=201)
async def register(token: str, payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    inv = await _invite_by_token(token, db)
    if inv.status == "completed":
        raise HTTPException(status_code=409, detail="This interview has already been completed")
    if not payload.consent_given:
        raise HTTPException(status_code=400, detail="Consent is required to begin")

    version = (await db.execute(select(InterviewVersion).where(InterviewVersion.id == inv.version_id))).scalar_one_or_none()
    if not version or version.status != "published":
        raise HTTPException(status_code=400, detail="Interview unavailable")
    settings = await get_or_create_settings(inv.workspace_id, db)
    consent_snapshot = settings.consent_text or default_consent_text(settings.jurisdiction)

    # Match or create candidate by email within the workspace
    cand = (await db.execute(select(Candidate).where(
        Candidate.workspace_id == inv.workspace_id,
        func.lower(Candidate.email) == payload.email.lower(),
        Candidate.is_deleted == False))).scalar_one_or_none()
    if cand:
        cand.full_name = payload.full_name or cand.full_name
        if payload.phone:
            cand.phone = payload.phone
    else:
        cand = Candidate(
            workspace_id=inv.workspace_id, full_name=payload.full_name, email=payload.email,
            phone=payload.phone, language=payload.language or version.language, source="self_registration",
        )
        db.add(cand)
    cand.consent_given = True
    cand.consent_version = CONSENT_VERSION
    cand.consent_at = datetime.utcnow()
    await db.flush()

    application = Application(
        workspace_id=inv.workspace_id, candidate_id=cand.id, job_position_id=inv.job_position_id,
        stage="interview", status="active",
    )
    db.add(application)
    await db.flush()
    db.add(ApplicationHistory(
        workspace_id=inv.workspace_id, application_id=application.id,
        from_stage=None, to_stage="interview", reason="Candidate self-registered via invite",
    ))

    session = InterviewSession(
        workspace_id=inv.workspace_id, invite_id=inv.id, version_id=version.id,
        candidate_id=cand.id, application_id=application.id, session_token=new_token(),
        status="in_progress", language=payload.language or version.language,
        consent_given=True, consent_version=CONSENT_VERSION, consent_text_snapshot=consent_snapshot,
        consent_at=datetime.utcnow(), ai_identity_disclosed=True,
        recording_enabled=bool(version.recording_enabled), current_question_index=0,
        started_at=datetime.utcnow(),
        transcript=[
            {"role": "interviewer", "text": version.ai_identity_disclosure, "question_id": None},
            *([{"role": "interviewer", "text": version.introduction, "question_id": None}] if version.introduction else []),
        ],
        risk_signals={},
    )
    db.add(session)

    inv.candidate_id = cand.id
    inv.application_id = application.id
    inv.status = "in_progress"
    await db.commit()
    await db.refresh(session)
    return await _state(session, db)


@router.get("/sessions/{session_token}", response_model=SessionStateResponse)
async def session_state(session_token: str, db: AsyncSession = Depends(get_db)):
    session = await _session_by_token(session_token, db)
    return await _state(session, db)


@router.post("/sessions/{session_token}/answers", response_model=SessionStateResponse)
async def submit_answer(session_token: str, payload: AnswerSubmit, db: AsyncSession = Depends(get_db)):
    session = await _session_by_token(session_token, db)
    if session.status != "in_progress":
        return await _state(session, db)

    questions = await _ordered_questions(session.version_id, db)
    idx = session.current_question_index
    if idx >= len(questions):
        await _finalize(session, db)
        return await _state(session, db)
    current = questions[idx]

    # Record the answer
    count = await db.scalar(select(func.count()).select_from(
        select(SessionAnswer).where(SessionAnswer.session_id == session.id).subquery()))
    db.add(SessionAnswer(
        workspace_id=session.workspace_id, session_id=session.id, question_id=current.id,
        order_index=count or 0, question_text_snapshot=current.prompt_text,
        transcript_text=payload.transcript_text, duration_seconds=payload.duration_seconds,
        is_follow_up=payload.is_follow_up,
    ))
    # Append to the running transcript
    transcript = list(session.transcript or [])
    transcript.append({"role": "interviewer", "text": current.prompt_text, "question_id": str(current.id)})
    transcript.append({"role": "candidate", "text": payload.transcript_text, "question_id": str(current.id)})
    session.transcript = transcript
    if payload.risk_signals:
        merged = dict(session.risk_signals or {})
        merged.update(payload.risk_signals)
        session.risk_signals = merged

    # Branch rules grouped by question
    brows = await db.execute(select(BranchRule).where(
        BranchRule.version_id == session.version_id).order_by(BranchRule.order_index))
    rules_by_q = {}
    for r in brows.scalars().all():
        rules_by_q.setdefault(str(r.question_id), []).append(r)

    nxt, ended, knockout = next_index_after(questions, idx, payload.transcript_text, rules_by_q)
    if ended:
        if knockout:
            session.risk_signals = {**(session.risk_signals or {}), "knockout": True}
        await db.commit()
        await _finalize(session, db)
        return await _state(session, db)

    session.current_question_index = nxt
    await db.commit()
    await db.refresh(session)
    return await _state(session, db)


async def _finalize(session: InterviewSession, db: AsyncSession):
    """Mark the session complete, advance the application, and score it."""
    session.status = "completed"
    session.completed_at = datetime.utcnow()
    # Advance the application into review and log it
    if session.application_id:
        app = (await db.execute(select(Application).where(Application.id == session.application_id))).scalar_one_or_none()
        if app and app.stage in ("interview", "applied", "screening"):
            prev = app.stage
            app.stage = "review"
            db.add(ApplicationHistory(
                workspace_id=session.workspace_id, application_id=app.id,
                from_stage=prev, to_stage="review", reason="Interview completed",
            ))
    if session.invite_id:
        inv = (await db.execute(select(InterviewInvite).where(InterviewInvite.id == session.invite_id))).scalar_one_or_none()
        if inv:
            inv.status = "completed"
    await db.commit()
    # Score (commits internally); never let scoring errors break completion
    try:
        await run_and_store_score(session, db)
    except Exception:
        pass


@router.post("/sessions/{session_token}/complete", response_model=SessionStateResponse)
async def complete(session_token: str, db: AsyncSession = Depends(get_db)):
    session = await _session_by_token(session_token, db)
    if session.status == "in_progress":
        await _finalize(session, db)
    return await _state(session, db)


@router.get("/sessions/{session_token}/result", response_model=PublicResult)
async def result(session_token: str, db: AsyncSession = Depends(get_db)):
    session = await _session_by_token(session_token, db)
    if session.status != "completed":
        return PublicResult(status=session.status, message="Your interview is in progress.")
    settings = await get_or_create_settings(session.workspace_id, db)
    if not settings.candidates_see_scores:
        return PublicResult(status="completed", message="Thank you. Your interview has been submitted and the hiring team will be in touch.")
    score = (await db.execute(select(InterviewScore).where(InterviewScore.session_id == session.id))).scalar_one_or_none()
    if not score or score.status != "completed":
        return PublicResult(status="completed", message="Thank you. Your responses are being processed.")
    return PublicResult(
        status="completed", message="Thank you for completing your interview.",
        overall_score=score.overall_score, recommendation=score.recommendation, summary=score.summary,
    )



@router.post("/sessions/{session_token}/risk-signals")
async def update_risk_signals(session_token: str, payload: RiskSignalsUpdate, db: AsyncSession = Depends(get_db)):
    """Merge client-side integrity signals (tab switches, pastes, focus loss) into the session."""
    session = await _session_by_token(session_token, db)
    merged = dict(session.risk_signals or {})
    for k, v in (payload.signals or {}).items():
        # counts accumulate to the max reported (client sends running totals)
        try:
            merged[k] = max(int(merged.get(k, 0) or 0), int(v or 0))
        except (TypeError, ValueError):
            merged[k] = v
    session.risk_signals = merged
    await db.commit()
    return {"ok": True}


# ---------------- Voice mode (LiveKit) ----------------
# These reuse the existing LiveKit token pattern from app/api/routes/video.py.
# They are token-gated by the session_token (no login), so a candidate can join
# the interview room and the interview worker can fetch the script to conduct it.

from app.api.routes.video import generate_livekit_token, LIVEKIT_URL  # noqa: E402
from app.recruitment.schemas.public import PublicQuestion  # noqa: E402


@router.post("/sessions/{session_token}/voice-token")
async def voice_token(session_token: str, db: AsyncSession = Depends(get_db)):
    """Mint a LiveKit access token so the candidate can join the interview room."""
    session = await _session_by_token(session_token, db)
    if session.status not in ("in_progress", "created", "consented"):
        raise HTTPException(status_code=409, detail="This interview is not open for a voice session")
    room_name = f"interview-{session_token}"
    token = generate_livekit_token(room_name=room_name, participant_name="candidate", is_agent=False)
    return {"token": token, "room_name": room_name, "livekit_url": LIVEKIT_URL}


@router.get("/sessions/{session_token}/agent-config")
async def agent_config(session_token: str, db: AsyncSession = Depends(get_db)):
    """Interview script for the voice worker to conduct the session."""
    session = await _session_by_token(session_token, db)
    version = (await db.execute(select(InterviewVersion).where(InterviewVersion.id == session.version_id))).scalar_one_or_none()
    if not version:
        raise HTTPException(status_code=404, detail="Interview unavailable")
    questions = await _ordered_questions(session.version_id, db)
    return {
        "session_token": session_token,
        "status": session.status,
        "language": session.language,
        "ai_identity_disclosure": version.ai_identity_disclosure,
        "introduction": version.introduction,
        "instructions": version.instructions,
        "questions": [
            {"id": str(q.id), "order_index": q.order_index, "question_type": q.question_type, "prompt_text": q.prompt_text}
            for q in questions
        ],
    }
