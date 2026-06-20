"""
Public, candidate-facing API (Phase 2). NO login — addressed by opaque tokens.
Flow: view invite -> register (+consent) -> answer loop -> complete -> result.
"""
from datetime import datetime
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.recruitment.models.candidate import Candidate, Application, ApplicationHistory
from app.recruitment.models.interview import InterviewVersion, InterviewTemplate
from app.recruitment.models.question import InterviewQuestion, BranchRule
from app.recruitment.models.invite import InterviewInvite
from app.recruitment.models.session import InterviewSession, SessionAnswer
from app.recruitment.models.prescreen import PreScreenQuestion, PreScreenResult
from app.recruitment.models.document import CandidateDocument
from app.recruitment.services import storage as doc_storage
from app.recruitment.models.score import InterviewScore
from app.recruitment.schemas.public import (
    InvitePublicView, RegisterRequest, SessionStateResponse, PublicQuestion,
    AnswerSubmit, PublicResult, RiskSignalsUpdate,
)
from app.recruitment.services.tokens import new_token, default_consent_text, CONSENT_VERSION
from app.recruitment.services.session_flow import next_index_after, resolve_question_index
from app.recruitment.services.invite_access import evaluate_access, attempts_remaining, mask_email
from app.recruitment.services.prescreen import evaluate_prescreen
from app.recruitment.schemas.prescreen import (
    PreScreenPublicView, PreScreenPublicQuestion, PreScreenSubmit, PreScreenSubmitResult,
)
from app.recruitment.schemas.public import (
    PublicStatusView, StatusStep, PortalView, PortalRole, PortalApplyRequest, PortalApplyResult,
)
from app.recruitment.models.interview import InterviewTemplate as _Tpl
from app.recruitment.models.job import JobPosition as _Job
from app.models.user import Workspace as _Workspace
from app.recruitment.models.invite import InterviewInvite as _Invite
from app.recruitment.api.deps import get_or_create_settings
from app.recruitment.api.scoring_ops import run_and_store_score


async def _completed_count(invite_id, db: AsyncSession) -> int:
    return (await db.scalar(select(func.count()).select_from(InterviewSession).where(
        InterviewSession.invite_id == invite_id, InterviewSession.status == "completed",
        InterviewSession.is_deleted == False))) or 0


async def _in_progress_session(invite_id, candidate_email, db: AsyncSession):
    """Find a resumable in-progress session for this invite (to avoid duplicate attempts)."""
    rows = await db.execute(select(InterviewSession).where(
        InterviewSession.invite_id == invite_id, InterviewSession.status == "in_progress",
        InterviewSession.is_deleted == False).order_by(InterviewSession.created_at.desc()))
    return rows.scalars().first()

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
    cfg = q.config or {}
    return PublicQuestion(
        id=q.id, order_index=q.order_index, question_type=q.question_type, prompt_text=q.prompt_text,
        response_type=cfg.get("response_type", "text"),
        options=cfg.get("options", []) or [],
        scale=cfg.get("scale"),
    )


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
    done = await _completed_count(inv.id, db)
    return InvitePublicView(
        token=token, status=inv.status, interview_name=tpl.name,
        introduction=version.introduction, instructions=version.instructions,
        ai_identity_disclosure=version.ai_identity_disclosure, consent_text=consent,
        mode=version.mode, language=version.language,
        brand_name=settings.brand_name, brand_logo_url=settings.brand_logo_url,
        interviewer_avatar_url=settings.interviewer_avatar_url,
        expected_duration_minutes=version.expected_duration_minutes,
        already_completed=(inv.status == "completed"),
        attempts_remaining=attempts_remaining(inv.max_attempts, done),
        expires_at=inv.expires_at,
        email_locked=bool(inv.email),
    )


@router.post("/invites/{token}/register", response_model=SessionStateResponse, status_code=201)
async def register(token: str, payload: RegisterRequest, db: AsyncSession = Depends(get_db)):
    inv = await _invite_by_token(token, db)
    if not payload.consent_given:
        raise HTTPException(status_code=400, detail="Consent is required to begin")

    # Resume an existing in-progress attempt instead of starting a duplicate.
    existing = await _in_progress_session(inv.id, payload.email, db)
    if existing:
        return await _state(existing, db)

    # Enforce email lock, expiry/deadline and attempt limits.
    done = await _completed_count(inv.id, db)
    allowed, code, message = evaluate_access(
        status=inv.status, invite_email=inv.email, expires_at=inv.expires_at,
        max_attempts=inv.max_attempts, completed_count=done, candidate_email=payload.email,
    )
    if not allowed:
        status_code = 403 if code == "email_mismatch" else (410 if code in ("revoked", "expired") else 409)
        raise HTTPException(status_code=status_code, detail=message)

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

    # Map the answer to a real question by the submitted id (robust to retries / out-of-order).
    answered_idx = resolve_question_index(questions, payload.question_id, idx)
    current = questions[answered_idx]

    # Idempotency: update an existing answer for this (session, question, follow-up)
    # rather than inserting a duplicate, so retries don't create double answers.
    existing = (await db.execute(select(SessionAnswer).where(
        SessionAnswer.session_id == session.id,
        SessionAnswer.question_id == current.id,
        SessionAnswer.is_follow_up == payload.is_follow_up))).scalars().first()
    if existing:
        existing.transcript_text = payload.transcript_text
        existing.duration_seconds = payload.duration_seconds
        existing.question_text_snapshot = current.prompt_text
    else:
        count = await db.scalar(select(func.count()).select_from(
            select(SessionAnswer).where(SessionAnswer.session_id == session.id).subquery()))
        db.add(SessionAnswer(
            workspace_id=session.workspace_id, session_id=session.id, question_id=current.id,
            order_index=count or 0, question_text_snapshot=current.prompt_text,
            transcript_text=payload.transcript_text, duration_seconds=payload.duration_seconds,
            is_follow_up=payload.is_follow_up,
        ))
        # Append to the running transcript only for new answers (avoid duplicate lines on retry)
        transcript = list(session.transcript or [])
        transcript.append({"role": "interviewer", "text": current.prompt_text, "question_id": str(current.id)})
        transcript.append({"role": "candidate", "text": payload.transcript_text, "question_id": str(current.id)})
        session.transcript = transcript

    if payload.risk_signals:
        merged = dict(session.risk_signals or {})
        merged.update(payload.risk_signals)
        session.risk_signals = merged

    # A follow-up answer stays on the same question: record it but don't advance or finalize.
    if payload.is_follow_up:
        await db.commit()
        await db.refresh(session)
        return await _state(session, db)

    # Branch rules grouped by question
    brows = await db.execute(select(BranchRule).where(
        BranchRule.version_id == session.version_id).order_by(BranchRule.order_index))
    rules_by_q = {}
    for r in brows.scalars().all():
        rules_by_q.setdefault(str(r.question_id), []).append(r)

    nxt, ended, knockout = next_index_after(questions, answered_idx, payload.transcript_text, rules_by_q)
    if ended:
        if knockout:
            session.risk_signals = {**(session.risk_signals or {}), "knockout": True}
        await db.commit()
        await _finalize(session, db)
        return await _state(session, db)

    # Never move the pointer backwards (guards against duplicate/old posts).
    if nxt is not None:
        session.current_question_index = max(session.current_question_index, nxt)
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


@router.get("/invites/{token}/prescreen", response_model=PreScreenPublicView)
async def get_prescreen(token: str, db: AsyncSession = Depends(get_db)):
    inv = await _invite_by_token(token, db)
    rows = await db.execute(select(PreScreenQuestion).where(
        PreScreenQuestion.version_id == inv.version_id, PreScreenQuestion.is_deleted == False).order_by(PreScreenQuestion.order_index))
    qs = [PreScreenPublicQuestion(id=q.id, prompt=q.prompt, qtype=q.qtype, options=q.options or [], required=q.required)
          for q in rows.scalars().all()]
    return PreScreenPublicView(questions=qs)


@router.post("/sessions/{session_token}/prescreen", response_model=PreScreenSubmitResult)
async def submit_prescreen(session_token: str, payload: PreScreenSubmit, db: AsyncSession = Depends(get_db)):
    session = await _session_by_token(session_token, db)
    rows = await db.execute(select(PreScreenQuestion).where(
        PreScreenQuestion.version_id == session.version_id, PreScreenQuestion.is_deleted == False).order_by(PreScreenQuestion.order_index))
    questions = list(rows.scalars().all())
    if not questions:
        return PreScreenSubmitResult(eligible=True, message="No pre-screening required.")

    qdicts = [{"id": str(q.id), "prompt": q.prompt, "qtype": q.qtype,
               "knockout": q.knockout or {}, "required": q.required} for q in questions]
    answers = {str(a.question_id): a.value for a in payload.answers}
    eligible, _failed = evaluate_prescreen(qdicts, answers)

    prompts = {str(q.id): q.prompt for q in questions}
    stored = [{"question_id": qid, "prompt": prompts.get(qid, ""), "value": answers.get(qid)} for qid in prompts]

    # Avoid duplicate result rows on resubmit.
    existing = (await db.execute(select(PreScreenResult).where(
        PreScreenResult.session_id == session.id))).scalars().first()
    if existing:
        existing.answers = stored
        existing.auto_eligible = eligible
        if not existing.overridden:
            existing.eligible = eligible
        result = existing
    else:
        result = PreScreenResult(
            workspace_id=session.workspace_id, version_id=session.version_id,
            invite_id=session.invite_id, session_id=session.id, application_id=session.application_id,
            candidate_email=None, answers=stored, auto_eligible=eligible, eligible=eligible,
        )
        db.add(result)

    if not eligible:
        # Block the interview and auto-move the application out (recruiter can override).
        session.status = "ineligible"
        if session.application_id:
            app = (await db.execute(select(Application).where(Application.id == session.application_id))).scalar_one_or_none()
            if app and app.stage in ("applied", "screening", "interview"):
                prev = app.stage
                app.stage = "rejected"
                db.add(ApplicationHistory(workspace_id=session.workspace_id, application_id=app.id,
                                          from_stage=prev, to_stage="rejected", reason="Pre-screening: not eligible"))
        await db.commit()
        return PreScreenSubmitResult(eligible=False, message="Thank you. Based on your responses, this role isn't a match right now.")

    await db.commit()
    return PreScreenSubmitResult(eligible=True, message="You're all set. Let's begin.")


@router.post("/sessions/{session_token}/documents")
async def upload_candidate_document(session_token: str, file: UploadFile = File(...), kind: str = Form("resume"),
                                    db: AsyncSession = Depends(get_db)):
    """Candidate uploads a resume / cover letter during their interview flow."""
    session = await _session_by_token(session_token, db)
    data = await file.read()
    try:
        doc_storage.validate(file.filename, file.content_type, len(data))
    except doc_storage.StorageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not session.candidate_id:
        raise HTTPException(status_code=409, detail="This session has no candidate to attach documents to")
    path = doc_storage.save_bytes(data, file.filename)
    doc = CandidateDocument(
        workspace_id=session.workspace_id, candidate_id=session.candidate_id,
        application_id=session.application_id, kind=kind, filename=file.filename,
        content_type=file.content_type, size=len(data), storage_path=path, source="candidate",
    )
    db.add(doc)
    await db.commit()
    return {"ok": True, "filename": file.filename, "kind": kind}


@router.get("/invites/{token}/status", response_model=PublicStatusView)
async def candidate_status(token: str, db: AsyncSession = Depends(get_db)):
    """A privacy-safe status page for the candidate (no scores). Reachable with their invite link."""
    inv = await _invite_by_token(token, db)
    version = (await db.execute(select(InterviewVersion).where(InterviewVersion.id == inv.version_id))).scalar_one_or_none()
    tpl = None
    if version:
        tpl = (await db.execute(select(InterviewTemplate).where(InterviewTemplate.id == version.template_id))).scalar_one_or_none()
    settings = await get_or_create_settings(inv.workspace_id, db)

    cand_name = inv.candidate_name
    if not cand_name and inv.candidate_id:
        cand = (await db.execute(select(Candidate).where(Candidate.id == inv.candidate_id))).scalar_one_or_none()
        cand_name = cand.full_name if cand else None

    done = await _completed_count(inv.id, db)
    in_prog = await _in_progress_session(inv.id, None, db)
    remaining = attempts_remaining(inv.max_attempts, done)

    stage = None
    if inv.application_id:
        app = (await db.execute(select(Application).where(Application.id == inv.application_id))).scalar_one_or_none()
        stage = app.stage if app else None

    # Any ineligible session means knocked out on pre-screening.
    ineligible = (await db.scalar(select(func.count()).select_from(InterviewSession).where(
        InterviewSession.invite_id == inv.id, InterviewSession.status == "ineligible"))) or 0

    from datetime import datetime as _dt
    now = _dt.utcnow()
    if inv.status == "revoked":
        status, headline, msg = "closed", "This interview is closed", "This invitation is no longer active."
    elif inv.expires_at and now > inv.expires_at and done == 0:
        status, headline, msg = "expired", "This invitation has expired", "The deadline to complete this interview has passed."
    elif ineligible and done == 0:
        status, headline, msg = "ineligible", "Thanks for your interest", "Based on your pre-screening responses, this role isn't a match right now."
    elif in_prog:
        status, headline, msg = "in_progress", "Your interview is in progress", "You can pick up where you left off."
    elif done == 0:
        status, headline, msg = "not_started", "You haven't started yet", "When you're ready, you can begin your interview."
    elif stage == "advanced":
        status, headline, msg = "advanced", "Good news, you're moving forward", "Your application is progressing to the next stage. The team will be in touch."
    elif stage == "rejected":
        status, headline, msg = "not_selected", "Update on your application", "Thank you for interviewing. We won't be moving forward on this occasion, and we wish you well."
    else:
        status, headline, msg = "under_review", "Your interview is being reviewed", "Thanks for completing your interview. The team is reviewing it and will be in touch."

    def step(label, st):
        return StatusStep(label=label, state=st)

    interview_state = "done" if done > 0 else ("current" if in_prog or status == "not_started" else "upcoming")
    decision_state = "done" if status in ("advanced", "not_selected") else ("current" if status == "under_review" else "upcoming")
    steps = [
        step("Invited", "done"),
        step("Interview", interview_state),
        step("Decision", decision_state),
    ]
    can_resume = bool(in_prog) or (status == "not_started" and remaining > 0)

    return PublicStatusView(
        candidate_name=cand_name, interview_name=(tpl.name if tpl else "Interview"),
        brand_name=settings.brand_name, status=status, headline=headline, message=msg,
        attempts_remaining=remaining, expires_at=inv.expires_at, can_resume=can_resume, steps=steps,
    )


async def _workspace_by_slug(slug: str, db: AsyncSession) -> _Workspace:
    ws = (await db.execute(select(_Workspace).where(_Workspace.slug == slug))).scalar_one_or_none()
    if not ws:
        raise HTTPException(status_code=404, detail="Careers page not found")
    return ws


@router.get("/portal/{slug}", response_model=PortalView)
async def portal(slug: str, db: AsyncSession = Depends(get_db)):
    """Public careers page: published interviews a candidate can self-apply to."""
    ws = await _workspace_by_slug(slug, db)
    settings = await get_or_create_settings(ws.id, db)
    rows = await db.execute(select(_Tpl).where(
        _Tpl.workspace_id == ws.id, _Tpl.archived == False, _Tpl.latest_published_version_id.isnot(None)))
    roles = []
    for tpl in rows.scalars().all():
        version = (await db.execute(select(InterviewVersion).where(
            InterviewVersion.id == tpl.latest_published_version_id))).scalar_one_or_none()
        if not version:
            continue
        job = None
        if tpl.job_position_id:
            job = (await db.execute(select(_Job).where(_Job.id == tpl.job_position_id))).scalar_one_or_none()
            if job and job.status not in ("open",):
                continue  # only surface roles whose job is open
        roles.append(PortalRole(
            template_id=tpl.id, name=tpl.name, description=tpl.description,
            job_title=(job.title if job else None), location=(job.location if job else None),
            department=(job.department if job else None),
            mode=version.mode, expected_duration_minutes=version.expected_duration_minutes,
        ))
    return PortalView(
        workspace_name=ws.name, brand_name=settings.brand_name or ws.name,
        brand_logo_url=settings.brand_logo_url or ws.logo_url, roles=roles,
    )


@router.post("/portal/{slug}/apply/{template_id}", response_model=PortalApplyResult, status_code=201)
async def portal_apply(slug: str, template_id: UUID, payload: PortalApplyRequest, db: AsyncSession = Depends(get_db)):
    """Candidate self-applies; creates a candidate, application and a personal invite link."""
    ws = await _workspace_by_slug(slug, db)
    tpl = (await db.execute(select(_Tpl).where(
        _Tpl.id == template_id, _Tpl.workspace_id == ws.id, _Tpl.archived == False))).scalar_one_or_none()
    if not tpl or not tpl.latest_published_version_id:
        raise HTTPException(status_code=404, detail="This role is no longer accepting applications")

    email = payload.email.strip().lower()
    cand = (await db.execute(select(Candidate).where(
        Candidate.workspace_id == ws.id, func.lower(Candidate.email) == email, Candidate.is_deleted == False))).scalar_one_or_none()
    if not cand:
        cand = Candidate(workspace_id=ws.id, full_name=payload.full_name, email=payload.email, source="portal")
        db.add(cand)
        await db.flush()

    application = None
    if tpl.job_position_id:
        application = (await db.execute(select(Application).where(
            Application.workspace_id == ws.id, Application.candidate_id == cand.id,
            Application.job_position_id == tpl.job_position_id, Application.is_deleted == False))).scalar_one_or_none()
        if not application:
            application = Application(workspace_id=ws.id, candidate_id=cand.id,
                                      job_position_id=tpl.job_position_id, stage="applied", status="active")
            db.add(application)
            await db.flush()

    token = new_token()
    invite = _Invite(
        workspace_id=ws.id, version_id=tpl.latest_published_version_id, token=token,
        email=payload.email, candidate_name=payload.full_name, status="pending", max_attempts=1,
        candidate_id=cand.id, application_id=(application.id if application else None),
    )
    db.add(invite)
    await db.commit()
    return PortalApplyResult(token=token)
