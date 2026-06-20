"""Pre-screening / knockout API (Phase 11). Recruiter CRUD + recruiter override."""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.recruitment.models.interview import InterviewVersion
from app.recruitment.models.prescreen import PreScreenQuestion, PreScreenResult
from app.recruitment.models.candidate import Application, ApplicationHistory
from app.recruitment.schemas.prescreen import (
    PreScreenQuestionCreate, PreScreenQuestionUpdate, PreScreenQuestionResponse,
    PreScreenResultResponse, PreScreenOverrideRequest,
)
from app.recruitment.schemas.common import MessageResponse
from app.recruitment.api.deps import get_ctx, WorkspaceContext, ensure_draft

router = APIRouter(prefix="/recruitment", tags=["Recruitment - Pre-screening"])


async def _version_or_404(vid: UUID, ws_id: UUID, db: AsyncSession) -> InterviewVersion:
    v = (await db.execute(select(InterviewVersion).where(
        InterviewVersion.id == vid, InterviewVersion.workspace_id == ws_id))).scalar_one_or_none()
    if not v:
        raise HTTPException(status_code=404, detail="Interview version not found")
    return v


@router.get("/interviews/versions/{version_id}/prescreen", response_model=list[PreScreenQuestionResponse])
async def list_prescreen(version_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    await _version_or_404(version_id, ctx.id, db)
    rows = await db.execute(select(PreScreenQuestion).where(
        PreScreenQuestion.version_id == version_id, PreScreenQuestion.is_deleted == False).order_by(PreScreenQuestion.order_index))
    return list(rows.scalars().all())


@router.post("/interviews/versions/{version_id}/prescreen", response_model=PreScreenQuestionResponse, status_code=201)
async def add_prescreen(version_id: UUID, payload: PreScreenQuestionCreate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    version = await _version_or_404(version_id, ctx.id, db)
    ensure_draft(version)
    q = PreScreenQuestion(
        version_id=version_id, prompt=payload.prompt, qtype=payload.qtype,
        options=payload.options, knockout=payload.knockout or {}, required=payload.required,
        order_index=payload.order_index,
    )
    db.add(q)
    await db.commit()
    await db.refresh(q)
    return q


@router.patch("/prescreen-questions/{qid}", response_model=PreScreenQuestionResponse)
async def update_prescreen(qid: UUID, payload: PreScreenQuestionUpdate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    q = (await db.execute(select(PreScreenQuestion).where(PreScreenQuestion.id == qid))).scalar_one_or_none()
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    version = await _version_or_404(q.version_id, ctx.id, db)
    ensure_draft(version)
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(q, k, v)
    await db.commit()
    await db.refresh(q)
    return q


@router.delete("/prescreen-questions/{qid}", response_model=MessageResponse)
async def delete_prescreen(qid: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    q = (await db.execute(select(PreScreenQuestion).where(PreScreenQuestion.id == qid))).scalar_one_or_none()
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    version = await _version_or_404(q.version_id, ctx.id, db)
    ensure_draft(version)
    q.is_deleted = True
    await db.commit()
    return MessageResponse(message="Question removed")


@router.get("/applications/{application_id}/prescreen-result", response_model=PreScreenResultResponse)
async def get_result(application_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    r = (await db.execute(select(PreScreenResult).where(
        PreScreenResult.application_id == application_id, PreScreenResult.workspace_id == ctx.id
    ).order_by(PreScreenResult.created_at.desc()))).scalars().first()
    if not r:
        raise HTTPException(status_code=404, detail="No pre-screening result for this application")
    return r


@router.post("/prescreen-results/{result_id}/override", response_model=PreScreenResultResponse)
async def override_result(result_id: UUID, payload: PreScreenOverrideRequest, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Recruiter overrides an automatic ineligible/eligible decision."""
    ctx.require_edit()
    r = (await db.execute(select(PreScreenResult).where(
        PreScreenResult.id == result_id, PreScreenResult.workspace_id == ctx.id))).scalar_one_or_none()
    if not r:
        raise HTTPException(status_code=404, detail="Result not found")
    r.eligible = payload.eligible
    r.overridden = True
    r.override_note = payload.note
    # If making eligible again, pull the application back out of auto-rejection.
    if payload.eligible and r.application_id:
        app = (await db.execute(select(Application).where(Application.id == r.application_id))).scalar_one_or_none()
        if app and app.stage == "rejected":
            prev = app.stage
            app.stage = "screening"
            db.add(ApplicationHistory(workspace_id=ctx.id, application_id=app.id, from_stage=prev,
                                      to_stage="screening", actor_user_id=ctx.user.id,
                                      reason="Pre-screening override: marked eligible"))
    await db.commit()
    await db.refresh(r)
    return r
