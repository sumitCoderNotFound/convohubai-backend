"""Feature 4 - Questions + branching, scoped to a draft interview version."""
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.recruitment.models.interview import InterviewVersion
from app.recruitment.models.question import InterviewQuestion, BranchRule
from app.recruitment.schemas.question import (
    QuestionCreate, QuestionUpdate, QuestionReorder, QuestionResponse,
    BranchRuleCreate, BranchRuleResponse,
)
from app.recruitment.schemas.common import MessageResponse
from app.recruitment.api.deps import get_ctx, WorkspaceContext, ensure_draft

router = APIRouter(prefix="/recruitment/versions", tags=["Recruitment - Questions"])


async def _version_or_404(vid: UUID, ws_id: UUID, db: AsyncSession) -> InterviewVersion:
    res = await db.execute(select(InterviewVersion).where(
        InterviewVersion.id == vid, InterviewVersion.workspace_id == ws_id, InterviewVersion.is_deleted == False))
    v = res.scalar_one_or_none()
    if not v:
        raise HTTPException(status_code=404, detail="Interview version not found")
    return v


async def _question_or_404(qid: UUID, vid: UUID, db: AsyncSession) -> InterviewQuestion:
    res = await db.execute(select(InterviewQuestion).where(
        InterviewQuestion.id == qid, InterviewQuestion.version_id == vid))
    q = res.scalar_one_or_none()
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    return q


@router.get("/{version_id}/questions", response_model=List[QuestionResponse])
async def list_questions(version_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    await _version_or_404(version_id, ctx.id, db)
    rows = await db.execute(select(InterviewQuestion).where(
        InterviewQuestion.version_id == version_id).order_by(InterviewQuestion.order_index))
    return list(rows.scalars().all())


@router.post("/{version_id}/questions", response_model=QuestionResponse, status_code=201)
async def add_question(version_id: UUID, payload: QuestionCreate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    version = await _version_or_404(version_id, ctx.id, db)
    ensure_draft(version)
    if payload.order_index is None:
        count = await db.scalar(select(func.count()).select_from(
            select(InterviewQuestion).where(InterviewQuestion.version_id == version_id).subquery()))
        order_index = count or 0
    else:
        order_index = payload.order_index
    q = InterviewQuestion(
        workspace_id=ctx.id, version_id=version_id, order_index=order_index,
        question_type=payload.question_type.value, prompt_text=payload.prompt_text,
        config=payload.config.model_dump() if payload.config else {"required": True},
        is_knockout=payload.is_knockout, parent_question_id=payload.parent_question_id)
    db.add(q)
    await db.commit()
    await db.refresh(q)
    return q


@router.patch("/{version_id}/questions/{question_id}", response_model=QuestionResponse)
async def update_question(version_id: UUID, question_id: UUID, payload: QuestionUpdate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    version = await _version_or_404(version_id, ctx.id, db)
    ensure_draft(version)
    q = await _question_or_404(question_id, version_id, db)
    data = payload.model_dump(exclude_unset=True)
    if "question_type" in data and data["question_type"] is not None:
        data["question_type"] = data["question_type"].value
    if "config" in data and data["config"] is not None:
        q.config = data.pop("config")
    for k, v in data.items():
        setattr(q, k, v)
    await db.commit()
    await db.refresh(q)
    return q


@router.post("/{version_id}/questions/reorder", response_model=MessageResponse)
async def reorder_questions(version_id: UUID, payload: QuestionReorder, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    version = await _version_or_404(version_id, ctx.id, db)
    ensure_draft(version)
    for idx, qid in enumerate(payload.ordered_ids):
        q = await _question_or_404(qid, version_id, db)
        q.order_index = idx
    await db.commit()
    return MessageResponse(message="Reordered")


@router.delete("/{version_id}/questions/{question_id}", response_model=MessageResponse)
async def delete_question(version_id: UUID, question_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    version = await _version_or_404(version_id, ctx.id, db)
    ensure_draft(version)
    q = await _question_or_404(question_id, version_id, db)
    await db.delete(q)
    await db.commit()
    return MessageResponse(message="Question deleted")


@router.post("/{version_id}/questions/{question_id}/branch-rules", response_model=BranchRuleResponse, status_code=201)
async def add_branch_rule(version_id: UUID, question_id: UUID, payload: BranchRuleCreate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    version = await _version_or_404(version_id, ctx.id, db)
    ensure_draft(version)
    await _question_or_404(question_id, version_id, db)
    order_index = payload.order_index
    if order_index is None:
        count = await db.scalar(select(func.count()).select_from(
            select(BranchRule).where(BranchRule.question_id == question_id).subquery()))
        order_index = count or 0
    rule = BranchRule(
        workspace_id=ctx.id, version_id=version_id, question_id=question_id,
        order_index=order_index, condition=payload.condition, action=payload.action)
    db.add(rule)
    await db.commit()
    await db.refresh(rule)
    return rule
