"""Shared op: run scoring for a completed session and persist the result."""
from datetime import datetime
from sqlalchemy import select, delete
from sqlalchemy.orm import selectinload

from app.recruitment.models.rubric import RubricCriterion
from app.recruitment.models.session import InterviewSession, SessionAnswer
from app.recruitment.models.score import InterviewScore, CriterionScore
from app.recruitment.services.scoring import score_session


async def run_and_store_score(session: InterviewSession, db) -> InterviewScore:
    # Criteria (with anchors) for the session's version
    crows = await db.execute(
        select(RubricCriterion).options(selectinload(RubricCriterion.anchors))
        .where(RubricCriterion.version_id == session.version_id)
        .order_by(RubricCriterion.order_index)
    )
    criteria = list(crows.scalars().all())

    arows = await db.execute(
        select(SessionAnswer).where(SessionAnswer.session_id == session.id).order_by(SessionAnswer.order_index)
    )
    answers = list(arows.scalars().all())

    result = await score_session(session, criteria, answers)

    # Upsert one score per session
    existing = (await db.execute(select(InterviewScore).where(InterviewScore.session_id == session.id))).scalar_one_or_none()
    if existing:
        await db.execute(delete(CriterionScore).where(CriterionScore.score_id == existing.id))
        score = existing
    else:
        score = InterviewScore(workspace_id=session.workspace_id, session_id=session.id)
        db.add(score)

    score.application_id = session.application_id
    score.version_id = session.version_id
    score.status = result["status"]
    score.overall_score = result["overall_score"]
    score.recommendation = result["recommendation"]
    score.summary = result["summary"]
    score.quality_flag = result["quality_flag"]
    score.needs_human_review = result["needs_human_review"]
    score.risk_level = result["risk_level"]
    score.risk_signals = result["risk_signals"]
    score.model_used = result["model_used"]
    score.error = result["error"]
    score.scored_at = datetime.utcnow()
    await db.flush()

    for cr in result["criterion_scores"]:
        db.add(CriterionScore(
            workspace_id=session.workspace_id, score_id=score.id,
            criterion_id=cr["criterion_id"], criterion_name=cr["criterion_name"],
            weight=cr["weight"], order_index=cr["order_index"], raw_score=cr["raw_score"],
            weighted_contribution=cr["weighted_contribution"], confidence=cr["confidence"],
            evidence=cr["evidence"], reasoning=cr["reasoning"],
        ))
    await db.commit()
    await db.refresh(score)
    return score
