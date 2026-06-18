"""Feature 6 - Interview preview / simulation (no scoring, no credits)."""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.recruitment.models.interview import InterviewVersion
from app.recruitment.models.question import InterviewQuestion
from app.recruitment.schemas.simulation import SimulationRequest, SimulationResponse, SimulationTurn
from app.recruitment.services.simulation import simulate_interview
from app.recruitment.api.deps import get_ctx, WorkspaceContext

router = APIRouter(prefix="/recruitment/versions", tags=["Recruitment - Simulation"])


@router.post("/{version_id}/simulate", response_model=SimulationResponse)
async def simulate(version_id: UUID, payload: SimulationRequest, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    res = await db.execute(select(InterviewVersion).where(
        InterviewVersion.id == version_id, InterviewVersion.workspace_id == ctx.id, InterviewVersion.is_deleted == False))
    version = res.scalar_one_or_none()
    if not version:
        raise HTTPException(status_code=404, detail="Interview version not found")

    qres = await db.execute(select(InterviewQuestion).where(
        InterviewQuestion.version_id == version_id).order_by(InterviewQuestion.order_index))
    questions = list(qres.scalars().all())
    if not questions:
        raise HTTPException(status_code=400, detail="Add questions before previewing")

    turns = await simulate_interview(version, questions, payload.persona)
    return SimulationResponse(
        version_id=version_id,
        turns=[SimulationTurn(**t) for t in turns],
        note="Preview only. No candidate data stored and no credits consumed.")
