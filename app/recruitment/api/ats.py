"""ATS integration API (Phase 9). Manage connections + import jobs/candidates + push results."""
import asyncio
from datetime import datetime
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.recruitment.models.ats import AtsConnection, AtsMapping
from app.recruitment.models.job import JobPosition
from app.recruitment.models.candidate import Candidate, Application, ApplicationHistory
from app.recruitment.models.score import InterviewScore
from app.recruitment.schemas.ats import (
    AtsConnectionCreate, AtsConnectionUpdate, AtsConnectionResponse,
    TestResult, ImportRequest, ImportResult, PushResult,
)
from app.recruitment.schemas.common import MessageResponse
from app.recruitment.services.ats import get_provider, SUPPORTED_PROVIDERS
from app.recruitment.services.ats.providers import AtsError
from app.recruitment.api.deps import get_ctx, WorkspaceContext

router = APIRouter(prefix="/recruitment/ats", tags=["Recruitment - ATS"])


def _to_response(c: AtsConnection) -> AtsConnectionResponse:
    return AtsConnectionResponse(
        id=c.id, provider=c.provider, name=c.name, subdomain=c.subdomain, base_url=c.base_url,
        enabled=c.enabled, has_key=bool(c.api_key), last_sync_at=c.last_sync_at, created_at=c.created_at,
    )


async def _conn_or_404(cid: UUID, ws_id: UUID, db: AsyncSession) -> AtsConnection:
    c = (await db.execute(select(AtsConnection).where(
        AtsConnection.id == cid, AtsConnection.workspace_id == ws_id, AtsConnection.is_deleted == False))).scalar_one_or_none()
    if not c:
        raise HTTPException(status_code=404, detail="ATS connection not found")
    return c


def _provider_for(c: AtsConnection):
    return get_provider(c.provider, api_key=c.api_key or "", subdomain=c.subdomain or "", base_url=c.base_url or "")


@router.get("/providers")
async def list_providers(ctx: WorkspaceContext = Depends(get_ctx)):
    return {"providers": list(SUPPORTED_PROVIDERS)}


@router.get("/connections", response_model=list[AtsConnectionResponse])
async def list_connections(ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    rows = await db.execute(select(AtsConnection).where(
        AtsConnection.workspace_id == ctx.id, AtsConnection.is_deleted == False).order_by(AtsConnection.created_at.desc()))
    return [_to_response(c) for c in rows.scalars().all()]


@router.post("/connections", response_model=AtsConnectionResponse, status_code=201)
async def create_connection(payload: AtsConnectionCreate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    if payload.provider.lower() not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider. Supported: {', '.join(SUPPORTED_PROVIDERS)}")
    c = AtsConnection(
        workspace_id=ctx.id, created_by_id=ctx.user.id, provider=payload.provider.lower(),
        name=payload.name or payload.provider.title(), api_key=payload.api_key,
        subdomain=payload.subdomain, base_url=payload.base_url, enabled=True,
    )
    db.add(c)
    await db.commit()
    await db.refresh(c)
    return _to_response(c)


@router.patch("/connections/{cid}", response_model=AtsConnectionResponse)
async def update_connection(cid: UUID, payload: AtsConnectionUpdate, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    c = await _conn_or_404(cid, ctx.id, db)
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(c, k, v)
    await db.commit()
    await db.refresh(c)
    return _to_response(c)


@router.delete("/connections/{cid}", response_model=MessageResponse)
async def delete_connection(cid: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    c = await _conn_or_404(cid, ctx.id, db)
    c.is_deleted = True
    await db.commit()
    return MessageResponse(message="Connection removed")


@router.post("/connections/{cid}/test", response_model=TestResult)
async def test_connection(cid: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    c = await _conn_or_404(cid, ctx.id, db)
    try:
        provider = _provider_for(c)
        res = await asyncio.to_thread(provider.test)
        return TestResult(**res)
    except AtsError as e:
        return TestResult(ok=False, message=str(e))


@router.post("/connections/{cid}/import-jobs", response_model=ImportResult)
async def import_jobs(cid: UUID, payload: ImportRequest, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    c = await _conn_or_404(cid, ctx.id, db)
    result = ImportResult()
    try:
        provider = _provider_for(c)
        jobs = await asyncio.to_thread(provider.list_jobs, payload.limit)
    except AtsError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ATS request failed: {e}")

    result.total_seen = len(jobs)
    for j in jobs:
        try:
            mapping = (await db.execute(select(AtsMapping).where(
                AtsMapping.connection_id == c.id, AtsMapping.entity_type == "job",
                AtsMapping.external_id == j["external_id"]))).scalar_one_or_none()
            if mapping:
                job = (await db.execute(select(JobPosition).where(JobPosition.id == mapping.internal_id))).scalar_one_or_none()
                if job:
                    job.title = j["title"]
                    job.description = j.get("description") or job.description
                    result.matched += 1
                    continue
            job = JobPosition(workspace_id=ctx.id, title=j["title"], description=j.get("description"),
                              location=j.get("location"), status="open")
            db.add(job)
            await db.flush()
            db.add(AtsMapping(workspace_id=ctx.id, connection_id=c.id, entity_type="job",
                              external_id=j["external_id"], internal_id=job.id))
            result.created += 1
        except Exception as e:
            result.errors.append(f"{j.get('title','job')}: {e}")
    c.last_sync_at = datetime.utcnow()
    await db.commit()
    return result


@router.post("/connections/{cid}/import-candidates", response_model=ImportResult)
async def import_candidates(cid: UUID, payload: ImportRequest, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    c = await _conn_or_404(cid, ctx.id, db)
    result = ImportResult()

    job = None
    if payload.job_position_id:
        job = (await db.execute(select(JobPosition).where(
            JobPosition.id == payload.job_position_id, JobPosition.workspace_id == ctx.id))).scalar_one_or_none()
        if not job:
            raise HTTPException(status_code=404, detail="Target job not found")

    try:
        provider = _provider_for(c)
        cands = await asyncio.to_thread(provider.list_candidates, payload.limit)
    except AtsError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ATS request failed: {e}")

    result.total_seen = len(cands)
    for cd in cands:
        try:
            cand = None
            mapping = (await db.execute(select(AtsMapping).where(
                AtsMapping.connection_id == c.id, AtsMapping.entity_type == "candidate",
                AtsMapping.external_id == cd["external_id"]))).scalar_one_or_none()
            if mapping:
                cand = (await db.execute(select(Candidate).where(Candidate.id == mapping.internal_id))).scalar_one_or_none()
            if not cand and cd.get("email"):
                cand = (await db.execute(select(Candidate).where(
                    Candidate.workspace_id == ctx.id, func.lower(Candidate.email) == cd["email"].lower(),
                    Candidate.is_deleted == False))).scalar_one_or_none()
            if cand:
                result.matched += 1
            else:
                cand = Candidate(workspace_id=ctx.id, full_name=cd["full_name"], email=cd.get("email"),
                                 phone=cd.get("phone"), source=f"ats_{c.provider}")
                db.add(cand)
                await db.flush()
                result.created += 1
            if not mapping:
                db.add(AtsMapping(workspace_id=ctx.id, connection_id=c.id, entity_type="candidate",
                                  external_id=cd["external_id"], internal_id=cand.id))
            if job:
                exists = (await db.execute(select(Application).where(
                    Application.workspace_id == ctx.id, Application.candidate_id == cand.id,
                    Application.job_position_id == job.id, Application.is_deleted == False))).scalar_one_or_none()
                if not exists:
                    app = Application(workspace_id=ctx.id, candidate_id=cand.id, job_position_id=job.id,
                                      stage="applied", status="active")
                    db.add(app)
                    await db.flush()
                    db.add(ApplicationHistory(workspace_id=ctx.id, application_id=app.id, from_stage=None,
                                              to_stage="applied", actor_user_id=ctx.user.id, reason=f"Imported from {c.provider}"))
                    result.applications_created += 1
        except Exception as e:
            result.errors.append(f"{cd.get('full_name','candidate')}: {e}")
    c.last_sync_at = datetime.utcnow()
    await db.commit()
    return result


@router.post("/connections/{cid}/push/{application_id}", response_model=PushResult)
async def push_result(cid: UUID, application_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    """Push an interview score back to the ATS as a note on the candidate."""
    ctx.require_edit()
    c = await _conn_or_404(cid, ctx.id, db)
    app = (await db.execute(select(Application).where(
        Application.id == application_id, Application.workspace_id == ctx.id))).scalar_one_or_none()
    if not app:
        raise HTTPException(status_code=404, detail="Application not found")
    score = (await db.execute(select(InterviewScore).where(
        InterviewScore.application_id == application_id, InterviewScore.workspace_id == ctx.id))).scalar_one_or_none()
    if not score or score.status != "completed":
        raise HTTPException(status_code=409, detail="No completed score to push for this application")
    mapping = (await db.execute(select(AtsMapping).where(
        AtsMapping.connection_id == c.id, AtsMapping.entity_type == "candidate",
        AtsMapping.internal_id == app.candidate_id))).scalar_one_or_none()
    if not mapping:
        raise HTTPException(status_code=400, detail="This candidate has no link to the ATS (import them from the ATS first)")

    summary = f"AI interview score: {round(score.overall_score or 0)}/100 ({score.recommendation}). {score.summary or ''}"
    try:
        provider = _provider_for(c)
        res = await asyncio.to_thread(provider.push_result, mapping.external_id, summary)
        return PushResult(**res)
    except AtsError as e:
        return PushResult(ok=False, message=str(e))
