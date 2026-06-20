"""Candidate document API (Phase 12). Recruiter upload/list/download/delete."""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.recruitment.models.document import CandidateDocument
from app.recruitment.models.candidate import Candidate
from app.recruitment.schemas.document import DocumentResponse
from app.recruitment.schemas.common import MessageResponse
from app.recruitment.services import storage
from app.recruitment.api.deps import get_ctx, WorkspaceContext

router = APIRouter(prefix="/recruitment", tags=["Recruitment - Documents"])


@router.get("/candidates/{candidate_id}/documents", response_model=list[DocumentResponse])
async def list_documents(candidate_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    rows = await db.execute(select(CandidateDocument).where(
        CandidateDocument.candidate_id == candidate_id, CandidateDocument.workspace_id == ctx.id,
        CandidateDocument.is_deleted == False).order_by(CandidateDocument.created_at.desc()))
    return list(rows.scalars().all())


@router.post("/candidates/{candidate_id}/documents", response_model=DocumentResponse, status_code=201)
async def upload_document(candidate_id: UUID, file: UploadFile = File(...), kind: str = Form("resume"),
                          ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    cand = (await db.execute(select(Candidate).where(
        Candidate.id == candidate_id, Candidate.workspace_id == ctx.id))).scalar_one_or_none()
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")
    data = await file.read()
    try:
        storage.validate(file.filename, file.content_type, len(data))
    except storage.StorageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    path = storage.save_bytes(data, file.filename)
    doc = CandidateDocument(
        workspace_id=ctx.id, candidate_id=candidate_id, kind=kind, filename=file.filename,
        content_type=file.content_type, size=len(data), storage_path=path, source="recruiter",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


@router.get("/documents/{doc_id}/download")
async def download_document(doc_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    doc = (await db.execute(select(CandidateDocument).where(
        CandidateDocument.id == doc_id, CandidateDocument.workspace_id == ctx.id))).scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    import os
    if not os.path.exists(doc.storage_path):
        raise HTTPException(status_code=410, detail="File is no longer available on the server")
    return FileResponse(doc.storage_path, media_type=doc.content_type or "application/octet-stream", filename=doc.filename)


@router.delete("/documents/{doc_id}", response_model=MessageResponse)
async def delete_document(doc_id: UUID, ctx: WorkspaceContext = Depends(get_ctx), db: AsyncSession = Depends(get_db)):
    ctx.require_edit()
    doc = (await db.execute(select(CandidateDocument).where(
        CandidateDocument.id == doc_id, CandidateDocument.workspace_id == ctx.id))).scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    storage.delete_path(doc.storage_path)
    doc.is_deleted = True
    await db.commit()
    return MessageResponse(message="Document removed")
