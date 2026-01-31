"""
ConvoHubAI - Knowledge Base API Routes
Handles knowledge base management and document uploads
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.knowledge_base import KnowledgeBase, Document, KnowledgeBaseType, KnowledgeBaseStatus, DocumentType
from app.models.agent import Agent


router = APIRouter(prefix="/knowledge-bases", tags=["Knowledge Base"])


# ============================================
# SCHEMAS
# ============================================

class KnowledgeBaseCreate(BaseModel):
    name: str
    description: Optional[str] = None
    kb_type: str = "documents"


class KnowledgeBaseUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class KnowledgeBaseResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    kb_type: str
    status: str
    document_count: str
    total_chunks: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str]
    doc_type: str
    status: str
    file_size: int
    chunk_count: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class ConnectKnowledgeBaseRequest(BaseModel):
    knowledge_base_id: UUID


# ============================================
# KNOWLEDGE BASE ROUTES
# ============================================

@router.get("", response_model=List[KnowledgeBaseResponse])
async def list_knowledge_bases(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all knowledge bases in the workspace."""
    result = await db.execute(
        select(KnowledgeBase).where(
            and_(
                KnowledgeBase.workspace_id == current_user.current_workspace_id,
                KnowledgeBase.is_deleted == False,
            )
        ).order_by(KnowledgeBase.created_at.desc())
    )
    knowledge_bases = result.scalars().all()
    
    return [
        KnowledgeBaseResponse(
            id=kb.id,
            name=kb.name,
            description=kb.description,
            kb_type=kb.kb_type.value,
            status=kb.status.value,
            document_count=kb.document_count,
            total_chunks=kb.total_chunks,
            created_at=kb.created_at,
            updated_at=kb.updated_at,
        )
        for kb in knowledge_bases
    ]


@router.post("", response_model=KnowledgeBaseResponse)
async def create_knowledge_base(
    data: KnowledgeBaseCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new knowledge base."""
    kb = KnowledgeBase(
        name=data.name,
        description=data.description,
        kb_type=KnowledgeBaseType(data.kb_type),
        workspace_id=current_user.current_workspace_id,
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    
    return KnowledgeBaseResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        kb_type=kb.kb_type.value,
        status=kb.status.value,
        document_count=kb.document_count,
        total_chunks=kb.total_chunks,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific knowledge base."""
    result = await db.execute(
        select(KnowledgeBase).where(
            and_(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.workspace_id == current_user.current_workspace_id,
                KnowledgeBase.is_deleted == False,
            )
        )
    )
    kb = result.scalar_one_or_none()
    
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    return KnowledgeBaseResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        kb_type=kb.kb_type.value,
        status=kb.status.value,
        document_count=kb.document_count,
        total_chunks=kb.total_chunks,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


@router.patch("/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    kb_id: UUID,
    data: KnowledgeBaseUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a knowledge base."""
    result = await db.execute(
        select(KnowledgeBase).where(
            and_(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.workspace_id == current_user.current_workspace_id,
                KnowledgeBase.is_deleted == False,
            )
        )
    )
    kb = result.scalar_one_or_none()
    
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    if data.name is not None:
        kb.name = data.name
    if data.description is not None:
        kb.description = data.description
    
    kb.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(kb)
    
    return KnowledgeBaseResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        kb_type=kb.kb_type.value,
        status=kb.status.value,
        document_count=kb.document_count,
        total_chunks=kb.total_chunks,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a knowledge base."""
    result = await db.execute(
        select(KnowledgeBase).where(
            and_(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    kb = result.scalar_one_or_none()
    
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    kb.is_deleted = True
    kb.updated_at = datetime.utcnow()
    await db.commit()
    
    return {"message": "Knowledge base deleted"}


# ============================================
# DOCUMENT ROUTES
# ============================================

@router.get("/{kb_id}/documents", response_model=List[DocumentResponse])
async def list_documents(
    kb_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all documents in a knowledge base."""
    # Verify knowledge base access
    kb_result = await db.execute(
        select(KnowledgeBase).where(
            and_(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    kb = kb_result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    result = await db.execute(
        select(Document).where(
            and_(
                Document.knowledge_base_id == kb_id,
                Document.is_deleted == False,
            )
        ).order_by(Document.created_at.desc())
    )
    documents = result.scalars().all()
    
    return [
        DocumentResponse(
            id=doc.id,
            name=doc.name,
            description=doc.description,
            doc_type=doc.doc_type.value,
            status=doc.status.value,
            file_size=doc.file_size,
            chunk_count=doc.chunk_count,
            created_at=doc.created_at,
        )
        for doc in documents
    ]


@router.post("/{kb_id}/documents")
async def upload_document(
    kb_id: UUID,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a document to a knowledge base."""
    # Verify knowledge base access
    kb_result = await db.execute(
        select(KnowledgeBase).where(
            and_(
                KnowledgeBase.id == kb_id,
                KnowledgeBase.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    kb = kb_result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Determine document type from filename
    filename = file.filename or "unknown"
    extension = filename.split(".")[-1].lower()
    doc_type_map = {
        "pdf": DocumentType.PDF,
        "docx": DocumentType.DOCX,
        "txt": DocumentType.TXT,
        "md": DocumentType.MD,
        "html": DocumentType.HTML,
        "csv": DocumentType.CSV,
        "json": DocumentType.JSON,
    }
    doc_type = doc_type_map.get(extension, DocumentType.TXT)
    
    # Read file content
    content = await file.read()
    file_size = len(content)
    
    # Create document record
    document = Document(
        name=filename,
        doc_type=doc_type,
        file_size=file_size,
        mime_type=file.content_type,
        status=KnowledgeBaseStatus.PENDING,
        knowledge_base_id=kb_id,
        uploaded_by_id=current_user.id,
        content=content.decode("utf-8", errors="ignore") if file_size < 1000000 else None,
    )
    db.add(document)
    
    # Update knowledge base document count
    kb.document_count = str(int(kb.document_count or "0") + 1)
    kb.updated_at = datetime.utcnow()
    
    await db.commit()
    await db.refresh(document)
    
    return {
        "id": document.id,
        "name": document.name,
        "status": document.status.value,
        "message": "Document uploaded successfully. Processing will begin shortly."
    }


@router.delete("/{kb_id}/documents/{doc_id}")
async def delete_document(
    kb_id: UUID,
    doc_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a document from a knowledge base."""
    result = await db.execute(
        select(Document).where(
            and_(
                Document.id == doc_id,
                Document.knowledge_base_id == kb_id,
            )
        )
    )
    document = result.scalar_one_or_none()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    document.is_deleted = True
    document.updated_at = datetime.utcnow()
    
    # Update knowledge base document count
    kb_result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == kb_id)
    )
    kb = kb_result.scalar_one_or_none()
    if kb:
        kb.document_count = str(max(0, int(kb.document_count or "0") - 1))
        kb.updated_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Document deleted"}


# ============================================
# AGENT CONNECTION ROUTES
# ============================================

@router.post("/connect-to-agent/{agent_id}")
async def connect_knowledge_base_to_agent(
    agent_id: UUID,
    data: ConnectKnowledgeBaseRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Connect a knowledge base to an agent."""
    # Verify agent access
    agent_result = await db.execute(
        select(Agent).where(
            and_(
                Agent.id == agent_id,
                Agent.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    agent = agent_result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Verify knowledge base access
    kb_result = await db.execute(
        select(KnowledgeBase).where(
            and_(
                KnowledgeBase.id == data.knowledge_base_id,
                KnowledgeBase.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    kb = kb_result.scalar_one_or_none()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    
    # Connect
    agent.knowledge_base_id = data.knowledge_base_id
    agent.updated_at = datetime.utcnow()
    await db.commit()
    
    return {"message": f"Knowledge base '{kb.name}' connected to agent '{agent.name}'"}


@router.post("/disconnect-from-agent/{agent_id}")
async def disconnect_knowledge_base_from_agent(
    agent_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Disconnect knowledge base from an agent."""
    agent_result = await db.execute(
        select(Agent).where(
            and_(
                Agent.id == agent_id,
                Agent.workspace_id == current_user.current_workspace_id,
            )
        )
    )
    agent = agent_result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent.knowledge_base_id = None
    agent.updated_at = datetime.utcnow()
    await db.commit()
    
    return {"message": "Knowledge base disconnected from agent"}