"""
ConvoHubAI - Knowledge Base Model
RAG-powered knowledge for agents
"""
from sqlalchemy import Column, String, Boolean, Text, JSON, Enum, ForeignKey, BigInteger
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import enum

from app.models.base import BaseModel


class KnowledgeBaseType(str, enum.Enum):
    """Types of knowledge bases."""
    DOCUMENTS = "documents"
    WEBSITE = "website"
    API = "api"
    DATABASE = "database"


class KnowledgeBaseStatus(str, enum.Enum):
    """Knowledge base sync status."""
    PENDING = "pending"
    SYNCING = "syncing"
    SYNCED = "synced"
    FAILED = "failed"


class DocumentType(str, enum.Enum):
    """Document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    CSV = "csv"
    JSON = "json"
    URL = "url"


class KnowledgeBase(BaseModel):
    """Knowledge Base model for RAG."""
    __tablename__ = "knowledge_bases"
    
    # Basic info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Configuration
    kb_type = Column(
        Enum(KnowledgeBaseType),
        default=KnowledgeBaseType.DOCUMENTS,
        nullable=False
    )
    status = Column(
        Enum(KnowledgeBaseStatus),
        default=KnowledgeBaseStatus.PENDING,
        nullable=False
    )
    
    # Vector store
    vector_store_id = Column(String(255), nullable=True)  # Pinecone namespace/index
    embedding_model = Column(String(100), default="text-embedding-ada-002")
    
    # Settings
    chunk_size = Column(String(10), default="1000")
    chunk_overlap = Column(String(10), default="200")
    
    # Workspace
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Stats
    document_count = Column(String(10), default="0")
    total_chunks = Column(String(20), default="0")
    total_tokens = Column(String(20), default="0")
    storage_bytes = Column(BigInteger, default=0)
    
    # Relationships
    workspace = relationship("Workspace", back_populates="knowledge_bases")
    documents = relationship(
        "Document",
        back_populates="knowledge_base",
        cascade="all, delete-orphan"
    )
    agents = relationship("Agent", back_populates="knowledge_base")
    
    def __repr__(self):
        return f"<KnowledgeBase {self.name}>"


class Document(BaseModel):
    """Document in a knowledge base."""
    __tablename__ = "documents"
    
    # Basic info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # File info
    doc_type = Column(Enum(DocumentType), nullable=False)
    file_url = Column(String(500), nullable=True)
    file_size = Column(BigInteger, default=0)
    mime_type = Column(String(100), nullable=True)
    
    # For URL sources
    source_url = Column(String(500), nullable=True)
    
    # Processing status
    status = Column(
        Enum(KnowledgeBaseStatus),
        default=KnowledgeBaseStatus.PENDING,
        nullable=False
    )
    error_message = Column(Text, nullable=True)
    
    # Processing stats
    chunk_count = Column(String(10), default="0")
    token_count = Column(String(20), default="0")
    
    # Content (for small docs, stored directly)
    content = Column(Text, nullable=True)
    
    # Metadata
    extra_data = Column(JSON, nullable=True)
    
    # Knowledge base
    knowledge_base_id = Column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Uploader
    uploaded_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )
    
    # Relationships
    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    uploaded_by = relationship("User", foreign_keys=[uploaded_by_id])
    
    def __repr__(self):
        return f"<Document {self.name}>"
