"""
ConvoHubAI - Database Configuration
"""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from app.core.config import settings

# Async engine for FastAPI
async_engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Sync engine for Alembic migrations
if settings.database_url_sync:
    sync_engine = create_engine(
        settings.database_url_sync,
        echo=settings.debug,
        pool_pre_ping=True,
    )
else:
    # Convert async URL to sync URL
    sync_url = settings.database_url.replace("+asyncpg", "")
    sync_engine = create_engine(sync_url, echo=settings.debug, pool_pre_ping=True)

# Base class for all models
Base = declarative_base()


async def get_db() -> AsyncSession:
    """Dependency to get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database tables."""
    # Import all models to register them with Base.metadata
    from app.models.user import User, Workspace, WorkspaceMember
    from app.models.agent import Agent, AgentTemplate
    from app.models.knowledge_base import KnowledgeBase, Document
    from app.models.conversation import Conversation, Message
    from app.models.phone_number import PhoneNumber
    
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Database tables initialized")


async def close_db():
    """Close database connections."""
    await async_engine.dispose()