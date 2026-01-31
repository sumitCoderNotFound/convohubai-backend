"""
ConvoHubAI - API Routes
"""
from fastapi import APIRouter
from app.api.routes import auth, workspaces, agents, chat, knowledge_base, webhooks, simulation

# Main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router)
api_router.include_router(workspaces.router)
api_router.include_router(agents.router)
api_router.include_router(chat.router)
api_router.include_router(knowledge_base.router)
api_router.include_router(webhooks.router)
api_router.include_router(simulation.router)

__all__ = ["api_router"]