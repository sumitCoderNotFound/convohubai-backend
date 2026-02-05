"""
ConvoHubAI - API Routes
"""
from fastapi import APIRouter

# Main API router
api_router = APIRouter()

# Import and include all route modules one by one
# This helps identify which module has import errors

from app.api.routes import auth
api_router.include_router(auth.router)

from app.api.routes import workspaces
api_router.include_router(workspaces.router)

from app.api.routes import agents
api_router.include_router(agents.router)

from app.api.routes import chat
api_router.include_router(chat.router)

from app.api.routes import knowledge_base
api_router.include_router(knowledge_base.router)

from app.api.routes import webhooks
api_router.include_router(webhooks.router)

from app.api.routes import simulation
api_router.include_router(simulation.router)

from app.api.routes import voice
api_router.include_router(voice.router)

from app.api.routes import flows
api_router.include_router(flows.router)

from app.api.routes import dashboard
api_router.include_router(dashboard.router)

from app.api.routes import analytics
api_router.include_router(analytics.router)

from app.api.routes import settings
api_router.include_router(settings.router)

from app.api.routes import monitor
api_router.include_router(monitor.router)


__all__ = ["api_router"]