"""Recruitment API router aggregator."""
from fastapi import APIRouter

from app.recruitment.api.jobs import router as jobs_router
from app.recruitment.api.candidates import router as candidates_router
from app.recruitment.api.interviews import router as interviews_router
from app.recruitment.api.questions import router as questions_router
from app.recruitment.api.rubrics import router as rubrics_router
from app.recruitment.api.simulation import router as simulation_router
from app.recruitment.api.invites import router as invites_router
from app.recruitment.api.sessions import router as sessions_router
from app.recruitment.api.settings import router as settings_router
from app.recruitment.api.public import router as public_router
from app.recruitment.api.analytics import router as analytics_router
from app.recruitment.api.ats import router as ats_router
from app.recruitment.api.prescreen import router as prescreen_router
from app.recruitment.api.documents import router as documents_router
from app.recruitment.api.email_templates import router as email_templates_router

recruitment_router = APIRouter()
# Phase 1
recruitment_router.include_router(jobs_router)
recruitment_router.include_router(candidates_router)
recruitment_router.include_router(interviews_router)
recruitment_router.include_router(questions_router)
recruitment_router.include_router(rubrics_router)
recruitment_router.include_router(simulation_router)
# Phase 2
recruitment_router.include_router(invites_router)
recruitment_router.include_router(sessions_router)
recruitment_router.include_router(settings_router)
recruitment_router.include_router(public_router)
recruitment_router.include_router(analytics_router)
recruitment_router.include_router(ats_router)
recruitment_router.include_router(prescreen_router)
recruitment_router.include_router(documents_router)
recruitment_router.include_router(email_templates_router)

__all__ = ["recruitment_router"]
