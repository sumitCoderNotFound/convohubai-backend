"""ATS provider abstraction (Phase 9)."""
from app.recruitment.services.ats.providers import get_provider, SUPPORTED_PROVIDERS

__all__ = ["get_provider", "SUPPORTED_PROVIDERS"]
