"""Recruitment settings schemas (the three product decisions; Phase 2)."""
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from app.recruitment.models.enums import Jurisdiction


class SettingsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    jurisdiction: str
    consent_text: Optional[str] = None
    default_recording_enabled: bool
    candidates_see_scores: bool
    brand_name: Optional[str] = None
    brand_logo_url: Optional[str] = None


class SettingsUpdate(BaseModel):
    jurisdiction: Optional[Jurisdiction] = None
    consent_text: Optional[str] = None
    default_recording_enabled: Optional[bool] = None
    candidates_see_scores: Optional[bool] = None
    brand_name: Optional[str] = None
    brand_logo_url: Optional[str] = None
