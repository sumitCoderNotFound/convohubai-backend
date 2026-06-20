"""Phase 2 - RecruitmentSettings: the three product decisions, per workspace."""
from sqlalchemy import Column, String, Text, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import BaseModel


class RecruitmentSettings(BaseModel):
    """Workspace-level recruitment policy (one row per workspace)."""
    __tablename__ = "recruitment_settings"

    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)

    # Decision D1 — jurisdiction drives consent wording
    jurisdiction = Column(String(10), default="uk", nullable=False)
    consent_text = Column(Text, nullable=True)  # overrides the jurisdiction default when set

    # Decision D3 — default recording behaviour for new interview versions
    default_recording_enabled = Column(Boolean, default=False, nullable=False)

    # Decision D2 — whether candidates can see their own score on the results screen
    candidates_see_scores = Column(Boolean, default=False, nullable=False)

    # Branding for the candidate portal
    brand_name = Column(String(255), nullable=True)
    brand_logo_url = Column(String(500), nullable=True)
    interviewer_avatar_url = Column(String(500), nullable=True)  # Phase 14: photo for the animated avatar

    def __repr__(self):
        return f"<RecruitmentSettings ws={self.workspace_id}>"
