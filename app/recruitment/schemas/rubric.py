"""Rubric criterion / anchor schemas (Feature 5)."""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime
from uuid import UUID
from app.recruitment.models.enums import CriterionLevel


class ScoreAnchorInput(BaseModel):
    level: CriterionLevel
    descriptor: str = Field(..., min_length=1)


class ScoreAnchorResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    level: str
    descriptor: str


class CriterionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    weight: float = Field(..., ge=0, le=100)
    evidence_instructions: Optional[str] = None
    order_index: Optional[int] = None
    anchors: List[ScoreAnchorInput] = []


class CriterionUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    weight: Optional[float] = Field(None, ge=0, le=100)
    evidence_instructions: Optional[str] = None
    order_index: Optional[int] = None
    anchors: Optional[List[ScoreAnchorInput]] = None


class CriterionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    version_id: UUID
    name: str
    description: Optional[str] = None
    weight: float
    evidence_instructions: Optional[str] = None
    order_index: int
    is_blocked_sensitive: bool
    anchors: List[ScoreAnchorResponse] = []
    created_at: datetime


class RubricResponse(BaseModel):
    version_id: UUID
    criteria: List[CriterionResponse]
    total_weight: float
    weights_valid: bool
