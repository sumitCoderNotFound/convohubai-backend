"""Job schemas (Feature 1)."""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from app.recruitment.models.enums import JobStatus, EmploymentType


class JobCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    department: Optional[str] = None
    location: Optional[str] = None
    employment_type: EmploymentType = EmploymentType.FULL_TIME
    competency_profile: Optional[Dict[str, Any]] = None
    required_criteria: List[str] = []
    preferred_criteria: List[str] = []
    disqualifying_criteria: List[str] = []
    is_general_assessment: bool = False


class JobUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    department: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[EmploymentType] = None
    status: Optional[JobStatus] = None
    competency_profile: Optional[Dict[str, Any]] = None
    required_criteria: Optional[List[str]] = None
    preferred_criteria: Optional[List[str]] = None
    disqualifying_criteria: Optional[List[str]] = None


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    workspace_id: UUID
    title: str
    description: Optional[str] = None
    department: Optional[str] = None
    location: Optional[str] = None
    employment_type: str
    status: str
    competency_profile: Optional[Dict[str, Any]] = None
    required_criteria: Optional[List[Any]] = None
    preferred_criteria: Optional[List[Any]] = None
    disqualifying_criteria: Optional[List[Any]] = None
    is_general_assessment: bool
    created_at: datetime
    updated_at: datetime


class JobListResponse(BaseModel):
    items: List[JobResponse]
    total: int
    page: int
    page_size: int


class ParseJobDescriptionRequest(BaseModel):
    description: str = Field(..., min_length=20)


class ParseJobDescriptionResponse(BaseModel):
    competency_profile: Dict[str, Any]
    suggested_criteria: List[Dict[str, Any]]
    source: str  # "ai" | "fallback"
