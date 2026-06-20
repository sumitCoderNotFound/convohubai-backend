"""Pre-screening schemas (Phase 11)."""
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Any
from uuid import UUID
from datetime import datetime


class PreScreenQuestionCreate(BaseModel):
    prompt: str
    qtype: str = "yes_no"  # yes_no | single_select | number | text
    options: List[str] = Field(default_factory=list)
    knockout: Optional[dict] = None
    required: bool = True
    order_index: int = 0


class PreScreenQuestionUpdate(BaseModel):
    prompt: Optional[str] = None
    qtype: Optional[str] = None
    options: Optional[List[str]] = None
    knockout: Optional[dict] = None
    required: Optional[bool] = None
    order_index: Optional[int] = None


class PreScreenQuestionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    version_id: UUID
    prompt: str
    qtype: str
    options: List[str] = Field(default_factory=list)
    knockout: Optional[dict] = None
    required: bool
    order_index: int


# Candidate-facing: knockout rules are intentionally hidden.
class PreScreenPublicQuestion(BaseModel):
    id: UUID
    prompt: str
    qtype: str
    options: List[str] = Field(default_factory=list)
    required: bool


class PreScreenPublicView(BaseModel):
    questions: List[PreScreenPublicQuestion] = Field(default_factory=list)


class PreScreenAnswer(BaseModel):
    question_id: UUID
    value: Any = None


class PreScreenSubmit(BaseModel):
    answers: List[PreScreenAnswer] = Field(default_factory=list)


class PreScreenSubmitResult(BaseModel):
    eligible: bool
    message: str


class PreScreenResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    candidate_email: Optional[str] = None
    answers: List[dict] = Field(default_factory=list)
    auto_eligible: bool
    eligible: bool
    overridden: bool
    override_note: Optional[str] = None
    created_at: datetime


class PreScreenOverrideRequest(BaseModel):
    eligible: bool
    note: Optional[str] = None
