"""Question / branch-rule schemas (Feature 4)."""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID
from app.recruitment.models.enums import QuestionType


class QuestionConfig(BaseModel):
    max_answer_seconds: Optional[int] = Field(None, ge=5, le=900)
    probing_depth: int = Field(0, ge=0, le=5)
    required: bool = True
    options: Optional[List[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    response_type: Optional[str] = None   # text | single_select | multi_select | number | rating | yes_no | info
    scale: Optional[int] = Field(None, ge=2, le=10)


class QuestionCreate(BaseModel):
    question_type: QuestionType = QuestionType.OPEN_RESPONSE
    prompt_text: str = Field(..., min_length=1)
    order_index: Optional[int] = None
    config: Optional[QuestionConfig] = None
    is_knockout: bool = False
    parent_question_id: Optional[UUID] = None


class QuestionUpdate(BaseModel):
    question_type: Optional[QuestionType] = None
    prompt_text: Optional[str] = None
    order_index: Optional[int] = None
    config: Optional[QuestionConfig] = None
    is_knockout: Optional[bool] = None


class QuestionReorder(BaseModel):
    ordered_ids: List[UUID]


class QuestionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    version_id: UUID
    order_index: int
    question_type: str
    prompt_text: str
    config: Optional[Dict[str, Any]] = None
    is_knockout: bool
    parent_question_id: Optional[UUID] = None
    created_at: datetime


class BranchRuleCreate(BaseModel):
    condition: Dict[str, Any]
    action: Dict[str, Any]
    order_index: Optional[int] = None


class BranchRuleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    question_id: UUID
    order_index: int
    condition: Dict[str, Any]
    action: Dict[str, Any]
