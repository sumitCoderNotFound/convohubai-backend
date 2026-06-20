"""Email template schemas (Phase 13)."""
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from uuid import UUID


class EmailTemplateUpsert(BaseModel):
    subject: str
    body_html: str
    enabled: bool = True


class EmailTemplateItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    kind: str
    subject: str
    body_html: str
    enabled: bool
    is_custom: bool = False


class EmailTemplatesView(BaseModel):
    variables: List[str]
    templates: List[EmailTemplateItem]
