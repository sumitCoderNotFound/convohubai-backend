"""Shared schema helpers."""
from pydantic import BaseModel
from typing import Generic, TypeVar, List

T = TypeVar("T")


class Page(BaseModel):
    """Pagination envelope."""
    total: int
    page: int
    page_size: int


class MessageResponse(BaseModel):
    message: str
