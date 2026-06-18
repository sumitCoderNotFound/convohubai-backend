"""Interview preview / simulation schemas (Feature 6)."""
from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID


class SimulationRequest(BaseModel):
    persona: Optional[str] = Field(
        "an average, reasonably qualified candidate",
        description="Persona the simulated candidate should adopt.",
    )


class SimulationTurn(BaseModel):
    role: str  # "interviewer" | "candidate"
    question_id: Optional[UUID] = None
    text: str


class SimulationResponse(BaseModel):
    version_id: UUID
    is_preview: bool = True
    consumes_credits: bool = False
    turns: List[SimulationTurn]
    note: str
