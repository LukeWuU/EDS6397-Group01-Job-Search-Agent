"""Persistent candidate memory models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

MemoryFactType = Literal["skill", "candidate_fact"]
NormalizedValue = Union[str, int, float, bool, list[str]]


class MemoryProvenance(BaseModel):
    """Provenance required for every persisted memory fact."""

    source: Literal["candidate_review"]
    review_round: int = Field(ge=1, le=2)
    run_id: str
    reviewer_role: str


class MemoryFact(BaseModel):
    """Runtime memory fact learned during the single review pause."""

    fact_id: str
    fact_type: MemoryFactType
    statement: str
    normalized_value: NormalizedValue
    skill_tags: list[str]
    evidence_refs: list[str]
    provenance: MemoryProvenance
    created_at: datetime
    applied_in_run: bool

    @field_validator("fact_type")
    @classmethod
    def validate_fact_type(cls, value: str) -> str:
        """Restrict memory facts to supported categories."""
        if value not in {"skill", "candidate_fact"}:
            raise ValueError(f"Unsupported memory fact type: {value}")
        return value

    @model_validator(mode="after")
    def validate_nonempty_provenance(self) -> "MemoryFact":
        """Ensure provenance is present for persisted facts."""
        if not self.provenance.run_id.strip():
            raise ValueError("Memory fact provenance.run_id must be non-empty")
        return self


class CandidateMemory(BaseModel):
    """Candidate memory document loaded at application startup."""

    schema_version: str
    candidate_id: str
    facts: list[MemoryFact]
