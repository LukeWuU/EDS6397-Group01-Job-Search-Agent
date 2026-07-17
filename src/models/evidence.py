"""Evidence registry models grounded in candidate source files."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvidenceProvenance(BaseModel):
    """Flexible provenance metadata attached to evidence records."""

    model_config = ConfigDict(extra="allow")

    origin: str
    fictional: bool = True
    measurement_context: str | None = None
    data_classification: str | None = None
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class EvidenceRecord(BaseModel):
    """Single auditable evidence record supporting candidate claims."""

    evidence_id: str
    evidence_type: str
    source_file: str
    source_record_id: str
    claim: str
    supported_skills: list[str]
    provenance: EvidenceProvenance
    allowed_uses: list[str]


class EvidenceRegistry(BaseModel):
    """Complete evidence registry for the candidate."""

    schema_version: str
    candidate_id: str
    evidence_records: list[EvidenceRecord]
