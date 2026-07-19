"""Tests for candidate memory loading."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.memory import CandidateMemory, MemoryFact, MemoryProvenance
from src.services.memory_loader import MemoryLoaderError, load_memory

MEMORY_PATH = ROOT / "memory.json"
CANDIDATE_ID = "cand-mira-solenne-001"


def test_empty_memory_document_loads(tmp_path: Path) -> None:
    """An isolated empty memory document validates successfully."""
    empty_memory_path = tmp_path / "memory.json"
    empty_memory_path.write_text(
        (
            '{\n'
            '  "schema_version": "1.0",\n'
            f'  "candidate_id": "{CANDIDATE_ID}",\n'
            '  "facts": []\n'
            '}\n'
        ),
        encoding="utf-8",
    )

    memory = load_memory(empty_memory_path, CANDIDATE_ID)

    assert memory.candidate_id == CANDIDATE_ID
    assert memory.schema_version == "1.0"
    assert memory.facts == []


def test_current_repository_memory_loads() -> None:
    """The repository memory document remains schema-valid after a final run."""
    memory = load_memory(MEMORY_PATH, CANDIDATE_ID)

    assert memory.candidate_id == CANDIDATE_ID
    assert memory.schema_version == "1.0"
    assert all(
        fact.provenance.source == "candidate_review"
        for fact in memory.facts
    )


def test_candidate_id_mismatch_fails(tmp_path: Path) -> None:
    """Memory candidate_id must match the expected candidate."""
    bad_memory = tmp_path / "memory.json"
    bad_memory.write_text(
        '{"schema_version":"1.0","candidate_id":"other","facts":[]}',
        encoding="utf-8",
    )

    with pytest.raises(MemoryLoaderError, match="candidate_id mismatch"):
        load_memory(bad_memory, CANDIDATE_ID)


def test_invalid_fact_type_fails() -> None:
    """Unsupported memory fact types are rejected."""
    with pytest.raises(ValidationError):
        MemoryFact(
            fact_id="fact-001",
            fact_type="job_opinion",
            statement="Unsupported fact",
            normalized_value="value",
            skill_tags=[],
            evidence_refs=["EV-EXP-001"],
            provenance=MemoryProvenance(
                source="candidate_review",
                review_round=1,
                run_id="run-001",
                reviewer_role="reviewer",
            ),
            created_at=datetime.now(timezone.utc),
            applied_in_run=False,
        )


def test_missing_provenance_on_nonempty_fact_fails() -> None:
    """Facts must include complete provenance metadata."""
    with pytest.raises(ValidationError):
        MemoryFact(
            fact_id="fact-001",
            fact_type="skill",
            statement="Candidate knows Kubernetes",
            normalized_value="Kubernetes",
            skill_tags=["Kubernetes"],
            evidence_refs=["EV-SKILL-MLOPS"],
            provenance=MemoryProvenance(
                source="candidate_review",
                review_round=1,
                run_id="",
                reviewer_role="reviewer",
            ),
            created_at=datetime.now(timezone.utc),
            applied_in_run=True,
        )


def test_nonempty_memory_document_validates() -> None:
    """A valid nonempty memory fact passes model validation."""
    memory = CandidateMemory(
        schema_version="1.0",
        candidate_id=CANDIDATE_ID,
        facts=[
            MemoryFact(
                fact_id="fact-001",
                fact_type="skill",
                statement="Candidate has used LangChain in coursework.",
                normalized_value="LangChain",
                skill_tags=["LangChain"],
                evidence_refs=["EV-PROJ-001"],
                provenance=MemoryProvenance(
                    source="candidate_review",
                    review_round=1,
                    run_id="run-001",
                    reviewer_role="reviewer",
                ),
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                applied_in_run=True,
            )
        ],
    )

    assert len(memory.facts) == 1
    assert memory.facts[0].fact_type == "skill"
