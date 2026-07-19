"""Focused tests for deterministic review-memory storage."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.memory import CandidateMemory
from src.services.candidate_loader import load_candidate_bundle
from src.services.memory_loader import load_memory
from src.services.memory_store import (
    MemoryPersistenceError,
    apply_review_facts,
    deterministic_review_fact_id,
    save_memory_atomic,
)
from src.tools.scoring import build_candidate_skill_universe
from src.workflow.human_review import ReviewFactInput, ReviewFactType


def _empty_memory() -> CandidateMemory:
    return CandidateMemory(
        schema_version="1.0",
        candidate_id="cand-mira-solenne-001",
        facts=[],
    )


def _graphql(*tags: str) -> ReviewFactInput:
    return ReviewFactInput(
        fact_type=ReviewFactType.SKILL,
        statement="GraphQL is a skill I know.",
        normalized_value="GraphQL",
        skill_tags=list(tags) or ["GraphQL"],
    )


def test_apply_save_and_load_graphql_atomically(tmp_path: Path) -> None:
    original = _empty_memory()
    before = original.model_dump()
    updated = apply_review_facts(original, [_graphql()], 1, ["job-a"])
    path = tmp_path / "memory.json"
    save_memory_atomic(updated, path)
    loaded = load_memory(path, original.candidate_id)

    assert original.model_dump() == before
    assert len(loaded.facts) == 1
    fact = loaded.facts[0]
    assert fact.fact_id == deterministic_review_fact_id(_graphql())
    assert fact.fact_type == "skill"
    assert fact.provenance.source == "candidate_review"
    assert fact.provenance.review_round == 1
    assert fact.provenance.reviewer_role == "candidate"
    assert "human-review" in fact.provenance.run_id
    assert fact.evidence_refs == ["job-a"]
    assert path.read_bytes().endswith(b"\n")
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert not (tmp_path / ".memory.json.tmp").exists()

def test_new_fact_records_actual_utc_creation_time(monkeypatch) -> None:
    """A newly learned fact records the UTC time supplied at creation."""
    fixed_created_at = datetime(2026, 7, 19, 2, 35, 54, tzinfo=timezone.utc)

    monkeypatch.setattr(
        "src.services.memory_store._current_utc_created_at",
        lambda: fixed_created_at,
    )

    updated = apply_review_facts(
        _empty_memory(),
        [_graphql()],
        1,
        ["job-a"],
    )

    assert updated.facts[0].created_at == fixed_created_at
    assert updated.facts[0].created_at.utcoffset() is not None
    assert updated.facts[0].created_at.utcoffset().total_seconds() == 0

def test_equivalent_fact_deduplicates_and_merges_tags() -> None:
    original = _empty_memory()
    first = apply_review_facts(original, [_graphql("GraphQL", "API")], 1, ["job-a"])
    second = apply_review_facts(
        first,
        [_graphql("Schema", "graphql")],
        2,
        ["job-b"],
    )
    assert len(second.facts) == 1
    assert second.facts[0].skill_tags == ["API", "GraphQL", "Schema"]
    assert second.facts[0].evidence_refs == ["job-a", "job-b"]
    assert first.facts[0].skill_tags == ["API", "GraphQL"]


def test_candidate_fact_does_not_enter_skill_universe() -> None:
    bundle = load_candidate_bundle(
        ROOT / "candidate" / "profile.json",
        ROOT / "candidate" / "portfolio.json",
        ROOT / "candidate" / "evidence_registry.json",
    )
    candidate_fact = ReviewFactInput(
        fact_type=ReviewFactType.CANDIDATE_FACT,
        statement="GraphQL is a technology I prefer to avoid.",
        normalized_value="GraphQL",
        skill_tags=["GraphQL"],
    )
    updated = apply_review_facts(_empty_memory(), [candidate_fact], 1, ["job-a"])
    universe = build_candidate_skill_universe(bundle, updated)
    assert "graphql" not in universe.canonical_skills
    assert updated.facts[0].fact_type == "candidate_fact"


def test_invalid_fact_type_round_and_parent_fail_cleanly(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        ReviewFactInput(
            fact_type="opinion",
            statement="Unsupported.",
            normalized_value="x",
            skill_tags=[],
        )
    with pytest.raises(MemoryPersistenceError, match="1 or 2"):
        apply_review_facts(_empty_memory(), [_graphql()], 0, ["job-a"])
    with pytest.raises(MemoryPersistenceError, match="parent directory"):
        save_memory_atomic(_empty_memory(), tmp_path / "missing" / "memory.json")
    existing = tmp_path / "memory.json"
    existing.write_text(
        '{"schema_version":"1.0","candidate_id":"different","facts":[]}\n',
        encoding="utf-8",
    )
    before = existing.read_bytes()
    with pytest.raises(MemoryPersistenceError, match="candidate_id mismatch"):
        save_memory_atomic(_empty_memory(), existing)
    assert existing.read_bytes() == before
