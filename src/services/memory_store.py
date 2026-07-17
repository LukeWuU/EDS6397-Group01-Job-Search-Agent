"""Deterministic review-memory updates and atomic persistence."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from src.models.memory import CandidateMemory, MemoryFact, MemoryProvenance

if TYPE_CHECKING:
    from src.workflow.human_review import ReviewFactInput


class MemoryPersistenceError(Exception):
    """Raised when review memory cannot be validated or saved atomically."""


def _normalized_value_key(value: object) -> str:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, list):
        normalized = sorted({" ".join(item.casefold().split()) for item in value})
        return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _fact_key(fact_type: str, statement: str, normalized_value: object) -> str:
    normalized = _normalized_value_key(normalized_value)
    identity = normalized if normalized not in {"", "[]"} else " ".join(statement.casefold().split())
    return f"{fact_type}:{identity}"


def deterministic_review_fact_id(fact: "ReviewFactInput") -> str:
    """Return a stable ID for an equivalent candidate-stated fact."""
    key = _fact_key(fact.fact_type.value, fact.statement, fact.normalized_value)
    return f"fact-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:20]}"


def _deterministic_created_at(fact_id: str) -> datetime:
    """Provide a schema-required timestamp without making results nondeterministic."""
    seconds = int(hashlib.sha256(fact_id.encode("utf-8")).hexdigest()[:8], 16)
    return datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds % 31_536_000)


def _merge_tags(existing: Sequence[str], incoming: Sequence[str]) -> list[str]:
    by_key: dict[str, str] = {}
    for tag in [*existing, *incoming]:
        cleaned = " ".join(tag.split())
        if cleaned:
            by_key.setdefault(cleaned.casefold(), cleaned)
    return [by_key[key] for key in sorted(by_key)]


def apply_review_facts(
    memory: CandidateMemory,
    learned_facts: Sequence["ReviewFactInput"],
    review_round: int,
    source_job_ids: Sequence[str],
) -> CandidateMemory:
    """Return a new memory with validated, deduplicated candidate review facts."""
    if not memory.candidate_id.strip():
        raise MemoryPersistenceError("CandidateMemory.candidate_id must be nonempty")
    if review_round not in {1, 2}:
        raise MemoryPersistenceError("Memory provenance review_round must be 1 or 2")
    source_ids = sorted({job_id.strip() for job_id in source_job_ids if job_id.strip()})
    if learned_facts and not source_ids:
        raise MemoryPersistenceError("Learned facts require at least one source job ID")

    facts = [fact.model_copy(deep=True) for fact in memory.facts]
    existing_by_key = {
        _fact_key(fact.fact_type, fact.statement, fact.normalized_value): index
        for index, fact in enumerate(facts)
    }

    for learned in learned_facts:
        fact_type = learned.fact_type.value
        if fact_type not in {"skill", "candidate_fact"}:
            raise MemoryPersistenceError(f"Unsupported review fact type: {fact_type!r}")
        key = _fact_key(fact_type, learned.statement, learned.normalized_value)
        evidence_refs = source_ids
        if key in existing_by_key:
            index = existing_by_key[key]
            existing = facts[index]
            facts[index] = existing.model_copy(
                update={
                    "skill_tags": _merge_tags(existing.skill_tags, learned.skill_tags),
                    "evidence_refs": sorted(set(existing.evidence_refs) | set(evidence_refs)),
                    "applied_in_run": True,
                },
                deep=True,
            )
            continue

        fact_id = deterministic_review_fact_id(learned)
        run_id = f"human-review:r{review_round}:{','.join(source_ids)}"
        memory_fact = MemoryFact(
            fact_id=fact_id,
            fact_type=fact_type,
            statement=learned.statement.strip(),
            normalized_value=learned.normalized_value,
            skill_tags=_merge_tags([], learned.skill_tags),
            evidence_refs=evidence_refs,
            provenance=MemoryProvenance(
                source="candidate_review",
                review_round=review_round,
                run_id=run_id,
                reviewer_role="candidate",
            ),
            created_at=_deterministic_created_at(fact_id),
            applied_in_run=True,
        )
        existing_by_key[key] = len(facts)
        facts.append(memory_fact)

    return memory.model_copy(update={"facts": facts}, deep=True)


def save_memory_atomic(memory: CandidateMemory, path: Path) -> None:
    """Save schema-compatible UTF-8 JSON through an atomic sibling replacement."""
    path = path.resolve()
    if not path.parent.is_dir():
        raise MemoryPersistenceError(
            f"Memory parent directory must already exist: {path.parent}"
        )
    if path.exists():
        try:
            existing_payload = json.loads(path.read_text(encoding="utf-8"))
            existing_candidate_id = existing_payload.get("candidate_id")
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise MemoryPersistenceError(
                f"Existing memory file is unreadable and will not be overwritten: {path}"
            ) from exc
        if existing_candidate_id != memory.candidate_id:
            raise MemoryPersistenceError(
                "candidate_id mismatch prevents memory overwrite: "
                f"existing={existing_candidate_id!r}, new={memory.candidate_id!r}"
            )
    temp_path = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(
        memory.model_dump(mode="json"),
        indent=2,
        ensure_ascii=False,
    ) + "\n"
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except (OSError, TypeError, ValueError) as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise MemoryPersistenceError(f"Failed to save memory atomically to {path}: {exc}") from exc


__all__ = [
    "MemoryPersistenceError",
    "apply_review_facts",
    "deterministic_review_fact_id",
    "save_memory_atomic",
]
