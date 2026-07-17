"""Deterministic candidate memory loader."""

from __future__ import annotations

import json
from pathlib import Path

from src.models.memory import CandidateMemory


class MemoryLoaderError(Exception):
    """Raised when candidate memory cannot be loaded or validated."""


def load_memory(path: Path, expected_candidate_id: str) -> CandidateMemory:
    """Load and validate persistent candidate memory without mutating the file."""
    if not path.is_file():
        raise MemoryLoaderError(f"Memory file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    memory = CandidateMemory.model_validate(payload)

    if memory.candidate_id != expected_candidate_id:
        raise MemoryLoaderError(
            "candidate_id mismatch: "
            f"expected {expected_candidate_id!r}, found {memory.candidate_id!r}"
        )

    return memory
