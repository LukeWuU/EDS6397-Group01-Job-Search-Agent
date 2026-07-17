"""Focused tests for the exact five model-callable schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agent.tool_registry import (
    AssignmentToolRegistry,
    FilteringCallArguments,
)


def test_registry_exposes_exactly_five_typed_assignment_tools():
    schemas = AssignmentToolRegistry.model_schemas()
    names = [item["function"]["name"] for item in schemas]
    assert names == [
        "filter_jobs",
        "score_jobs",
        "analyze_fit",
        "tailor_resume",
        "generate_cover_letter",
    ]
    serialized = repr(schemas)
    assert "ResumeEditPlan" in serialized
    assert "CoverLetterPlan" in serialized
    for forbidden in (
        "human_review",
        "memory_store",
        "load_jobs",
        "finalize_resume",
        "tracing",
        "output_writer",
    ):
        assert forbidden not in names


def test_decision_summary_is_concise_nonempty_and_not_private_reasoning():
    assert FilteringCallArguments(
        decision_summary="  Filter the loaded jobs once. "
    ).decision_summary == "Filter the loaded jobs once."
    with pytest.raises(ValidationError):
        FilteringCallArguments(decision_summary="")
    with pytest.raises(ValidationError):
        FilteringCallArguments(decision_summary="x" * 501)
    with pytest.raises(ValidationError, match="chain-of-thought"):
        FilteringCallArguments(
            decision_summary="Here is my hidden chain of thought before filtering."
        )
