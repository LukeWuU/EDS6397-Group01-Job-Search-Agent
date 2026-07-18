"""Focused tests for the exact five model-callable schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agent.tool_registry import (
    AssignmentToolRegistry,
    FilteringCallArguments,
    ToolRegistryError,
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


@pytest.mark.parametrize(
    "tool_name",
    [
        "filter_jobs",
        "score_jobs",
        "analyze_fit",
        "tailor_resume",
        "generate_cover_letter",
    ],
)
def test_registry_can_expose_exactly_one_phase_specific_schema(tool_name):
    schemas = AssignmentToolRegistry.model_schemas([tool_name])
    assert [item["function"]["name"] for item in schemas] == [tool_name]


def test_registry_rejects_unknown_schema_subset_without_changing_complete_registry():
    with pytest.raises(ToolRegistryError, match="Unknown schema"):
        AssignmentToolRegistry.model_schemas(["not-a-tool"])
    assert len(AssignmentToolRegistry.model_schemas()) == 5
