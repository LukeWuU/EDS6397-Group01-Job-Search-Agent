"""Focused deterministic state-guard tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent.state import AgentPhase, AgentRunState, StateInvariantError


def _state() -> AgentRunState:
    return AgentRunState(
        run_id="run-test",
        candidate_id="candidate-1",
        loaded_memory_candidate_id="candidate-1",
    )


def test_prerequisites_duplicates_and_rejections_do_not_mutate_state():
    state = _state()
    before = state.snapshot()
    with pytest.raises(StateInvariantError, match="filtering"):
        state.validate_tool_call("score_jobs")
    with pytest.raises(StateInvariantError, match="scoring"):
        state.validate_tool_call("analyze_fit", job_id="job-1")
    assert state.snapshot() == before

    state.filtering_result = SimpleNamespace()  # type: ignore[assignment]
    state.scoring_result = SimpleNamespace()  # type: ignore[assignment]
    state.top_3_job_ids = ["job-1", "job-2", "job-3"]
    with pytest.raises(StateInvariantError, match="Top 3"):
        state.validate_tool_call("analyze_fit", job_id="job-4")
    state.fit_analyses = {"job-1": SimpleNamespace()}  # type: ignore[assignment]
    with pytest.raises(StateInvariantError, match="all three"):
        state.validate_tool_call("tailor_resume", job_id="job-2")
    assert list(state.fit_analyses) == ["job-1"]


def test_review_and_cover_letter_gates_and_duplicate_execution():
    state = _state()
    state.top_3_job_ids = ["a", "b", "c"]
    state.scoring_result = SimpleNamespace()  # type: ignore[assignment]
    state.fit_analyses = {
        key: SimpleNamespace(job_id=key) for key in state.top_3_job_ids
    }  # type: ignore[assignment]
    assert not state.can_start_human_review()
    with pytest.raises(StateInvariantError, match="Human Review"):
        state.validate_tool_call("generate_cover_letter", job_id="a")

    state.draft_resumes = {
        key: SimpleNamespace(job_id=key, revision_round=0)
        for key in state.top_3_job_ids
    }  # type: ignore[assignment]
    assert state.can_start_human_review()
    state.begin_human_review()
    assert state.phase == AgentPhase.HUMAN_REVIEW
    assert state.pause_count == 1
    with pytest.raises(StateInvariantError, match="already ran"):
        state.validate_tool_call("tailor_resume", job_id="a", revision_round=0)
