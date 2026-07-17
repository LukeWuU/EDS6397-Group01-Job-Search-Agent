"""Focused tests for the injectable console Human Review provider."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.workflow.console_review import (
    ConsoleReviewAbort,
    ConsoleReviewDecisionProvider,
)
from tests.test_human_review_workflow import _draft


def _state(call=1):
    return SimpleNamespace(
        session_id="review-session",
        provider_call_count=call,
        pause_count=1,
    )


def _input(values, outputs=None, required_before_first=None):
    iterator = iter(values)
    calls = 0

    def read(prompt):
        nonlocal calls
        calls += 1
        if calls == 1 and outputs is not None and required_before_first is not None:
            combined = "\n".join(outputs)
            assert all(item in combined for item in required_before_first)
        return next(iterator)

    return read


def test_displays_all_initial_drafts_before_first_decision(tmp_path: Path):
    drafts = [_draft(tmp_path / job_id, job_id) for job_id in ("a", "b", "c")]
    outputs = []
    provider = ConsoleReviewDecisionProvider(
        input_fn=_input(
            ["a", "", "", "approve", "", "n", "A", "", "no"],
            outputs,
            ["Job ID: a", "Job ID: b", "Job ID: c"],
        ),
        output_fn=outputs.append,
    )
    decisions = provider(drafts, _state())
    assert [item.decision for item in decisions] == ["approve"] * 3
    assert provider.call_count == 1
    text = "\n".join(outputs)
    assert "Draft PDF:" in text
    assert "Change log:" in text
    assert "Before:" in text and "After:" in text
    assert "Evidence citations:" in text


def test_invalid_decision_reprompts_and_reject_requires_comments(tmp_path: Path):
    draft = _draft(tmp_path / "draft", "job-a")
    outputs = []
    provider = ConsoleReviewDecisionProvider(
        input_fn=_input(
            ["maybe", "r", "", "Please emphasize the API evidence.", "n"]
        ),
        output_fn=outputs.append,
    )
    decision = provider([draft], _state())[0]
    assert decision.decision == "reject"
    assert decision.comments == "Please emphasize the API evidence."
    assert "Enter approve/a or reject/r." in outputs
    assert "Reject decisions require nonempty comments." in outputs


def test_skill_candidate_fact_and_multiple_facts_are_parsed(tmp_path: Path):
    draft = _draft(tmp_path / "draft", "job-a")
    provider = ConsoleReviewDecisionProvider(
        input_fn=_input(
            [
                "approve",
                "Looks good.",
                "yes",
                "skill",
                "Candidate knows GraphQL.",
                "",
                "GraphQL, graphql, Python, Python",
                "yes",
                "candidate_fact",
                "Candidate mentors junior engineers.",
                "mentors junior engineers",
                "",
                "no",
            ]
        ),
        output_fn=lambda _: None,
    )
    decision = provider([draft], _state())[0]
    assert decision.comments == "Looks good."
    assert len(decision.learned_facts) == 2
    skill, fact = decision.learned_facts
    assert skill.fact_type == "skill"
    assert skill.skill_tags == ["GraphQL", "Python"]
    assert fact.fact_type == "candidate_fact"
    assert fact.statement == "Candidate mentors junior engineers."


def test_skill_without_value_or_tags_reprompts_fact(tmp_path: Path):
    draft = _draft(tmp_path / "draft", "job-a")
    outputs = []
    provider = ConsoleReviewDecisionProvider(
        input_fn=_input(
            [
                "a",
                "",
                "y",
                "skill",
                "Empty skill evidence.",
                "",
                "",
                "skill",
                "Candidate knows GraphQL.",
                "GraphQL",
                "",
                "n",
            ]
        ),
        output_fn=outputs.append,
    )
    decision = provider([draft], _state())[0]
    assert decision.learned_facts[0].normalized_value == "GraphQL"
    assert any("Invalid candidate fact" in line for line in outputs)


def test_revised_batch_uses_same_provider_and_session(tmp_path: Path):
    first = _draft(tmp_path / "r0", "job-a")
    revised = _draft(tmp_path / "r1", "job-a", 1, "Please revise.")
    provider = ConsoleReviewDecisionProvider(
        input_fn=_input(["r", "Please revise.", "n", "a", "", "n"]),
        output_fn=lambda _: None,
    )
    assert provider([first], _state(1))[0].decision == "reject"
    assert provider([revised], _state(2))[0].decision == "approve"
    assert provider.call_count == 2


@pytest.mark.parametrize(
    ("exception", "expected"),
    [(EOFError(), ConsoleReviewAbort), (KeyboardInterrupt(), KeyboardInterrupt)],
)
def test_eof_and_keyboard_interrupt_abort(tmp_path: Path, exception, expected):
    draft = _draft(tmp_path / "draft", "job-a")

    def abort(_prompt):
        raise exception

    provider = ConsoleReviewDecisionProvider(
        input_fn=abort,
        output_fn=lambda _: None,
    )
    with pytest.raises(expected):
        provider([draft], _state())
