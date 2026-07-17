"""Focused integration tests for the one continuous runtime loop."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.agent.client import NormalizedAssistantMessage, NormalizedToolCall
from src.agent.runtime import AgentLoopLimitError, JobSearchAgentRuntime
from src.config import AppConfig
from src.observability.tracing import NoOpAgentTracer
from src.services.candidate_loader import load_candidate_bundle
from src.services.jobs_loader import load_jobs
from src.services.memory_loader import load_memory
from src.tools.filtering import filtering_tool
from src.tools.fit_analysis import fit_analysis_tool
from src.tools.scoring import scoring_tool
from src.workflow.human_review import (
    ResumeReviewDecision,
    ReviewDecisionType,
    ReviewFactInput,
    ReviewFactType,
)
from tests.test_cover_letter_tool import (
    _fake_compiler as fake_cover_compiler,
    _valid_plan as valid_cover_plan,
)
from tests.test_resume_tailoring_tool import (
    _fake_compiler as fake_resume_compiler,
    _valid_plan as valid_resume_plan,
)

ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tool(name: str, arguments: dict, index: int) -> NormalizedToolCall:
    return NormalizedToolCall(id=f"call-{index}", name=name, arguments=arguments)


class ScriptedClient:
    model_name = "scripted-local-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.object_identity = id(self)
        self.revision_saw_memory = False

    def chat(self, messages, tools):
        self.calls.append(
            {
                "message_count": len(messages),
                "messages": list(messages),
                "tool_names": [item["function"]["name"] for item in tools],
                "client_id": id(self),
            }
        )
        if messages and "resume_revision_request" in messages[-1].get("content", ""):
            self.revision_saw_memory = "Kubernetes" in messages[-1]["content"]
        if not self.responses:
            raise AssertionError("Scripted client exhausted")
        return self.responses.pop(0)


class ReviewProvider:
    def __init__(self, rejected_job_id: str):
        self.rejected_job_id = rejected_job_id
        self.calls = 0

    def __call__(self, pending_drafts, session_state):
        self.calls += 1
        decisions = []
        for draft in pending_drafts:
            if self.calls == 1 and draft.job_id == self.rejected_job_id:
                decisions.append(
                    ResumeReviewDecision(
                        job_id=draft.job_id,
                        decision=ReviewDecisionType.REJECT,
                        comments="Surface the newly confirmed Kubernetes skill.",
                        learned_facts=[
                            ReviewFactInput(
                                fact_type=ReviewFactType.SKILL,
                                statement="The candidate has Kubernetes experience.",
                                normalized_value="Kubernetes",
                                skill_tags=["Kubernetes"],
                            )
                        ],
                    )
                )
            else:
                decisions.append(
                    ResumeReviewDecision(
                        job_id=draft.job_id,
                        decision=ReviewDecisionType.APPROVE,
                    )
                )
        return decisions


def _workflow_plans():
    bundle = load_candidate_bundle(
        ROOT / "candidate/profile.json",
        ROOT / "candidate/portfolio.json",
        ROOT / "candidate/evidence_registry.json",
    )
    memory = load_memory(ROOT / "memory.json", bundle.profile.candidate_id)
    jobs = load_jobs(ROOT / "data/AI_ML_Jobs_Dataset_20.csv")
    accepted = filtering_tool(jobs, bundle.profile).accepted_jobs
    scores = scoring_tool(accepted, bundle, memory).top_3
    jobs_by_id = {job.job_id: job for job in accepted}
    analyses = {
        score.job_id: fit_analysis_tool(
            jobs_by_id[score.job_id],
            score,
            bundle,
            memory,
            ROOT / "candidate/sample_resume.tex",
        )
        for score in scores
    }
    resume_plans = {
        job_id: valid_resume_plan(jobs_by_id[job_id], analyses[job_id], bundle)
        for job_id in analyses
    }
    for plan in resume_plans.values():
        plan.experience_bullet_edits[0].new_text = (
            "Built Python and scikit-learn risk models over SQL datasets, raising "
            "validated recall by 14% while holding the false-positive rate constant."
        )
    cover_plans = {
        job_id: valid_cover_plan(jobs_by_id[job_id]) for job_id in analyses
    }
    return bundle, scores, resume_plans, cover_plans


def _responses(scores, resume_plans, cover_plans, *, early_invalid=False):
    ids = [item.job_id for item in scores]
    responses = []
    if early_invalid:
        responses.append(
            NormalizedAssistantMessage(
                tool_calls=[
                    _tool(
                        "score_jobs",
                        {"decision_summary": "Score before filtering."},
                        0,
                    )
                ]
            )
        )
    responses.extend(
        [
            NormalizedAssistantMessage(
                tool_calls=[
                    _tool(
                        "filter_jobs",
                        {"decision_summary": "Filter all loaded jobs exactly once."},
                        1,
                    )
                ]
            ),
            NormalizedAssistantMessage(
                tool_calls=[
                    _tool(
                        "score_jobs",
                        {
                            "decision_summary": (
                                "Ask Python to score only accepted jobs and select Top 3."
                            )
                        },
                        2,
                    )
                ]
            ),
            NormalizedAssistantMessage(
                tool_calls=[
                    _tool(
                        "analyze_fit",
                        {
                            "job_id": job_id,
                            "decision_summary": f"Analyze Top 3 job {job_id}.",
                        },
                        10 + index,
                    )
                    for index, job_id in enumerate([ids[2], ids[0], ids[1]])
                ]
            ),
            NormalizedAssistantMessage(
                tool_calls=[
                    _tool(
                        "tailor_resume",
                        {
                            "job_id": job_id,
                            "edit_plan": resume_plans[job_id].model_dump(mode="json"),
                            "decision_summary": f"Create revision-zero draft for {job_id}.",
                        },
                        20 + index,
                    )
                    for index, job_id in enumerate([ids[1], ids[2], ids[0]])
                ]
            ),
            NormalizedAssistantMessage(
                tool_calls=[
                    _tool(
                        "tailor_resume",
                        {
                            "job_id": ids[0],
                            "edit_plan": resume_plans[ids[0]].model_dump(mode="json"),
                            "decision_summary": (
                                f"Revise rejected resume {ids[0]} using review feedback."
                            ),
                        },
                        30,
                    )
                ]
            ),
            NormalizedAssistantMessage(
                tool_calls=[
                    _tool(
                        "generate_cover_letter",
                        {
                            "job_id": job_id,
                            "plan": cover_plans[job_id].model_dump(mode="json"),
                            "decision_summary": (
                                f"Generate the approved cover letter for {job_id}."
                            ),
                        },
                        40 + index,
                    )
                    for index, job_id in enumerate([ids[1], ids[0], ids[2]])
                ]
            ),
        ]
    )
    return responses


def _runtime(tmp_path, client, provider, tracer):
    memory_path = tmp_path / "memory.json"
    shutil.copyfile(ROOT / "memory.json", memory_path)
    return JobSearchAgentRuntime(
        client=client,
        review_decision_provider=provider,
        config=AppConfig(langfuse_enabled=False),
        jobs_path=ROOT / "data/AI_ML_Jobs_Dataset_20.csv",
        profile_path=ROOT / "candidate/profile.json",
        portfolio_path=ROOT / "candidate/portfolio.json",
        evidence_path=ROOT / "candidate/evidence_registry.json",
        memory_path=memory_path,
        base_resume_tex_path=ROOT / "candidate/sample_resume.tex",
        base_resume_pdf_path=ROOT / "candidate/sample_resume.pdf",
        run_workspace=tmp_path / "workspace",
        final_output_root=tmp_path / "final",
        tracer=tracer,
        run_id="scripted-complete-run",
    )


def test_full_actual_tool_run_recovers_and_uses_same_client(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.tools.resume_tailoring.compile_resume_pdf", fake_resume_compiler
    )
    monkeypatch.setattr(
        "src.tools.cover_letter.compile_cover_letter_pdf", fake_cover_compiler
    )
    protected = [
        ROOT / "memory.json",
        ROOT / "candidate/profile.json",
        ROOT / "candidate/portfolio.json",
        ROOT / "candidate/evidence_registry.json",
    ]
    before = {path: _hash(path) for path in protected}
    outputs_before = sorted(
        str(path.relative_to(ROOT)) for path in (ROOT / "outputs").rglob("*")
    ) if (ROOT / "outputs").exists() else []

    _, scores, resume_plans, cover_plans = _workflow_plans()
    client = ScriptedClient(
        _responses(
            scores,
            resume_plans,
            cover_plans,
            early_invalid=True,
        )
    )
    provider = ReviewProvider(scores[0].job_id)
    tracer = NoOpAgentTracer()
    runtime = _runtime(tmp_path, client, provider, tracer)
    result = runtime.run()

    assert result.completed is True
    assert result.model_call_count == 7
    assert result.tool_call_count == 12
    assert result.invalid_tool_attempt_count == 1
    assert result.fit_analysis_count == 3
    assert result.draft_resume_count == 3
    assert result.pause_count == 1
    assert result.finalized_resume_count == 3
    assert result.cover_letter_count == 3
    assert provider.calls == 2
    assert client.revision_saw_memory
    assert {call["client_id"] for call in client.calls} == {id(client)}
    assert all(
        call["tool_names"]
        == [
            "filter_jobs",
            "score_jobs",
            "analyze_fit",
            "tailor_resume",
            "generate_cover_letter",
        ]
        for call in client.calls
    )
    assert [call["message_count"] for call in client.calls] == sorted(
        call["message_count"] for call in client.calls
    )
    assert set(record.tool_name for record in result.tool_execution_records) == {
        "filter_jobs",
        "score_jobs",
        "analyze_fit",
        "tailor_resume",
        "generate_cover_letter",
    }
    required = {
        "job_details.json",
        "fit_analysis.txt",
        "fit_analysis.json",
        "resume_before.pdf",
        "resume_after.tex",
        "resume_after.pdf",
        "resume_change_log.json",
        "cover_letter.tex",
        "cover_letter.pdf",
        "cover_letter_evidence.json",
    }
    assert len(result.output_folders) == 3
    for folder in result.output_folders.values():
        assert required.issubset({path.name for path in folder.iterdir()})
    assert len(tracer.traces) == 1
    assert tracer.flush_count == 1
    names = [item.name for item in tracer.traces[0].observations]
    assert "human_review_pause" in names
    assert "memory_write" in names
    assert "resume_finalization" in names
    assert "output_packaging" in names
    assert {path: _hash(path) for path in protected} == before
    outputs_after = sorted(
        str(path.relative_to(ROOT)) for path in (ROOT / "outputs").rglob("*")
    ) if (ROOT / "outputs").exists() else []
    assert outputs_after == outputs_before


class AlwaysInvalidClient:
    model_name = "scripted-invalid"

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        return NormalizedAssistantMessage(
            tool_calls=[
                _tool(
                    "score_jobs",
                    {"decision_summary": "Repeat invalid early scoring."},
                    self.calls,
                )
            ]
        )


def test_three_repeated_invalid_turns_raise_loop_limit(tmp_path):
    _, scores, _, _ = _workflow_plans()
    del scores
    client = AlwaysInvalidClient()
    runtime = _runtime(
        tmp_path,
        client,
        ReviewProvider("unused"),
        NoOpAgentTracer(),
    )
    with pytest.raises(AgentLoopLimitError, match="consecutive invalid"):
        runtime.run()
    assert client.calls == 3
    assert runtime.state is not None
    assert len(runtime.state.invalid_tool_attempts) == 3
