"""Focused integration tests for the one continuous runtime loop."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.agent.client import (
    ChatModelTransportError,
    NormalizedAssistantMessage,
    NormalizedToolCall,
)
from src.agent.runtime import AgentLoopLimitError, JobSearchAgentRuntime
from src.agent.state import AgentPhase, StateInvariantError
from src.agent.tool_registry import ToolArgumentsError
from src.config import AppConfig
from src.observability.tracing import LangfuseAgentTracer, NoOpAgentTracer
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
from tests.test_tracing import FakeLangfuse

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
        if messages:
            content = messages[-1].get("content", "")
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("type") == "workflow_checkpoint":
                contract = payload.get("next_action_contract", {})
                if contract.get("phase") == "resume_revision":
                    target_context = payload.get("target_context") or contract.get(
                        "target_context"
                    )
                    if isinstance(target_context, dict):
                        memory_facts = target_context.get("current_memory_facts", [])
                        self.revision_saw_memory = any(
                            "Kubernetes" in str(fact.get("normalized_value", ""))
                            or "Kubernetes" in str(fact.get("statement", ""))
                            for fact in memory_facts
                            if isinstance(fact, dict)
                        )
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


def _responses(
    scores,
    resume_plans,
    cover_plans,
    *,
    malformed_tailoring=False,
):
    ids = [item.job_id for item in scores]
    responses = [
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
    ]
    responses.extend(
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
            ]
        )
        for index, job_id in enumerate(ids)
    )
    if malformed_tailoring:
        responses.extend(
            [
                NormalizedAssistantMessage(
                    tool_calls=[
                        _tool(
                            "tailor_resume",
                            {
                                "decision_summary": "Use an incorrect generic plan.",
                                "job_id": ids[0],
                                "edit_plan": {
                                    "education": [],
                                    "experience": [],
                                    "projects": [],
                                    "skills": [],
                                },
                            },
                            20,
                        )
                    ]
                ),
                NormalizedAssistantMessage(
                    tool_calls=[
                        _tool(
                            "tailor_resume",
                            {
                                "decision_summary": "Use incorrect flattened fields.",
                                "job_id": ids[0],
                                "edit_plan": {
                                    "job_id": ids[0],
                                    "professional_summary": resume_plans[
                                        ids[0]
                                    ].professional_summary.model_dump(mode="json"),
                                    "project_swap": resume_plans[
                                        ids[1]
                                    ].project_swap.model_dump(mode="json"),
                                },
                                "experience_bullet_edits": [
                                    item.model_dump(mode="json")
                                    for item in resume_plans[
                                        ids[0]
                                    ].experience_bullet_edits
                                ],
                                "plan_rationale": "Incorrectly flattened.",
                                "project_swap": None,
                            },
                            21,
                        )
                    ]
                ),
            ]
        )
    responses.extend(
        NormalizedAssistantMessage(
            tool_calls=[
                _tool(
                    "tailor_resume",
                    {
                        "job_id": job_id,
                        "edit_plan": resume_plans[job_id].model_dump(mode="json"),
                        "decision_summary": (
                            f"Create revision-zero draft for {job_id}."
                        ),
                    },
                    30 + index,
                )
            ]
        )
        for index, job_id in enumerate(ids)
    )
    responses.append(
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
                    40,
                )
            ]
        )
    )
    responses.extend(
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
                    50 + index,
                )
            ]
        )
        for index, job_id in enumerate(ids)
    )
    return responses


def _runtime(
    tmp_path,
    client,
    provider,
    tracer,
    *,
    config=None,
    progress_callback=None,
):
    memory_path = tmp_path / "memory.json"
    shutil.copyfile(ROOT / "memory.json", memory_path)
    return JobSearchAgentRuntime(
        client=client,
        review_decision_provider=provider,
        config=config or AppConfig(langfuse_enabled=False),
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
        progress_callback=progress_callback,
    )


def _runtime_at_tailoring(tmp_path):
    tracer = NoOpAgentTracer()
    runtime = _runtime(
        tmp_path,
        ScriptedClient([]),
        ReviewProvider("unused"),
        tracer,
    )
    runtime.trace = tracer.start_trace("agent_run", run_id=runtime.run_id)
    runtime._load_inputs()
    assert runtime.registry is not None
    assert runtime.state is not None
    runtime.registry.execute(
        "filter_jobs",
        {"decision_summary": "Filter once."},
    )
    runtime.registry.execute(
        "score_jobs",
        {"decision_summary": "Score deterministically."},
    )
    for job_id in runtime.state.top_3_job_ids:
        runtime.registry.execute(
            "analyze_fit",
            {
                "decision_summary": f"Analyze {job_id}.",
                "job_id": job_id,
            },
        )
    assert runtime.state.phase == AgentPhase.TAILORING
    return runtime


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
            malformed_tailoring=True,
        )
    )
    provider = ReviewProvider(scores[0].job_id)
    fake_langfuse = FakeLangfuse()
    tracer = LangfuseAgentTracer(
        AppConfig(langfuse_enabled=True, langfuse_public_trace=True),
        client=fake_langfuse,
    )
    progress = []
    runtime = _runtime(
        tmp_path,
        client,
        provider,
        tracer,
        progress_callback=progress.append,
    )
    result = runtime.run()

    assert result.completed is True
    assert result.model_call_count == 14
    assert result.tool_call_count == 12
    assert result.invalid_tool_attempt_count == 2
    assert result.fit_analysis_count == 3
    assert result.draft_resume_count == 3
    assert result.pause_count == 1
    assert result.finalized_resume_count == 3
    assert result.cover_letter_count == 3
    assert result.trace_url == f"https://trace.local/{result.trace_id}"
    assert provider.calls == 2
    assert client.revision_saw_memory
    assert {call["client_id"] for call in client.calls} == {id(client)}
    assert [call["tool_names"] for call in client.calls] == [
        ["filter_jobs"],
        ["score_jobs"],
        ["analyze_fit"],
        ["analyze_fit"],
        ["analyze_fit"],
        ["tailor_resume"],
        ["tailor_resume"],
        ["tailor_resume"],
        ["tailor_resume"],
        ["tailor_resume"],
        ["tailor_resume"],
        ["generate_cover_letter"],
        ["generate_cover_letter"],
        ["generate_cover_letter"],
    ]
    message_counts = [call["message_count"] for call in client.calls]
    assert message_counts[:5] == sorted(message_counts[:5])
    assert message_counts[5] == 2
    assert all(count <= 14 for count in message_counts)
    assert runtime.state is not None
    assert runtime.state.consecutive_invalid_call_count == 0
    first_recovery = json.loads(client.calls[6]["messages"][-1]["content"])
    second_recovery = json.loads(client.calls[7]["messages"][-1]["content"])
    assert first_recovery["allowed_tool"] == "tailor_resume"
    assert first_recovery["target_job_id"] == scores[0].job_id
    assert set(first_recovery["field_diagnostics"]["replacement_schema_keys"]) == {
        "education",
        "experience",
        "projects",
        "skills",
    }
    assert "outer.job_id" not in first_recovery["field_diagnostics"]["missing_fields"]
    assert set(second_recovery["field_diagnostics"]["misplaced_fields"]) == {
        "experience_bullet_edits",
        "plan_rationale",
        "project_swap",
    }
    assert "project_swap must be null" in second_recovery["error"]
    serialized_recovery = json.dumps(first_recovery)
    assert "Outer job_id and edit_plan.job_id" in serialized_recovery
    assert first_recovery["target_context"]["project_swap"] is None
    contracts = []
    for call in client.calls:
        contract = None
        for message in reversed(call["messages"]):
            if message.get("role") != "user":
                continue
            payload = json.loads(message["content"])
            if "next_action_contract" in payload:
                contract = payload["next_action_contract"]
                break
        assert contract is not None
        contracts.append(contract)
    assert [
        (item["allowed_tool"], item["target_job_id"])
        for item in contracts
    ] == [
        ("filter_jobs", None),
        ("score_jobs", None),
        ("analyze_fit", scores[0].job_id),
        ("analyze_fit", scores[1].job_id),
        ("analyze_fit", scores[2].job_id),
        ("tailor_resume", scores[0].job_id),
        ("tailor_resume", scores[0].job_id),
        ("tailor_resume", scores[0].job_id),
        ("tailor_resume", scores[1].job_id),
        ("tailor_resume", scores[2].job_id),
        ("tailor_resume", scores[0].job_id),
        ("generate_cover_letter", scores[0].job_id),
        ("generate_cover_letter", scores[1].job_id),
        ("generate_cover_letter", scores[2].job_id),
    ]
    first_tailoring_context = contracts[5]["target_context"]
    assert first_tailoring_context["target_job_id"] == scores[0].job_id
    assert first_tailoring_context["rank"] == 1
    assert first_tailoring_context["project_swap"] is None
    assert scores[1].job_id not in json.dumps(first_tailoring_context)
    assert scores[2].job_id not in json.dumps(first_tailoring_context)
    first_tailoring_messages = client.calls[5]["messages"]
    assert first_tailoring_messages[0]["role"] == "system"
    tailoring_checkpoint = json.loads(first_tailoring_messages[1]["content"])
    assert tailoring_checkpoint["type"] == "workflow_checkpoint"
    assert "current_state" not in json.dumps(first_tailoring_messages)
    assert "exact_tailor_resume_structural_template" not in json.dumps(
        tailoring_checkpoint
    )
    assert len(json.dumps(first_tailoring_messages)) < len(
        json.dumps(client.calls[4]["messages"])
    )
    assert set(runtime.state.fit_analyses) == {score.job_id for score in scores}
    assert runtime.registry is not None
    assert len(runtime.registry.model_schemas()) == 5
    assert len({id(runtime)}) == 1
    assert len({call["client_id"] for call in client.calls}) == 1
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
    assert fake_langfuse.root.public_calls == 1
    assert fake_langfuse.trace_url_arguments == [result.trace_id]
    assert len({item.trace_id for item in tracer.traces[0].observations}) == 1
    assert tracer.traces[0].trace_public is True
    assert tracer.traces[0].flushed is True
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
    assert progress[:3] == [
        "Agent phase: filtering",
        "Model call 1/40: waiting for scripted-local-model",
        "Model call 1: response received",
    ]
    assert "Agent phase: fit analysis 1/3" in progress
    assert "Agent phase: resume tailoring 1/3" in progress
    assert "Agent phase: cover letter 1/3" in progress
    safe_progress = "\n".join(progress)
    for sensitive in (
        scores[0].job_id,
        "decision_summary",
        "candidate evidence",
        "PRIVATE_SECRET",
    ):
        assert sensitive not in safe_progress


def test_deterministic_targets_and_target_bound_tailoring_contract(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    assert runtime.state is not None
    ids = runtime.state.top_3_job_ids

    first_contract = runtime._next_action_contract()
    assert first_contract["allowed_tool"] == "tailor_resume"
    assert first_contract["target_job_id"] == ids[0]
    assert first_contract["target_rank"] == 1
    assert first_contract["initial_draft"] is True
    assert first_contract["target_context"]["project_swap"] is None
    assert first_contract["required_argument_shape"]["job_id"] == ids[0]
    assert (
        first_contract["required_argument_shape"]["edit_plan"]["job_id"]
        == ids[0]
    )
    assert len(
        first_contract["required_argument_shape"]["edit_plan"][
            "experience_bullet_edits"
        ]
    ) == 2
    assert "exact_tailor_resume_structural_template" not in first_contract
    plan_limits = " ".join(first_contract["constraints"])
    assert "at most 55 words" in plan_limits
    assert "at most 32 words" in plan_limits
    assert "Exactly two different experience_bullet_edits" in plan_limits
    safety_contract = " ".join(first_contract["constraints"])
    assert "job_posting citation supports relevance only" in safety_contract
    assert "candidate-side evidence" in safety_contract
    assert "genuine-gap skill" in safety_contract
    assert "Never invent patient data" in safety_contract
    assert "Protected resume regions" in safety_contract

    complete_analyses = dict(runtime.state.fit_analyses)
    runtime.state.fit_analyses = {
        ids[1]: complete_analyses[ids[1]],
        ids[2]: complete_analyses[ids[2]],
    }
    runtime.state.phase = AgentPhase.FIT_ANALYSIS
    fit_contract = runtime._next_action_contract()
    assert fit_contract["allowed_tool"] == "analyze_fit"
    assert fit_contract["target_job_id"] == ids[0]

    runtime.state.fit_analyses = complete_analyses
    runtime.state.phase = AgentPhase.TAILORING
    runtime.state.draft_resumes = {ids[0]: object()}  # type: ignore[dict-item]
    next_draft_contract = runtime._next_action_contract()
    assert next_draft_contract["target_job_id"] == ids[1]

    runtime.state.draft_resumes = {}
    revision_contract = runtime._next_action_contract(
        revision_job_id=ids[2],
        revision_round=1,
        revision_feedback="Keep the revision concise.",
    )
    assert revision_contract["allowed_tool"] == "tailor_resume"
    assert revision_contract["target_job_id"] == ids[2]
    assert revision_contract["initial_draft"] is False
    assert revision_contract["target_context"]["revision_feedback"] == (
        "Keep the revision concise."
    )


def test_tailor_resume_malformed_nesting_targets_and_cross_job_swap_are_rejected(
    tmp_path,
):
    runtime = _runtime_at_tailoring(tmp_path)
    assert runtime.state is not None
    assert runtime.registry is not None
    ids = runtime.state.top_3_job_ids
    contract = runtime._next_action_contract()
    job = runtime.registry._job(ids[0])
    valid_plan = valid_resume_plan(
        job,
        runtime.state.fit_analyses[ids[0]],
        runtime.registry.bundle,
    ).model_dump(mode="json")
    valid_arguments = {
        "decision_summary": "Create the first deterministic draft.",
        "job_id": ids[0],
        "edit_plan": valid_plan,
    }
    valid_call = _tool("tailor_resume", valid_arguments, 1)
    runtime._validate_call_for_contract(valid_call, contract)
    parsed = runtime.registry.parse_arguments("tailor_resume", valid_arguments)
    assert parsed.job_id == ids[0]
    assert parsed.edit_plan.job_id == ids[0]

    missing_outer = _tool(
        "tailor_resume",
        {
            "decision_summary": "Omit the outer target.",
            "edit_plan": valid_plan,
        },
        2,
    )
    with pytest.raises(StateInvariantError, match="Outer job_id"):
        runtime._validate_call_for_contract(missing_outer, contract)
    assert "outer.job_id" in runtime._argument_diagnostics(
        missing_outer,
        contract,
    )["missing_fields"]

    mismatched_outer = _tool(
        "tailor_resume",
        {**valid_arguments, "job_id": ids[1]},
        3,
    )
    with pytest.raises(StateInvariantError, match="Outer job_id"):
        runtime._validate_call_for_contract(mismatched_outer, contract)

    mismatched_nested = _tool(
        "tailor_resume",
        {
            **valid_arguments,
            "edit_plan": {**valid_plan, "job_id": ids[1]},
        },
        4,
    )
    with pytest.raises(StateInvariantError, match="edit_plan.job_id"):
        runtime._validate_call_for_contract(mismatched_nested, contract)

    generic_arguments = {
        "decision_summary": "Use generic keys.",
        "job_id": ids[0],
        "edit_plan": {
            "education": [],
            "experience": [],
            "projects": [],
            "skills": [],
        },
    }
    with pytest.raises(ToolArgumentsError, match="professional_summary"):
        runtime.registry.parse_arguments("tailor_resume", generic_arguments)
    generic_diagnostics = runtime._argument_diagnostics(
        _tool("tailor_resume", generic_arguments, 7),
        contract,
    )
    assert set(generic_diagnostics["replacement_schema_keys"]) == {
        "education",
        "experience",
        "projects",
        "skills",
    }

    flattened_arguments = {
        "decision_summary": "Flatten plan fields.",
        "job_id": ids[0],
        "edit_plan": {
            "job_id": ids[0],
            "professional_summary": valid_plan["professional_summary"],
        },
        "experience_bullet_edits": valid_plan["experience_bullet_edits"],
        "project_swap": None,
        "plan_rationale": "Misplaced.",
    }
    with pytest.raises(ToolArgumentsError, match="extra_forbidden"):
        runtime.registry.parse_arguments("tailor_resume", flattened_arguments)
    assert set(
        runtime._argument_diagnostics(
            _tool("tailor_resume", flattened_arguments, 8),
            contract,
        )["misplaced_fields"]
    ) == {"experience_bullet_edits", "plan_rationale", "project_swap"}

    later_swap = runtime.state.fit_analyses[ids[1]].projects.swap_suggestion
    assert later_swap is not None
    contaminated = _tool(
        "tailor_resume",
        {
            **valid_arguments,
            "edit_plan": {
                **valid_plan,
                "project_swap": {
                    "remove_project_id": later_swap.remove_project_id,
                    "add_project_id": later_swap.add_project_id,
                    "reason": later_swap.reason,
                    "citations": [
                        item.model_dump(mode="json")
                        for item in later_swap.citations
                    ],
                },
            },
        },
        5,
    )
    with pytest.raises(StateInvariantError, match="must be null"):
        runtime._validate_call_for_contract(contaminated, contract)

    wrong_job = _tool(
        "tailor_resume",
        {**valid_arguments, "job_id": ids[1]},
        6,
    )
    with pytest.raises(StateInvariantError, match=ids[0]):
        runtime._validate_call_for_contract(wrong_job, contract)


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
    fake_langfuse = FakeLangfuse()
    tracer = LangfuseAgentTracer(
        AppConfig(langfuse_enabled=True, langfuse_public_trace=True),
        client=fake_langfuse,
    )
    runtime = _runtime(
        tmp_path,
        client,
        ReviewProvider("unused"),
        tracer,
    )
    with pytest.raises(AgentLoopLimitError, match="consecutive invalid"):
        runtime.run()
    assert client.calls == 3
    assert runtime.state is not None
    assert len(runtime.state.invalid_tool_attempts) == 3
    assert fake_langfuse.root.ended == 1
    assert fake_langfuse.root.public_calls == 0
    assert fake_langfuse.trace_url_arguments == []
    assert fake_langfuse.flushes == 1
    assert runtime.trace is not None
    assert runtime.trace.trace_url is None


def test_model_transport_failure_flushes_private_trace_without_outputs_or_memory_write(
    tmp_path,
):
    class TransportFailureClient:
        model_name = "qwen3:8b"

        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools):
            del messages, tools
            self.calls += 1
            raise ChatModelTransportError(
                "Local Ollama chat request failed: ReadTimeout"
            )

    source_memory_before = _hash(ROOT / "memory.json")
    fake_langfuse = FakeLangfuse()
    config = AppConfig(
        langfuse_enabled=True,
        langfuse_public_trace=True,
    )
    tracer = LangfuseAgentTracer(config, client=fake_langfuse)
    client = TransportFailureClient()
    progress = []
    runtime = _runtime(
        tmp_path,
        client,
        ReviewProvider("unused"),
        tracer,
        config=config,
        progress_callback=progress.append,
    )
    copied_memory_before = _hash(runtime.memory_path)

    result = runtime.run()

    assert result.completed is False
    assert result.failure_reason == (
        "ChatModelTransportError: "
        "Local Ollama chat request failed: ReadTimeout"
    )
    assert client.calls == 1
    assert result.model_call_count == 0
    assert _hash(runtime.memory_path) == copied_memory_before
    assert _hash(ROOT / "memory.json") == source_memory_before
    assert runtime.final_output_root.is_dir()
    assert not any(runtime.final_output_root.rglob("*"))
    assert runtime.run_workspace.is_dir()
    assert not any(runtime.run_workspace.rglob("*"))
    assert fake_langfuse.root.ended == 1
    assert fake_langfuse.root.public_calls == 0
    assert fake_langfuse.trace_url_arguments == []
    assert fake_langfuse.flushes == 1
    assert runtime.trace is not None
    assert runtime.trace.trace_url is None
    assert runtime.trace.trace_public is False
    assert progress == [
        "Agent phase: filtering",
        "Model call 1/40: waiting for qwen3:8b",
    ]


def test_tailoring_compaction_keeps_target_context_and_drops_obsolete_history(
    tmp_path,
):
    runtime = _runtime_at_tailoring(tmp_path)
    assert runtime.state is not None
    ids = runtime.state.top_3_job_ids
    for index in range(6):
        runtime.conversation.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "type": "current_state",
                        "state": runtime.state.snapshot(),
                        "obsolete": "x" * 4000,
                    }
                ),
            }
        )
        runtime.conversation.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [],
            }
        )
    full_size = len(json.dumps(runtime.conversation))
    contract = runtime._next_action_contract()
    runtime._apply_conversation_checkpoint(contract)
    compact_size = len(json.dumps(runtime.conversation))
    checkpoint = json.loads(runtime.conversation[1]["content"])

    assert runtime.conversation[0]["role"] == "system"
    assert len(runtime.conversation) == 2
    assert checkpoint["type"] == "workflow_checkpoint"
    assert checkpoint["target_context"]["target_job_id"] == ids[0]
    assert "editable_experience_bullets" in checkpoint["target_context"]
    assert ids[1] not in json.dumps(checkpoint["target_context"])
    assert "current_state" not in json.dumps(runtime.conversation)
    assert compact_size < full_size * 0.5


def test_exact_structural_template_is_recovery_only(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    assert runtime.state is not None
    ids = runtime.state.top_3_job_ids
    contract = runtime._next_action_contract()

    runtime._append_invalid_message(
        None,
        "Model response reached the generation limit before completing a tool call",
        contract,
    )
    length_payload = json.loads(runtime.conversation[-1]["content"])
    assert "exact_tailor_resume_structural_template" not in length_payload

    generic_call = _tool(
        "tailor_resume",
        {
            "decision_summary": "bad",
            "job_id": ids[0],
            "edit_plan": {
                "education": [],
                "experience": [],
                "projects": [],
                "skills": [],
            },
        },
        99,
    )
    runtime._append_invalid_message(generic_call, "Invalid shape", contract)
    recovery_payload = json.loads(runtime.conversation[-1]["content"])
    assert "exact_tailor_resume_structural_template" in recovery_payload
    template = recovery_payload["exact_tailor_resume_structural_template"]
    assert template["job_id"] == ids[0]
    assert template["edit_plan"]["job_id"] == ids[0]


def test_length_limited_empty_response_is_invalid_without_side_effects(
    tmp_path,
):
    runtime = _runtime_at_tailoring(tmp_path)
    assert runtime.state is not None
    contract = runtime._next_action_contract()
    memory_before = _hash(runtime.memory_path)
    progress: list[str] = []
    runtime.progress_callback = progress.append
    runtime.state.model_call_count = 6
    response = NormalizedAssistantMessage(
        content="",
        tool_calls=[],
        done_reason="length",
        eval_count=2048,
    )

    tool_history_before = len(runtime.state.tool_execution_history)
    invalid_before = len(runtime.state.invalid_tool_attempts)

    valid, invalid = runtime._execute_response_calls(response, contract)

    assert valid == 0
    assert invalid == 1
    assert len(runtime.state.draft_resumes) == 0
    assert len(runtime.state.tool_execution_history) == tool_history_before
    assert len(runtime.state.invalid_tool_attempts) == invalid_before + 1
    assert _hash(runtime.memory_path) == memory_before
    assert runtime.state.invalid_tool_attempts[-1].error.startswith(
        "Model response reached the generation limit"
    )
    assert "Model call 6: no complete tool call returned" in progress
    assert "Model completion: length limit" in progress
    assert "decision_summary" not in "\n".join(progress)


def test_cover_letter_contract_includes_concise_limits(tmp_path):
    from types import SimpleNamespace

    runtime = _runtime_at_tailoring(tmp_path)
    assert runtime.state is not None
    runtime.state.human_review = SimpleNamespace(completed=True)
    runtime.state.finalized_resumes = {
        job_id: SimpleNamespace(approved_revision_round=0)
        for job_id in runtime.state.top_3_job_ids
    }
    contract = runtime._next_action_contract()
    assert contract["allowed_tool"] == "generate_cover_letter"
    limits = " ".join(contract["constraints"])
    assert "at most 90 words" in limits
    assert "at most 15 words" in limits
    assert contract["target_context"]["approved_resume_revision"] == 0
