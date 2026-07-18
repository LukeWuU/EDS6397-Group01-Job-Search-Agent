"""Focused integration tests for the one continuous runtime loop."""

from __future__ import annotations

import hashlib
import json
import shutil
import copy
from pathlib import Path

import pytest

from src.agent.client import (
    ChatModelTransportError,
    NormalizedAssistantMessage,
    NormalizedToolCall,
)
from src.agent.runtime import (
    AgentLoopLimitError,
    JobSearchAgentRuntime,
    _TailorResumeTextDraftNoSwap,
    _TailorResumeTextDraftWithSwap,
)
from src.agent.prompts import TAILOR_RESUME_CONSTRAINTS, TAILOR_RESUME_PLAN_LIMITS
from src.agent.state import AgentPhase, StateInvariantError
from src.agent.tool_registry import AssignmentToolRegistry, ToolArgumentsError
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


def _compact_draft_from_plan(
    plan,
    job_id: str,
    *,
    decision_summary: str = "Create draft.",
):
    return {
        "decision_summary": decision_summary,
        "job_id": job_id,
        "professional_summary": {
            "new_text": plan.professional_summary.new_text,
            "reason": plan.professional_summary.reason,
        },
        "bullet_1": {
            "new_text": plan.experience_bullet_edits[0].new_text,
            "reason": plan.experience_bullet_edits[0].reason,
        },
        "bullet_2": {
            "new_text": plan.experience_bullet_edits[1].new_text,
            "reason": plan.experience_bullet_edits[1].reason,
        },
        "project_swap_reason": (
            plan.project_swap.reason if plan.project_swap is not None else None
        ),
        "plan_rationale": plan.plan_rationale,
    }


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
        valid_compact = _compact_draft_from_plan(
            resume_plans[ids[0]],
            ids[0],
            decision_summary="Use an incorrect generic plan.",
        )
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
                                **valid_compact,
                                "job_id": ids[1],
                                "decision_summary": "Use the wrong target job.",
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
                    _compact_draft_from_plan(
                        resume_plans[job_id],
                        job_id,
                        decision_summary=(
                            f"Create revision-zero draft for {job_id}."
                        ),
                    ),
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
                    _compact_draft_from_plan(
                        resume_plans[ids[0]],
                        ids[0],
                        decision_summary=(
                            f"Revise rejected resume {ids[0]} using review feedback."
                        ),
                    ),
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
    assert message_counts[6] == 3
    assert message_counts[7] == 3
    assert all(count <= 14 for count in message_counts)
    assert runtime.state is not None
    assert runtime.state.consecutive_invalid_call_count == 0
    first_recovery = json.loads(client.calls[6]["messages"][-1]["content"])
    second_recovery = json.loads(client.calls[7]["messages"][-1]["content"])
    assert first_recovery["type"] == "invalid_tool_call_recovery"
    assert first_recovery["allowed_tool"] == "tailor_resume"
    assert first_recovery["target_job_id"] == scores[0].job_id
    assert first_recovery["error_category"] in {"draft_schema", "validation"}
    assert "edit_plan" in first_recovery["field_diagnostics"]["extra_fields"]
    assert "outer.job_id" not in first_recovery["field_diagnostics"]["missing_fields"]
    assert second_recovery["error_category"] in {"hydration", "target_job"}
    assert "job_id mismatch" in second_recovery["error"] or "job_id" in second_recovery["error"].casefold()
    assert "edit_plan" not in first_recovery["required_argument_shape"]
    assert "citations" not in json.dumps(first_recovery["required_argument_shape"])
    assert "target_context" not in first_recovery
    assert "constraints" not in first_recovery
    assert "exact_tailor_resume_structural_template" not in first_recovery
    first_checkpoint = json.loads(client.calls[6]["messages"][1]["content"])
    assert first_checkpoint["target_context"]["project_swap_required"] is False
    contracts = []
    for call in client.calls:
        contracts.append(_checkpoint_contract_from_messages(call["messages"]))
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
    first_tailoring_messages = client.calls[5]["messages"]
    tailoring_checkpoint = json.loads(first_tailoring_messages[1]["content"])
    first_tailoring_context = tailoring_checkpoint["target_context"]
    assert first_tailoring_context["target_job_id"] == scores[0].job_id
    assert first_tailoring_context["rank"] == 1
    assert first_tailoring_context["project_swap_required"] is False
    assert "citation_contract" not in first_tailoring_context
    assert "bullet_1_source" in first_tailoring_context
    assert "36%" in json.dumps(first_tailoring_context["bullet_2_source"])
    assert "14%" in json.dumps(first_tailoring_context["bullet_1_source"])
    assert "target_context" not in tailoring_checkpoint["next_action_contract"]
    assert scores[1].job_id not in json.dumps(first_tailoring_context)
    assert scores[2].job_id not in json.dumps(first_tailoring_context)
    assert first_tailoring_messages[0]["role"] == "system"
    assert tailoring_checkpoint["type"] == "workflow_checkpoint"
    assert "current_state" not in json.dumps(first_tailoring_messages)
    assert "exact_tailor_resume_structural_template" not in json.dumps(
        tailoring_checkpoint
    )
    shape = contracts[5]["required_argument_shape"]
    assert "edit_plan" not in shape
    assert "bullet_1" in shape
    assert "bullet_2" in shape
    assert "experience_bullet_edits" not in shape
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
    assert first_contract["target_context"]["project_swap_required"] is False
    assert first_contract["required_argument_shape"]["job_id"] == ids[0]
    assert "edit_plan" not in first_contract["required_argument_shape"]
    assert "bullet_1" in first_contract["required_argument_shape"]
    assert "bullet_2" in first_contract["required_argument_shape"]
    assert first_contract["required_role_phrase"] == "AI engineer"
    assert first_contract["target_context"]["required_role_phrase"] == "AI engineer"
    assert "bullet_1_source" in first_contract["target_context"]
    assert "bullet_2_source" in first_contract["target_context"]
    schemas = runtime._model_schemas_for_contract(first_contract)
    assert schemas[0]["function"]["name"] == "tailor_resume"
    assert "ResumeEditPlan" not in json.dumps(schemas)
    plan_limits = " ".join(first_contract["constraints"])
    assert "Python supplies bullet IDs" in plan_limits
    assert "at most 55 words" in plan_limits
    assert "bullet_1 only from bullet_1_source" in plan_limits
    assert "Do not transfer metrics" in plan_limits
    safety_contract = " ".join(first_contract["constraints"])
    assert "genuine-gap" in safety_contract
    assert "project_swap_reason" in safety_contract

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


def test_tailor_resume_draft_schema_and_hydration_validation(tmp_path):
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
    )
    valid_draft = _compact_draft_from_plan(valid_plan, ids[0])
    raw_call = _tool("tailor_resume", valid_draft, 1)
    hydrated = runtime._hydrate_tailor_resume_call(raw_call, contract)
    runtime._validate_call_for_contract(hydrated, contract)
    parsed = runtime.registry.parse_arguments("tailor_resume", hydrated.arguments)
    assert parsed.job_id == ids[0]
    assert parsed.edit_plan.job_id == ids[0]
    assert raw_call.arguments == valid_draft

    with pytest.raises(ToolArgumentsError, match="draft schema rejection"):
        runtime._hydrate_tailor_resume_call(
            _tool(
                "tailor_resume",
                {
                    "decision_summary": "Missing required draft fields.",
                    "job_id": ids[0],
                    "edit_plan": {"education": []},
                },
                2,
            ),
            contract,
        )

    with pytest.raises(StateInvariantError, match="job_id mismatch"):
        runtime._hydrate_tailor_resume_call(
            _tool("tailor_resume", {**valid_draft, "job_id": ids[1]}, 3),
            contract,
        )

    extra_bullet_slot = {
        **valid_draft,
        "bullet_3": {"new_text": "extra", "reason": "extra"},
    }
    with pytest.raises(ToolArgumentsError, match="draft schema rejection"):
        runtime._hydrate_tailor_resume_call(
            _tool("tailor_resume", extra_bullet_slot, 4),
            contract,
        )

    generic_diagnostics = runtime._argument_diagnostics(
        _tool(
            "tailor_resume",
            {
                "decision_summary": "Use generic keys.",
                "job_id": ids[0],
                "edit_plan": {"education": []},
            },
            7,
        ),
        contract,
    )
    assert "edit_plan" in generic_diagnostics["extra_fields"]

    hydrated_swap = runtime._hydrate_tailor_resume_call(raw_call, contract)
    assert hydrated_swap.arguments["edit_plan"]["project_swap"] is None

    swap_job_id = next(
        job_id
        for job_id in ids
        if runtime.state.fit_analyses[job_id].projects.swap_suggestion is not None
    )
    if swap_job_id != ids[0]:
        swap_contract = runtime._next_action_contract()
        while swap_contract["target_job_id"] != swap_job_id:
            runtime.state.draft_resumes[swap_contract["target_job_id"]] = object()
            swap_contract = runtime._next_action_contract()
        swap_plan = valid_resume_plan(
            runtime.registry._job(swap_job_id),
            runtime.state.fit_analyses[swap_job_id],
            runtime.registry.bundle,
        )
        swap_draft = _compact_draft_from_plan(swap_plan, swap_job_id)
        swap_draft["project_swap_reason"] = None
        with pytest.raises(ToolArgumentsError, match="draft schema rejection"):
            runtime._hydrate_tailor_resume_call(
                _tool("tailor_resume", swap_draft, 8),
                swap_contract,
            )


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
    for index in range(8):
        runtime.conversation.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "type": "current_state",
                        "state": runtime.state.snapshot(),
                        "obsolete": "x" * 8000,
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
    assert "bullet_1_source" in checkpoint["target_context"]
    assert "bullet_2_source" in checkpoint["target_context"]
    assert ids[1] not in json.dumps(checkpoint["target_context"])
    assert "current_state" not in json.dumps(runtime.conversation)
    assert compact_size < full_size * 0.5


def test_exact_structural_template_is_recovery_only(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    assert runtime.state is not None
    ids = runtime.state.top_3_job_ids
    contract = runtime._next_action_contract()

    runtime._rebuild_bounded_recovery(
        contract,
        error="Model response reached the generation limit before completing a tool call",
        tool_call=None,
        length_limited=True,
    )
    length_payload = json.loads(runtime.conversation[-1]["content"])
    assert length_payload["type"] == "tool_call_retry"
    assert length_payload["reason"] == "generation_limit"
    assert "exact_tailor_resume_structural_template" not in length_payload
    assert len(runtime.conversation) == 3

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
    assert recovery_payload["type"] == "invalid_tool_call_recovery"
    assert recovery_payload["error_category"] == "draft_schema"
    assert "exact_tailor_resume_structural_template" not in recovery_payload
    assert "required_argument_shape" in recovery_payload
    shape = recovery_payload["required_argument_shape"]
    assert shape["job_id"] == ids[0]
    assert "edit_plan" not in shape
    assert "citations" not in json.dumps(shape)


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
    assert len(runtime.conversation) == 3
    assert runtime.conversation[0]["role"] == "system"
    retry = json.loads(runtime.conversation[-1]["content"])
    assert retry["type"] == "tool_call_retry"
    assert retry["reason"] == "generation_limit"
    assert "required_argument_shape" not in retry
    assert "invalid_tool_call" not in json.dumps(runtime.conversation)
    assert "Model call 6: no complete tool call returned" in progress
    assert "Model completion: length limit" in progress
    assert "decision_summary" not in "\n".join(progress)

    runtime.state.model_call_count = 7
    response_again = NormalizedAssistantMessage(content="", tool_calls=[])
    runtime._execute_response_calls(response_again, contract)
    assert len(runtime.conversation) == 3
    assert runtime._serialized_conversation_char_count() == len(
        json.dumps(runtime.conversation, separators=(",", ":"))
    )
    second_size = runtime._serialized_conversation_char_count()
    runtime.state.model_call_count = 8
    runtime._execute_response_calls(response_again, contract)
    assert len(runtime.conversation) == 3
    assert runtime._serialized_conversation_char_count() == second_size


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


def _checkpoint_contract_from_messages(messages):
    contract = None
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        payload = json.loads(message["content"])
        if payload.get("type") == "workflow_checkpoint":
            contract = dict(payload["next_action_contract"])
            if "target_context" in payload:
                contract["target_context"] = payload["target_context"]
            return contract
        if "next_action_contract" in payload:
            contract = payload["next_action_contract"]
            if "target_context" in payload:
                contract["target_context"] = payload["target_context"]
            return contract
    assert contract is not None
    return contract


def _legacy_duplicated_tailoring_messages(runtime, contract):
    """Approximate the pre-compaction full-schema tailoring payload for size comparison."""
    assert runtime.registry is not None
    target_job_id = contract["target_job_id"]
    citation_contract = runtime._build_citation_contract(target_job_id)
    target_context = runtime._tailoring_context(target_job_id)
    bullets = runtime._editable_tailoring_bullets()[:2]
    target_context["citation_contract"] = citation_contract
    target_context["editable_experience_bullets"] = [
        {
            "bullet_id": bullet.bullet_id,
            "text": target_context[f"bullet_{index + 1}_source"]["current_text"],
            "required_citations": citation_contract["bullet_required_citations"][
                bullet.bullet_id
            ],
        }
        for index, bullet in enumerate(bullets)
    ]
    full_shape = {
        "decision_summary": "<concise explanation>",
        "job_id": target_job_id,
        "edit_plan": {
            "job_id": target_job_id,
            "professional_summary": {
                "new_text": "<tailored summary>",
                "reason": "<reason>",
                "citations": citation_contract["summary_required_citations"],
            },
            "experience_bullet_edits": [
                {
                    "bullet_id": bullet["bullet_id"],
                    "new_text": "<tailored bullet text>",
                    "reason": "<reason>",
                    "citations": citation_contract["bullet_required_citations"][
                        bullet["bullet_id"]
                    ],
                }
                for bullet in target_context["editable_experience_bullets"]
            ],
            "skill_section_edits": [],
            "project_swap": None,
            "plan_rationale": "<concise rationale>",
        },
    }
    duplicated_contract = {
        **runtime._contract_for_model(contract),
        "target_context": target_context,
        "required_argument_shape": full_shape,
        "constraints": list(TAILOR_RESUME_CONSTRAINTS)
        + list(TAILOR_RESUME_PLAN_LIMITS),
    }
    checkpoint = runtime._build_workflow_checkpoint(
        {**contract, "target_context": target_context}
    )
    checkpoint["next_action_contract"] = duplicated_contract
    checkpoint["target_context"] = target_context
    legacy_schemas = runtime.registry.model_schemas(["tailor_resume"])
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": json.dumps(checkpoint, separators=(",", ":"))},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "user", "content": json.dumps({"tools": legacy_schemas}, separators=(",", ":"))},
    ]


def _current_tailoring_messages(runtime, contract):
    runtime._apply_conversation_checkpoint(contract)
    return list(runtime.conversation)


def _tailoring_contract(runtime):
    contract = runtime._next_action_contract()
    assert contract["allowed_tool"] == "tailor_resume"
    return contract


def _bad_semantic_tailor_draft(runtime):
    contract = _tailoring_contract(runtime)
    assert runtime.registry is not None
    assert runtime.state is not None
    job_id = contract["target_job_id"]
    job = runtime.registry._job(job_id)
    analysis = runtime.state.fit_analyses[job_id]
    plan = valid_resume_plan(job, analysis, runtime.registry.bundle)
    draft = _compact_draft_from_plan(plan, job_id)
    draft["professional_summary"]["new_text"] = (
        "Machine learning engineer with Python and retrieval systems."
    )
    return contract, _tool("tailor_resume", draft, 1)


def _transferred_metric_tailor_draft(runtime):
    contract = _tailoring_contract(runtime)
    assert runtime.registry is not None
    assert runtime.state is not None
    job_id = contract["target_job_id"]
    job = runtime.registry._job(job_id)
    plan = valid_resume_plan(
        job,
        runtime.state.fit_analyses[job_id],
        runtime.registry.bundle,
    )
    plan.experience_bullet_edits[0].new_text = (
        "Built Python and scikit-learn risk models over SQL datasets, raising "
        "validated recall by 14% while holding the false-positive rate constant."
    )
    draft = _compact_draft_from_plan(plan, job_id)
    transferred_text = draft["bullet_2"]["new_text"]
    draft["bullet_1"]["new_text"] = transferred_text
    draft["bullet_2"]["new_text"] = (
        "Built Python and scikit-learn risk models over SQL datasets, raising "
        "validated recall by 14% while holding the false-positive rate constant."
    )
    return contract, _tool("tailor_resume", draft, 1)


def _valid_compact_tailor_call(runtime):
    contract = _tailoring_contract(runtime)
    assert runtime.registry is not None
    assert runtime.state is not None
    job_id = contract["target_job_id"]
    job = runtime.registry._job(job_id)
    plan = valid_resume_plan(
        job,
        runtime.state.fit_analyses[job_id],
        runtime.registry.bundle,
    )
    plan.experience_bullet_edits[0].new_text = (
        "Built Python and scikit-learn risk models over SQL datasets, raising "
        "validated recall by 14% while holding the false-positive rate constant."
    )
    draft = _compact_draft_from_plan(plan, job_id)
    return contract, _tool("tailor_resume", draft, 1)


def test_tailor_resume_model_facing_schema_constraints(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    contract = _tailoring_contract(runtime)
    schemas = runtime._model_schemas_for_contract(contract)
    assert schemas[0]["function"]["name"] == "tailor_resume"
    schema_text = json.dumps(schemas[0]["function"]["parameters"])
    properties = schemas[0]["function"]["parameters"]["properties"]
    assert "bullet_1" in properties
    assert "bullet_2" in properties
    assert "experience_bullet_edits" not in properties
    assert "new_text" in schema_text
    assert "reason" in schema_text
    for forbidden in (
        "citation",
        "evidence_id",
        "bullet_id",
        "candidate_id",
        "remove_project_id",
        "add_project_id",
        "skill_section_edits",
        "edit_plan",
    ):
        assert forbidden not in schema_text
    with pytest.raises(Exception):
        _TailorResumeTextDraftNoSwap.model_validate(
            {
                "decision_summary": "x",
                "job_id": contract["target_job_id"],
                "professional_summary": {"new_text": "x", "reason": "x"},
                "bullet_1": {"new_text": "x", "reason": "x"},
                "project_swap_reason": None,
                "plan_rationale": "x",
            }
        )
    with pytest.raises(Exception):
        _TailorResumeTextDraftNoSwap.model_validate(
            {
                "decision_summary": "x",
                "job_id": contract["target_job_id"],
                "professional_summary": {"new_text": "x", "reason": "x"},
                "bullet_1": {"new_text": "x", "reason": "x"},
                "bullet_2": {"new_text": "x", "reason": "x"},
                "project_swap_reason": None,
                "plan_rationale": "x",
                "edit_plan": {},
            }
        )


def test_tailor_resume_compact_schema_is_smaller_than_full_registry_schema(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    assert runtime.registry is not None
    contract = _tailoring_contract(runtime)
    compact_size = len(
        json.dumps(
            runtime._model_schemas_for_contract(contract),
            separators=(",", ":"),
        )
    )
    full_size = len(
        json.dumps(
            runtime.registry.model_schemas(["tailor_resume"]),
            separators=(",", ":"),
        )
    )
    reduction = 1 - (compact_size / full_size)
    assert reduction >= 0.55


def test_hydration_injects_exact_citations_and_ids(tmp_path):
    from src.models.candidate import CandidateProfile
    from src.models.job import Job

    runtime = _runtime_at_tailoring(tmp_path)
    assert runtime.registry is not None
    contract = _tailoring_contract(runtime)
    job_id = contract["target_job_id"]
    candidate_id = runtime.registry.bundle.profile.candidate_id
    citation_contract = runtime._build_citation_contract(job_id)
    default_job = citation_contract["default_job_citation"]
    assert default_job == {
        "source_type": "job_posting",
        "source_id": job_id,
        "source_field": "required_skills_raw",
        "evidence_id": None,
    }
    assert "citation_contract" not in contract["target_context"]
    assert "bullet_1_source" in contract["target_context"]
    assert "allowed_numeric_claims" in contract["target_context"]["bullet_1_source"]
    assert "allowed_numeric_claims" in contract["target_context"]["bullet_2_source"]

    _, raw_call = _valid_compact_tailor_call(runtime)
    raw_snapshot = copy.deepcopy(raw_call.arguments)
    hydrated = runtime._hydrate_tailor_resume_call(raw_call, contract)
    assert raw_call.arguments == raw_snapshot

    summary_citations = hydrated.arguments["edit_plan"]["professional_summary"][
        "citations"
    ]
    assert summary_citations[0]["source_type"] == "job_posting"
    assert summary_citations[0]["source_id"] == job_id
    assert summary_citations[0]["source_field"] in Job.model_fields
    assert summary_citations[0]["source_field"] == "required_skills_raw"
    assert summary_citations[1] == {
        "source_type": "candidate_profile",
        "source_id": candidate_id,
        "source_field": "experience",
        "evidence_id": None,
    }
    assert summary_citations[1]["source_field"] in CandidateProfile.model_fields

    bullet_one = hydrated.arguments["edit_plan"]["experience_bullet_edits"][0]
    assert bullet_one["bullet_id"] == "exp-primary-bullet-1"
    assert bullet_one["citations"][0] == {
        "source_type": "experience_bullet",
        "source_id": "exp-primary-bullet-1",
        "source_field": "text",
        "evidence_id": "EV-EXP-BULLET-001",
    }
    assert bullet_one["citations"][1] == default_job
    bullet_two = hydrated.arguments["edit_plan"]["experience_bullet_edits"][1]
    assert bullet_two["bullet_id"] == "exp-primary-bullet-2"
    assert bullet_two["citations"][0]["evidence_id"] == "EV-EXP-BULLET-002"
    assert hydrated.arguments["edit_plan"]["skill_section_edits"] == []
    assert hydrated.arguments["job_id"] == job_id
    assert hydrated.arguments["edit_plan"]["job_id"] == job_id
    assert hydrated.arguments["edit_plan"]["project_swap"] is None


def test_semantic_text_recovery_after_role_phrase_failure(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    progress: list[str] = []
    runtime.progress_callback = progress.append
    runtime.state.model_call_count = 6

    contract, call = _bad_semantic_tailor_draft(runtime)
    response = NormalizedAssistantMessage(tool_calls=[call])
    valid, invalid = runtime._execute_response_calls(response, contract)
    assert valid == 0
    assert invalid == 1
    recovery = json.loads(runtime.conversation[-1]["content"])
    assert recovery["type"] == "invalid_tool_call_recovery"
    assert recovery["error_category"] == "semantic_text"
    assert recovery["required_role_phrase"] == "AI engineer"
    assert recovery["tailor_recovery_mode"] == "patch"
    assert recovery["patch_fields"] == ["professional_summary"]
    assert recovery["instruction"].count('"AI engineer"') == 1
    assert "patch call" in recovery["instruction"]
    assert "professional_summary" in recovery["required_argument_shape"]
    assert "bullet_1" not in recovery["required_argument_shape"]
    assert len(runtime.conversation) == 3
    assert "Validation category: semantic_text" in progress
    assert "Validation category: citation" not in progress

    contract2, call2 = _valid_compact_tailor_call(runtime)
    runtime._clear_tailor_patch_recovery()
    hydrated1 = runtime._hydrate_tailor_resume_call(call, contract)
    hydrated2 = runtime._hydrate_tailor_resume_call(call2, contract2)
    assert (
        hydrated1.arguments["edit_plan"]["professional_summary"]["citations"]
        == hydrated2.arguments["edit_plan"]["professional_summary"]["citations"]
    )


def test_semantic_recovery_omits_structural_template(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    contract, call = _bad_semantic_tailor_draft(runtime)
    runtime._append_invalid_message(
        call,
        'Tailor resume semantic rejection: the professional summary must explicitly include the role phrase "AI engineer".',
        contract,
    )
    payload = json.loads(runtime.conversation[-1]["content"])
    assert payload["type"] == "invalid_tool_call_recovery"
    assert payload["error_category"] == "semantic_text"
    assert payload["required_role_phrase"] == "AI engineer"
    assert "exact_tailor_resume_structural_template" not in payload
    assert len(runtime.conversation) == 3


def test_hydrated_compact_draft_executes_resume_tool(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.tools.resume_tailoring.compile_resume_pdf", fake_resume_compiler
    )
    runtime = _runtime_at_tailoring(tmp_path)
    assert runtime.state is not None
    tracer = NoOpAgentTracer()
    runtime.trace = tracer.start_trace("agent_run", run_id=runtime.run_id)
    contract, call = _valid_compact_tailor_call(runtime)
    response = NormalizedAssistantMessage(tool_calls=[call])
    valid, invalid = runtime._execute_response_calls(response, contract)
    assert valid == 1
    assert invalid == 0
    assert len(runtime.state.draft_resumes) == 1
    assert runtime.state.draft_resumes[contract["target_job_id"]] is not None
    tool_span = next(
        item
        for item in runtime.trace.record.observations
        if item.name == "tool_call:tailor_resume"
    )
    assert "edit_plan" in tool_span.input["arguments"]
    assert tool_span.input["arguments"]["edit_plan"]["professional_summary"]["citations"]


def test_first_tailoring_payload_is_deduplicated_and_smaller(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    ids = runtime.state.top_3_job_ids
    sizes: list[int] = []
    for rank in range(1, 4):
        runtime.state.phase = AgentPhase.TAILORING
        runtime.state.draft_resumes = {
            ids[index]: object() for index in range(rank - 1)
        }
        contract = runtime._next_action_contract()
        current_messages = _current_tailoring_messages(runtime, contract)
        sizes.append(len(json.dumps(current_messages, separators=(",", ":"))))
        serialized = json.dumps(current_messages)
        assert '"source_type"' not in serialized
        assert '"evidence_id"' not in serialized
    assert max(sizes) < 8731
    contract = _tailoring_contract(runtime)
    legacy_messages = _legacy_duplicated_tailoring_messages(runtime, contract)
    current_messages = _current_tailoring_messages(runtime, contract)
    current_size = len(json.dumps(current_messages, separators=(",", ":")))
    legacy_size = len(json.dumps(legacy_messages, separators=(",", ":")))
    reduction = 1 - (current_size / legacy_size)
    assert reduction >= 0.30
    checkpoint = json.loads(current_messages[1]["content"])
    shape = checkpoint["next_action_contract"]["required_argument_shape"]
    assert "edit_plan" not in shape
    assert checkpoint["target_context"]["target_job_id"] == contract["target_job_id"]
    assert "target_context" not in checkpoint["next_action_contract"]
    schemas = runtime._model_schemas_for_contract(contract)
    assert "ResumeEditPlan" not in json.dumps(schemas)


def test_missing_tool_call_stop_recovery_is_bounded(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    contract = _tailoring_contract(runtime)
    runtime._apply_conversation_checkpoint(contract)
    runtime.state.model_call_count = 6
    response = NormalizedAssistantMessage(content="", tool_calls=[], done_reason="stop")
    valid, invalid = runtime._execute_response_calls(response, contract)
    assert valid == 0
    assert invalid == 1
    assert len(runtime.conversation) == 3
    retry = json.loads(runtime.conversation[-1]["content"])
    assert retry["type"] == "tool_call_retry"
    assert retry["reason"] == "missing_tool_call"
    checkpoint = json.loads(runtime.conversation[1]["content"])
    assert "target_context" in checkpoint
    assert "target_context" not in checkpoint["next_action_contract"]


def test_model_call_payload_diagnostics_are_recorded_not_in_messages(tmp_path):
    tracer = NoOpAgentTracer()
    runtime = _runtime(
        tmp_path,
        ScriptedClient(
            [NormalizedAssistantMessage(content="", tool_calls=[])]
        ),
        ReviewProvider("unused"),
        tracer,
    )
    runtime.trace = tracer.start_trace("agent_run", run_id=runtime.run_id)
    runtime._load_inputs()
    assert runtime.registry is not None
    runtime.registry.execute("filter_jobs", {"decision_summary": "Filter once."})
    runtime.registry.execute("score_jobs", {"decision_summary": "Score once."})
    for job_id in runtime.state.top_3_job_ids:
        runtime.registry.execute(
            "analyze_fit",
            {"decision_summary": f"Analyze {job_id}.", "job_id": job_id},
        )
    contract = runtime._next_action_contract()
    runtime._apply_conversation_checkpoint(contract)
    before = runtime._serialized_conversation_char_count()
    runtime._call_model(runtime.trace, contract)
    generation = runtime.trace.record.observations[-1]
    assert generation.metadata["model_message_count"] == 2
    assert generation.metadata["serialized_message_char_count"] == before
    assert generation.metadata["serialized_tool_schema_char_count"] > 0
    assert generation.metadata["phase"] == "resume_tailoring"
    assert generation.metadata["target_rank"] == 1
    assert generation.metadata["model_argument_mode"] == "tailor_resume_text_draft"
    assert generation.metadata["semantic_bullet_slot_mode"] == "named"
    assert generation.metadata["required_role_phrase"] == "AI engineer"
    assert generation.metadata["evidence_reconciliation_applied"] is True
    assert "reconciled_aligned_skill_count" in generation.metadata
    assert "model_message_count" not in runtime.conversation[-1]["content"]


def test_required_role_phrase_validation(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    contract = _tailoring_contract(runtime)
    assert runtime._derive_required_role_phrase("AI Engineer") == "AI engineer"
    assert runtime._summary_includes_role_phrase(
        "Experienced AI ENGINEER delivering Python systems.",
        "AI engineer",
    )
    assert not runtime._summary_includes_role_phrase(
        "Machine learning engineer with Python systems.",
        "AI engineer",
    )
    _, bad_call = _bad_semantic_tailor_draft(runtime)
    hydrated = runtime._hydrate_tailor_resume_call(bad_call, contract)
    with pytest.raises(ToolArgumentsError, match='role phrase "AI engineer"'):
        runtime._validate_hydrated_tailor_semantics(hydrated, contract)


def test_bullet_slot_mapping_is_deterministic_when_semantic_text_swapped(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    contract, call = _transferred_metric_tailor_draft(runtime)
    hydrated = runtime._hydrate_tailor_resume_call(call, contract)
    edits = hydrated.arguments["edit_plan"]["experience_bullet_edits"]
    assert edits[0]["bullet_id"] == "exp-primary-bullet-1"
    assert edits[1]["bullet_id"] == "exp-primary-bullet-2"
    assert "36%" in edits[0]["new_text"]
    assert "14%" in edits[1]["new_text"]


def test_evidence_recovery_for_transferred_metric(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.tools.resume_tailoring.compile_resume_pdf", fake_resume_compiler
    )
    runtime = _runtime_at_tailoring(tmp_path)
    progress: list[str] = []
    runtime.progress_callback = progress.append
    runtime.state.model_call_count = 6
    contract, call = _transferred_metric_tailor_draft(runtime)
    response = NormalizedAssistantMessage(tool_calls=[call])
    valid, invalid = runtime._execute_response_calls(response, contract)
    assert valid == 0
    assert invalid == 1
    recovery = json.loads(runtime.conversation[-1]["content"])
    assert recovery["error_category"] == "evidence"
    assert recovery["tailor_recovery_mode"] == "patch"
    assert "bullet_1" in recovery["patch_fields"]
    assert recovery["rejected_bullet_slot"] == "bullet_1"
    assert "bullet_1_source" in recovery
    assert "36%" in recovery["error"]
    assert "Validation category: evidence" in progress
    assert "Validation category: citation" not in progress


def test_dynamic_project_swap_schema(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    assert runtime.state is not None
    ids = runtime.state.top_3_job_ids
    no_swap_contract = runtime._next_action_contract()
    assert no_swap_contract["project_swap_required"] is False
    with pytest.raises(ToolArgumentsError, match="draft schema rejection"):
        runtime._parse_tailor_draft_arguments(
            {
                "decision_summary": "x",
                "job_id": no_swap_contract["target_job_id"],
                "professional_summary": {"new_text": "AI engineer x", "reason": "x"},
                "bullet_1": {"new_text": "x", "reason": "x"},
                "bullet_2": {"new_text": "x", "reason": "x"},
                "project_swap_reason": "should not be accepted",
                "plan_rationale": "x",
            },
            no_swap_contract,
        )
    runtime._parse_tailor_draft_arguments(
        {
            "decision_summary": "x",
            "job_id": no_swap_contract["target_job_id"],
            "professional_summary": {"new_text": "AI engineer x", "reason": "x"},
            "bullet_1": {"new_text": "x", "reason": "x"},
            "bullet_2": {"new_text": "x", "reason": "x"},
            "project_swap_reason": None,
            "plan_rationale": "x",
        },
        no_swap_contract,
    )

    swap_job_id = next(
        job_id
        for job_id in ids
        if runtime.state.fit_analyses[job_id].projects.swap_suggestion is not None
    )
    swap_contract = runtime._next_action_contract()
    while swap_contract["target_job_id"] != swap_job_id:
        runtime.state.draft_resumes[swap_contract["target_job_id"]] = object()
        swap_contract = runtime._next_action_contract()
    assert swap_contract["project_swap_required"] is True
    with pytest.raises(ToolArgumentsError, match="draft schema rejection"):
        runtime._parse_tailor_draft_arguments(
            {
                "decision_summary": "x",
                "job_id": swap_job_id,
                "professional_summary": {"new_text": "AI engineer x", "reason": "x"},
                "bullet_1": {"new_text": "x", "reason": "x"},
                "bullet_2": {"new_text": "x", "reason": "x"},
                "project_swap_reason": None,
                "plan_rationale": "x",
            },
            swap_contract,
        )
    swap_draft = runtime._parse_tailor_draft_arguments(
        {
            "decision_summary": "x",
            "job_id": swap_job_id,
            "professional_summary": {"new_text": "AI engineer x", "reason": "x"},
            "bullet_1": {"new_text": "x", "reason": "x"},
            "bullet_2": {"new_text": "x", "reason": "x"},
            "project_swap_reason": "Swap to stronger portfolio evidence.",
            "plan_rationale": "x",
        },
        swap_contract,
    )
    hydrated = runtime._hydrate_tailor_resume_call(
        _tool("tailor_resume", swap_draft.model_dump(mode="json"), 1),
        swap_contract,
    )
    swap = hydrated.arguments["edit_plan"]["project_swap"]
    expected = runtime.state.fit_analyses[swap_job_id].projects.swap_suggestion
    assert swap["remove_project_id"] == expected.remove_project_id
    assert swap["add_project_id"] == expected.add_project_id


def _contract_for_rank(runtime, rank: int):
    assert runtime.state is not None
    ids = runtime.state.top_3_job_ids
    runtime.state.phase = AgentPhase.TAILORING
    runtime.state.draft_resumes = {ids[index]: object() for index in range(rank - 1)}
    contract = runtime._next_action_contract()
    assert contract["target_rank"] == rank
    return contract


def test_evidence_reconciliation_is_disjoint_and_fixture_correct(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    assert runtime.state is not None
    rank3_id = runtime.state.top_3_job_ids[2]
    raw = copy.deepcopy(runtime.state.fit_analyses[rank3_id])
    reconciled = runtime._reconcile_tailoring_evidence(rank3_id)
    unchanged = runtime.state.fit_analyses[rank3_id]

    assert unchanged.core_skills.genuine_gaps == raw.core_skills.genuine_gaps
    assert unchanged.core_skills.aligned_skills == raw.core_skills.aligned_skills
    assert "data pipelines" in raw.core_skills.genuine_gaps
    assert "evaluation" in raw.core_skills.aligned_skills
    assert "data pipelines" in reconciled.aligned_skills
    assert "data pipelines" not in reconciled.genuine_gaps
    assert "evaluation" not in reconciled.aligned_skills
    assert "evaluation" in reconciled.genuine_gaps
    assert "data pipelines" in reconciled.skills_moved_from_gap_to_supported
    assert "evaluation" in reconciled.skills_moved_from_aligned_to_gap
    assert reconciled.reconciliation_applied is True

    aligned_keys = set()
    for skill in reconciled.aligned_skills:
        from src.tools.scoring import normalize_skill

        key = normalize_skill(skill, has_vector_search=True)
        if key:
            aligned_keys.add(key)
    evidenced_keys = set()
    for skill in reconciled.evidenced_elsewhere_skills:
        from src.tools.scoring import normalize_skill

        key = normalize_skill(skill, has_vector_search=True)
        if key:
            evidenced_keys.add(key)
    gap_keys = set()
    for skill in reconciled.genuine_gaps:
        from src.tools.scoring import normalize_skill

        key = normalize_skill(skill, has_vector_search=True)
        if key:
            gap_keys.add(key)
    assert not aligned_keys & evidenced_keys
    assert not aligned_keys & gap_keys
    assert not evidenced_keys & gap_keys

    contract = _contract_for_rank(runtime, 3)
    context = contract["target_context"]
    assert context["genuine_gaps"] == reconciled.genuine_gaps
    assert context["aligned_skills"] == reconciled.aligned_skills
    assert "evaluation" in context["do_not_claim_skills"]


def test_draft_audit_collects_multiple_field_issues(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    contract, call = _bad_semantic_tailor_draft(runtime)
    transferred, transfer_call = _transferred_metric_tailor_draft(runtime)
    del transferred
    draft = copy.deepcopy(transfer_call.arguments)
    draft["professional_summary"]["new_text"] = (
        "Machine learning engineer with evaluation and Python systems."
    )
    hydrated = runtime._hydrate_tailor_resume_call(
        _tool("tailor_resume", draft, 1),
        contract,
    )
    audit = runtime._audit_hydrated_tailor_draft(hydrated, contract)
    fields = set(audit.fields)
    assert "professional_summary" in fields
    assert "bullet_1" in fields
    assert len(audit.issues) >= 2
    categories = {issue.category for issue in audit.issues}
    assert "semantic_text" in categories
    assert "evidence" in categories


def test_draft_audit_supported_data_pipelines_passes(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    contract = _contract_for_rank(runtime, 3)
    assert runtime.registry is not None
    assert runtime.state is not None
    job_id = contract["target_job_id"]
    job = runtime.registry._job(job_id)
    plan = valid_resume_plan(
        job,
        runtime.state.fit_analyses[job_id],
        runtime.registry.bundle,
    )
    draft = _compact_draft_from_plan(plan, job_id)
    draft["professional_summary"]["new_text"] = (
        "AI engineer with SQL data pipelines and Python ML systems."
    )
    hydrated = runtime._hydrate_tailor_resume_call(
        _tool("tailor_resume", draft, 1),
        contract,
    )
    audit = runtime._audit_hydrated_tailor_draft(hydrated, contract)
    summary_issues = [
        issue for issue in audit.issues if issue.field == "professional_summary"
    ]
    assert not any("data pipelines" in issue.message for issue in summary_issues)


def test_draft_audit_unsupported_evaluation_fails(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    contract = _contract_for_rank(runtime, 3)
    assert runtime.registry is not None
    assert runtime.state is not None
    job_id = contract["target_job_id"]
    job = runtime.registry._job(job_id)
    plan = valid_resume_plan(
        job,
        runtime.state.fit_analyses[job_id],
        runtime.registry.bundle,
    )
    draft = _compact_draft_from_plan(plan, job_id)
    draft["professional_summary"]["new_text"] = (
        "AI engineer with evaluation frameworks and Python ML systems."
    )
    hydrated = runtime._hydrate_tailor_resume_call(
        _tool("tailor_resume", draft, 1),
        contract,
    )
    audit = runtime._audit_hydrated_tailor_draft(hydrated, contract)
    assert any(
        issue.field == "professional_summary" and issue.category == "evidence"
        for issue in audit.issues
    )


def test_patch_schema_contains_only_failed_fields(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    runtime.state.model_call_count = 6
    contract, call = _bad_semantic_tailor_draft(runtime)
    runtime._execute_response_calls(
        NormalizedAssistantMessage(tool_calls=[call]),
        contract,
    )
    patch_contract = runtime._next_action_contract()
    schemas = runtime._model_schemas_for_contract(patch_contract)
    properties = schemas[0]["function"]["parameters"]["properties"]
    assert set(properties) == {"job_id", "professional_summary"}
    assert "bullet_1" not in properties
    assert "bullet_2" not in properties

    runtime._clear_tailor_patch_recovery()
    contract2, call2 = _valid_compact_tailor_call(runtime)
    draft = copy.deepcopy(call2.arguments)
    draft["bullet_1"]["new_text"] = draft["bullet_2"]["new_text"]
    runtime.state.model_call_count = 7
    runtime._execute_response_calls(
        NormalizedAssistantMessage(tool_calls=[_tool("tailor_resume", draft, 2)]),
        contract2,
    )
    bullet_contract = runtime._next_action_contract()
    bullet_props = runtime._model_schemas_for_contract(bullet_contract)[0][
        "function"
    ]["parameters"]["properties"]
    assert set(bullet_props) == {"job_id", "bullet_1"}


def test_patch_merge_preserves_valid_fields_byte_for_byte(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    runtime.state.model_call_count = 6
    contract, call = _bad_semantic_tailor_draft(runtime)
    base_snapshot = copy.deepcopy(call.arguments)
    runtime._execute_response_calls(
        NormalizedAssistantMessage(tool_calls=[call]),
        contract,
    )
    assert runtime._tailor_patch_recovery is not None
    assert base_snapshot == runtime._tailor_patch_recovery["base_draft"]
    patch = {
        "job_id": contract["target_job_id"],
        "professional_summary": {
            "new_text": "AI engineer with Python ML and SQL systems.",
            "reason": "Align summary with target role.",
        },
    }
    patch_contract = runtime._next_action_contract()
    merged_call = _tool("tailor_resume", patch, 2)
    hydrated = runtime._hydrate_tailor_resume_call(merged_call, patch_contract)
    assert hydrated.arguments["decision_summary"] == base_snapshot["decision_summary"]
    assert hydrated.arguments["edit_plan"]["plan_rationale"] == base_snapshot[
        "plan_rationale"
    ]
    assert (
        hydrated.arguments["edit_plan"]["experience_bullet_edits"][0]["new_text"]
        == base_snapshot["bullet_1"]["new_text"]
    )
    assert (
        hydrated.arguments["edit_plan"]["experience_bullet_edits"][1]["new_text"]
        == base_snapshot["bullet_2"]["new_text"]
    )


def test_patch_recovery_payload_smaller_than_initial(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    runtime.state.model_call_count = 6
    contract, call = _bad_semantic_tailor_draft(runtime)
    initial_checkpoint = json.loads(
        _current_tailoring_messages(runtime, contract)[1]["content"]
    )
    initial_size = len(json.dumps(initial_checkpoint, separators=(",", ":")))
    runtime._execute_response_calls(
        NormalizedAssistantMessage(tool_calls=[call]),
        contract,
    )
    recovery = json.loads(runtime.conversation[-1]["content"])
    patch_size = len(json.dumps(recovery, separators=(",", ":")))
    assert patch_size < initial_size
    assert len(runtime.conversation) == 3


def test_rejection_category_mapping(tmp_path):
    classify = JobSearchAgentRuntime._classify_tailor_rejection
    assert classify("Candidate text claims unsupported required skill 'evaluation'") == "evidence"
    assert classify("Edited candidate text claims genuine-gap skill 'data pipelines'") == "evidence"
    assert classify("Tailor resume draft audit rejection: bullet_1: unsupported numeric claim '36%'") == "evidence"
    assert (
        classify(
            'Tailor resume semantic rejection: the professional summary must '
            'explicitly include the role phrase "AI engineer".'
        )
        == "semantic_text"
    )
    assert (
        classify(
            "Tailor resume hydration rejection: project_swap_reason is required"
        )
        == "hydration"
    )
    assert classify("Tool registry rejected the requested call: protected section edited") == "tool_execution"


def test_rank3_simultaneous_invalid_fields_patch_preserves_swap_reason(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    runtime.state.model_call_count = 6
    contract = _contract_for_rank(runtime, 3)
    if not contract["project_swap_required"]:
        pytest.skip("Rank 3 fixture has no project swap in this dataset")
    assert runtime.registry is not None
    assert runtime.state is not None
    job_id = contract["target_job_id"]
    job = runtime.registry._job(job_id)
    plan = valid_resume_plan(
        job,
        runtime.state.fit_analyses[job_id],
        runtime.registry.bundle,
    )
    draft = _compact_draft_from_plan(plan, job_id)
    draft["professional_summary"]["new_text"] = "Machine learning engineer with Python."
    draft["bullet_1"]["new_text"] = draft["bullet_2"]["new_text"]
    swap_reason = draft["project_swap_reason"]
    runtime._execute_response_calls(
        NormalizedAssistantMessage(tool_calls=[_tool("tailor_resume", draft, 1)]),
        contract,
    )
    recovery = json.loads(runtime.conversation[-1]["content"])
    assert recovery["tailor_recovery_mode"] == "patch"
    assert "project_swap_reason" not in recovery["patch_fields"]
    assert runtime._tailor_patch_recovery["base_draft"]["project_swap_reason"] == swap_reason


def test_build_resume_execution_fit_analysis_rank3_fixture(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    job_id = runtime.state.top_3_job_ids[2]
    raw = runtime.state.fit_analyses[job_id]
    execution = runtime._build_resume_execution_fit_analysis(job_id)

    assert execution is not raw
    assert execution.model_dump() != raw.model_dump()
    assert "data pipelines" in raw.core_skills.genuine_gaps
    assert "evaluation" in raw.core_skills.aligned_skills
    assert "data pipelines" in execution.core_skills.aligned_skills
    assert "data pipelines" not in execution.core_skills.genuine_gaps
    assert "evaluation" in execution.core_skills.genuine_gaps
    assert execution.job_id == raw.job_id
    assert execution.formatted_text == raw.formatted_text
    assert execution.projects.swap_suggestion == raw.projects.swap_suggestion
    assert execution.relevant_experience == raw.relevant_experience


def test_raw_fit_analysis_preserved_after_successful_tailor_execution(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "src.tools.resume_tailoring.compile_resume_pdf", fake_resume_compiler
    )
    runtime = _runtime_at_tailoring(tmp_path)
    contract, call = _valid_compact_tailor_call(runtime)
    job_id = contract["target_job_id"]
    raw_before = runtime.state.fit_analyses[job_id]
    raw_snapshot = copy.deepcopy(raw_before.model_dump())
    analyze_record = next(
        item
        for item in runtime.state.tool_execution_history
        if item.tool_name == "analyze_fit" and item.job_id == job_id
    )

    runtime.state.model_call_count = 6
    runtime.trace = NoOpAgentTracer().start_trace("agent_run", run_id=runtime.run_id)
    valid, invalid = runtime._execute_response_calls(
        NormalizedAssistantMessage(tool_calls=[call]),
        contract,
    )
    assert valid == 1
    assert invalid == 0
    raw_after = runtime.state.fit_analyses[job_id]
    assert raw_after is raw_before
    assert raw_after.model_dump() == raw_snapshot
    assert analyze_record.result_summary["core_skills"]["genuine_gaps"] == (
        raw_before.core_skills.genuine_gaps
    )
    tool_span = next(
        item
        for item in runtime.trace.record.observations
        if item.name == "tool_call:tailor_resume"
    )
    assert tool_span.metadata["reconciled_execution_evidence"] is True
    assert tool_span.metadata["raw_fit_analysis_preserved"] is True


def test_data_pipelines_not_rejected_as_genuine_gap_with_execution_copy(tmp_path):
    from src.tools.resume_tailoring import (
        ResumeEvidenceError,
        _candidate_supported_skills,
        _validate_no_genuine_gap_claims,
        _validate_required_skill_claims,
    )

    runtime = _runtime_at_tailoring(tmp_path)
    job_id = runtime.state.top_3_job_ids[2]
    raw = runtime.state.fit_analyses[job_id]
    execution = runtime._build_resume_execution_fit_analysis(job_id)
    text = "AI engineer with SQL data pipelines and Python ML systems."

    with pytest.raises(ResumeEvidenceError, match="genuine-gap skill 'data pipelines'"):
        _validate_no_genuine_gap_claims(text, raw)
    _validate_no_genuine_gap_claims(text, execution)

    assert runtime.registry is not None
    job = runtime.registry._job(job_id)
    raw_supported = _candidate_supported_skills(
        runtime.registry.bundle,
        runtime.registry.memory,
    )
    with pytest.raises(
        ResumeEvidenceError, match="unsupported required skill 'data pipelines'"
    ):
        _validate_required_skill_claims(text, job, raw_supported)

    execution_bundle, _ = runtime._build_resume_execution_bundle(job_id)
    execution_supported = _candidate_supported_skills(
        execution_bundle,
        runtime.registry.memory,
    )
    _validate_required_skill_claims(text, job, execution_supported)


def test_build_resume_execution_bundle_rank3_promotes_data_pipelines(tmp_path):
    runtime = _runtime_at_tailoring(tmp_path)
    job_id = runtime.state.top_3_job_ids[2]
    assert runtime.registry is not None
    raw_bundle = runtime.registry.bundle
    raw_record = raw_bundle.get_evidence("EV-EXP-BULLET-001")
    assert raw_record is not None
    assert "data pipelines" not in raw_record.supported_skills

    execution_bundle, metadata = runtime._build_resume_execution_bundle(job_id)
    assert execution_bundle is not raw_bundle
    promoted = execution_bundle.get_evidence("EV-EXP-BULLET-001")
    assert promoted is not None
    assert "data pipelines" in promoted.supported_skills
    assert metadata["promoted_execution_skills"] == ["data pipelines"]
    assert metadata["promoted_evidence_record_count"] == 1

    unchanged = execution_bundle.get_evidence("EV-EXP-BULLET-002")
    raw_second = raw_bundle.get_evidence("EV-EXP-BULLET-002")
    assert unchanged is not None and raw_second is not None
    assert unchanged.supported_skills == raw_second.supported_skills
    assert raw_record.supported_skills == raw_bundle.get_evidence(
        "EV-EXP-BULLET-001"
    ).supported_skills


def test_raw_candidate_bundle_preserved_after_successful_tailor_execution(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "src.tools.resume_tailoring.compile_resume_pdf", fake_resume_compiler
    )
    runtime = _runtime_at_tailoring(tmp_path)
    contract = _contract_for_rank(runtime, 3)
    job_id = contract["target_job_id"]
    assert runtime.registry is not None
    raw_bundle = runtime.registry.bundle
    raw_bundle_snapshot = copy.deepcopy(raw_bundle.model_dump())
    raw_record_snapshot = copy.deepcopy(
        raw_bundle.get_evidence("EV-EXP-BULLET-001").model_dump()
    )

    job = runtime.registry._job(job_id)
    plan = valid_resume_plan(
        job,
        runtime.state.fit_analyses[job_id],
        raw_bundle,
    )
    draft = _compact_draft_from_plan(plan, job_id)
    draft["professional_summary"]["new_text"] = (
        "AI engineer with SQL data pipelines and Python ML systems."
    )
    runtime.state.model_call_count = 6
    runtime.trace = NoOpAgentTracer().start_trace("agent_run", run_id=runtime.run_id)
    valid, invalid = runtime._execute_response_calls(
        NormalizedAssistantMessage(tool_calls=[_tool("tailor_resume", draft, 1)]),
        contract,
    )
    assert valid == 1
    assert invalid == 0
    assert runtime.registry.bundle is raw_bundle
    assert runtime.registry.bundle.model_dump() == raw_bundle_snapshot
    assert (
        runtime.registry.bundle.get_evidence("EV-EXP-BULLET-001").model_dump()
        == raw_record_snapshot
    )
    tool_span = next(
        item
        for item in runtime.trace.record.observations
        if item.name == "tool_call:tailor_resume"
    )
    assert tool_span.metadata["reconciled_execution_bundle"] is True
    assert tool_span.metadata["raw_candidate_bundle_preserved"] is True
    assert tool_span.metadata["promoted_execution_skills"] == ["data pipelines"]
    assert "profile" not in json.dumps(tool_span.metadata)


def test_raw_candidate_bundle_restored_after_tailor_execution_failure(
    tmp_path, monkeypatch
):
    def _fail_compile(*args, **kwargs):
        raise RuntimeError("simulated PDF compile failure")

    monkeypatch.setattr(
        "src.tools.resume_tailoring.compile_resume_pdf", _fail_compile
    )
    runtime = _runtime_at_tailoring(tmp_path)
    contract = _contract_for_rank(runtime, 3)
    job_id = contract["target_job_id"]
    assert runtime.registry is not None
    raw_bundle = runtime.registry.bundle
    raw_bundle_snapshot = copy.deepcopy(raw_bundle.model_dump())
    job = runtime.registry._job(job_id)
    plan = valid_resume_plan(
        job,
        runtime.state.fit_analyses[job_id],
        raw_bundle,
    )
    draft = _compact_draft_from_plan(plan, job_id)
    draft["professional_summary"]["new_text"] = (
        "AI engineer with SQL data pipelines and Python ML systems."
    )
    runtime.state.model_call_count = 6
    valid, invalid = runtime._execute_response_calls(
        NormalizedAssistantMessage(tool_calls=[_tool("tailor_resume", draft, 1)]),
        contract,
    )
    assert valid == 0
    assert invalid == 1
    assert runtime.registry.bundle is raw_bundle
    assert runtime.registry.bundle.model_dump() == raw_bundle_snapshot


def test_cover_letter_phase_observes_raw_candidate_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.tools.resume_tailoring.compile_resume_pdf", fake_resume_compiler
    )
    runtime = _runtime_at_tailoring(tmp_path)
    assert runtime.registry is not None
    raw_bundle = runtime.registry.bundle
    raw_bundle_snapshot = copy.deepcopy(raw_bundle.model_dump())

    for rank in range(1, 4):
        contract = _contract_for_rank(runtime, rank)
        job_id = contract["target_job_id"]
        job = runtime.registry._job(job_id)
        plan = valid_resume_plan(
            job,
            runtime.state.fit_analyses[job_id],
            raw_bundle,
        )
        draft = _compact_draft_from_plan(plan, job_id)
        if rank == 3:
            draft["professional_summary"]["new_text"] = (
                "AI engineer with SQL data pipelines and Python ML systems."
            )
        runtime.state.model_call_count = 5 + rank
        valid, invalid = runtime._execute_response_calls(
            NormalizedAssistantMessage(tool_calls=[_tool("tailor_resume", draft, rank)]),
            contract,
        )
        assert valid == 1
        assert invalid == 0
        assert runtime.registry.bundle is raw_bundle
        assert runtime.registry.bundle.model_dump() == raw_bundle_snapshot

    assert len(runtime.state.draft_resumes) == 3
    assert runtime.registry.bundle is raw_bundle


def test_rank3_data_pipelines_execution_no_genuine_gap_rejection(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "src.tools.resume_tailoring.compile_resume_pdf", fake_resume_compiler
    )
    runtime = _runtime_at_tailoring(tmp_path)
    contract = _contract_for_rank(runtime, 3)
    job_id = contract["target_job_id"]
    assert runtime.registry is not None
    job = runtime.registry._job(job_id)
    plan = valid_resume_plan(
        job,
        runtime.state.fit_analyses[job_id],
        runtime.registry.bundle,
    )
    draft = _compact_draft_from_plan(plan, job_id)
    draft["professional_summary"]["new_text"] = (
        "AI engineer with SQL data pipelines and Python ML systems."
    )
    runtime.state.model_call_count = 6
    valid, invalid = runtime._execute_response_calls(
        NormalizedAssistantMessage(tool_calls=[_tool("tailor_resume", draft, 1)]),
        contract,
    )
    assert valid == 1
    assert invalid == 0
    assert job_id in runtime.state.draft_resumes
    assert "data pipelines" in runtime.state.fit_analyses[job_id].core_skills.genuine_gaps


def test_raw_fit_analysis_restored_after_tailor_execution_failure(
    tmp_path, monkeypatch
):
    def _fail_compile(*args, **kwargs):
        raise RuntimeError("simulated PDF compile failure")

    monkeypatch.setattr(
        "src.tools.resume_tailoring.compile_resume_pdf", _fail_compile
    )
    runtime = _runtime_at_tailoring(tmp_path)
    contract, call = _valid_compact_tailor_call(runtime)
    job_id = contract["target_job_id"]
    raw_before = runtime.state.fit_analyses[job_id]
    raw_snapshot = copy.deepcopy(raw_before.model_dump())
    runtime.state.model_call_count = 6
    valid, invalid = runtime._execute_response_calls(
        NormalizedAssistantMessage(tool_calls=[call]),
        contract,
    )
    assert valid == 0
    assert invalid == 1
    raw_after = runtime.state.fit_analyses[job_id]
    assert raw_after is raw_before
    assert raw_after.model_dump() == raw_snapshot
    recovery = json.loads(runtime.conversation[-1]["content"])
    assert recovery["error_category"] == "tool_execution"


def test_rank3_evaluation_claim_rejected_by_resume_execution_path(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "src.tools.resume_tailoring.compile_resume_pdf", fake_resume_compiler
    )
    runtime = _runtime_at_tailoring(tmp_path)
    contract = _contract_for_rank(runtime, 3)
    job_id = contract["target_job_id"]
    assert runtime.registry is not None
    job = runtime.registry._job(job_id)
    plan = valid_resume_plan(
        job,
        runtime.state.fit_analyses[job_id],
        runtime.registry.bundle,
    )
    draft = _compact_draft_from_plan(plan, job_id)
    draft["professional_summary"]["new_text"] = (
        "AI engineer with evaluation frameworks and Python ML systems."
    )
    runtime.state.model_call_count = 6
    valid, invalid = runtime._execute_response_calls(
        NormalizedAssistantMessage(tool_calls=[_tool("tailor_resume", draft, 1)]),
        contract,
    )
    assert valid == 0
    assert invalid == 1
    recovery = json.loads(runtime.conversation[-1]["content"])
    assert recovery["error_category"] == "evidence"
    assert "evaluation" in recovery["error"].casefold()