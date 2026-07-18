"""The one production LLM runtime and its continuous tool-calling loop."""

from __future__ import annotations

import copy
import json
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.agent.client import (
    ChatModelClient,
    NormalizedAssistantMessage,
    NormalizedToolCall,
    OllamaChatModelClient,
)
from src.agent.prompts import (
    COVER_LETTER_PLAN_LIMITS,
    SYSTEM_PROMPT,
    TAILOR_RESUME_ARGUMENT_TEMPLATE,
    TAILOR_RESUME_CONSTRAINTS,
    TAILOR_RESUME_PLAN_LIMITS,
)
from src.agent.state import (
    AgentPhase,
    AgentRunResult,
    AgentRunState,
    InvalidToolAttempt,
    StateInvariantError,
)
from src.agent.tool_registry import (
    AssignmentToolRegistry,
    ToolArgumentsError,
    ToolExecutionError,
    ToolRegistryError,
)
from src.config import AppConfig, load_config
from src.models.candidate import CandidateProfile, ExperienceBullet
from src.models.memory import CandidateMemory
from src.observability.tracing import (
    AgentTracer,
    SpanContext,
    TraceContext,
    build_agent_tracer,
)
from src.services.candidate_loader import load_candidate_bundle
from src.services.jobs_loader import load_jobs
from src.services.memory_loader import load_memory
from src.services.output_writer import (
    write_fit_analysis_files,
    write_job_details,
)
from src.workflow.human_review import (
    ReviewDecisionProvider,
    run_human_review_session,
)

MAX_MODEL_CALLS = 40
MAX_TOOL_CALLS = 60
MAX_CONSECUTIVE_INVALID_TURNS = 3

_TAILOR_DEFAULT_JOB_CITATION_FIELD = "required_skills_raw"
_TAILOR_RECOMMENDED_JOB_SOURCE_FIELDS = (
    "required_skills_raw",
    "job_description",
    "title",
    "experience_requirement_raw",
)
_TAILOR_INVALID_JOB_SOURCE_FIELDS = (
    "aligned_skills",
    "job_posting",
    "requirements",
    "skills",
)
_TAILOR_CANDIDATE_SUMMARY_FIELD = "experience"
_CITATION_ERROR_MARKERS = (
    "unknown job citation field",
    "unknown candidate profile source id",
    "unknown candidate profile field",
    "unknown experience bullet id",
    "unknown experience bullet field",
    "actual evidence id",
    "requires its bullet citation",
    "requires job-posting",
    "citation",
)


class AgentRuntimeError(Exception):
    """Base error for the one runtime."""


class AgentLoopLimitError(AgentRuntimeError):
    """Raised when model/tool safety limits are reached."""


class AgentModelResponseError(AgentRuntimeError):
    """Raised when the model cannot provide a usable workflow action."""


class JobSearchAgentRuntime:
    """The application's only production LLM-based agent."""

    def __init__(
        self,
        *,
        client: ChatModelClient,
        review_decision_provider: ReviewDecisionProvider,
        config: AppConfig,
        jobs_path: Path,
        profile_path: Path,
        portfolio_path: Path,
        evidence_path: Path,
        memory_path: Path,
        base_resume_tex_path: Path,
        base_resume_pdf_path: Path,
        run_workspace: Path,
        final_output_root: Path,
        tracer: AgentTracer | None = None,
        run_id: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self.review_decision_provider = review_decision_provider
        self.config = config
        self.jobs_path = jobs_path
        self.profile_path = profile_path
        self.portfolio_path = portfolio_path
        self.evidence_path = evidence_path
        self.memory_path = memory_path
        self.base_resume_tex_path = base_resume_tex_path
        self.base_resume_pdf_path = base_resume_pdf_path
        self.run_workspace = run_workspace.resolve()
        self.final_output_root = final_output_root.resolve()
        self.tracer = tracer or build_agent_tracer(config)
        self.run_id = run_id or f"run-{uuid.uuid4().hex}"
        self.progress_callback = progress_callback

        self.state: AgentRunState | None = None
        self.registry: AssignmentToolRegistry | None = None
        self.trace: TraceContext | None = None
        self.conversation: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self._requested_tool_call_count = 0
        self._last_invalid_signature: str | None = None
        self._output_folders: dict[str, Path] = {}
        self._conversation_compacted = False

    @staticmethod
    def _phase_needs_compaction(phase: str) -> bool:
        return phase in {"resume_tailoring", "resume_revision", "cover_letter"}

    def _contract_for_model(self, contract: dict[str, Any]) -> dict[str, Any]:
        """Return a model-facing contract without recovery-only fields."""
        return {
            key: value
            for key, value in contract.items()
            if key != "exact_tailor_resume_structural_template"
        }

    def _build_workflow_checkpoint(self, contract: dict[str, Any]) -> dict[str, Any]:
        assert self.state is not None
        state = self.state
        completed_phases: list[str] = []
        if state.filtering_result is not None:
            completed_phases.append("filtering")
        if state.scoring_result is not None:
            completed_phases.append("scoring")
        if state.fit_analyses:
            completed_phases.append("fit_analysis")
        if state.draft_resumes:
            completed_phases.append("resume_tailoring")
        if state.human_review is not None:
            completed_phases.append("human_review")
        if state.cover_letters:
            completed_phases.append("cover_letter")

        checkpoint: dict[str, Any] = {
            "type": "workflow_checkpoint",
            "completed_phases": completed_phases,
            "filtered": state.filtering_result is not None,
            "scored": state.scoring_result is not None,
            "top_3_job_ids": list(state.top_3_job_ids),
            "top_3_ranks": {
                job_id: self._target_rank(job_id) for job_id in state.top_3_job_ids
            },
            "fit_analysis_job_ids": list(state.fit_analyses),
            "draft_resume_job_ids": list(state.draft_resumes),
            "finalized_resume_job_ids": list(state.finalized_resumes),
            "cover_letter_job_ids": list(state.cover_letters),
            "human_review_completed": bool(
                state.human_review and state.human_review.completed
            ),
            "pause_count": state.pause_count,
            "next_action_contract": self._contract_for_model(contract),
            "instruction": [
                "Return exactly one tool call.",
                "Do not return prose-only content.",
                "Do not call a different tool.",
                "Do not target another job.",
            ],
        }
        target_context = contract.get("target_context")
        if target_context is not None:
            checkpoint["target_context"] = target_context
        return checkpoint

    def _apply_conversation_checkpoint(self, contract: dict[str, Any]) -> None:
        self.conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    self._build_workflow_checkpoint(contract),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        self._conversation_compacted = True

    def _empty_tool_call_error(
        self,
        response: NormalizedAssistantMessage,
    ) -> tuple[str, bool]:
        if (
            not response.tool_calls
            and response.done_reason
            and "length" in response.done_reason.casefold()
        ):
            return (
                "Model response reached the generation limit before completing "
                "a tool call",
                True,
            )
        return (
            "Exactly one tool call is required; received "
            f"{len(response.tool_calls)}",
            False,
        )

    def _report_no_tool_call_progress(
        self,
        response: NormalizedAssistantMessage,
        *,
        length_limited: bool,
    ) -> None:
        assert self.state is not None
        self._report_progress(
            f"Model call {self.state.model_call_count}: "
            "no complete tool call returned"
        )
        if length_limited:
            self._report_progress("Model completion: length limit")

    def _report_citation_rejection_progress(self) -> None:
        assert self.state is not None
        self._report_progress(
            f"Model call {self.state.model_call_count}: "
            "tool call rejected by validation"
        )
        self._report_progress("Validation category: citation")

    @staticmethod
    def _is_citation_error(error: str) -> bool:
        lowered = error.casefold()
        return any(marker in lowered for marker in _CITATION_ERROR_MARKERS)

    def _editable_tailoring_bullets(self) -> list[ExperienceBullet]:
        assert self.registry is not None
        profile = self.registry.bundle.profile
        bullets: list[ExperienceBullet] = []
        for experience in profile.experience:
            if not experience.is_primary_role:
                continue
            for bullet in experience.bullets:
                if bullet.editable_for_job_tailoring:
                    bullets.append(bullet)
        return bullets

    @staticmethod
    def _bullet_evidence_id(bullet: ExperienceBullet) -> str | None:
        if not bullet.evidence_ids:
            return None
        return bullet.evidence_ids[0]

    def _job_posting_citation(
        self,
        job_id: str,
        *,
        source_field: str = _TAILOR_DEFAULT_JOB_CITATION_FIELD,
    ) -> dict[str, Any]:
        return {
            "source_type": "job_posting",
            "source_id": job_id,
            "source_field": source_field,
            "evidence_id": None,
        }

    def _candidate_profile_citation(
        self,
        *,
        source_field: str = _TAILOR_CANDIDATE_SUMMARY_FIELD,
    ) -> dict[str, Any]:
        assert self.registry is not None
        return {
            "source_type": "candidate_profile",
            "source_id": self.registry.bundle.profile.candidate_id,
            "source_field": source_field,
            "evidence_id": None,
        }

    def _experience_bullet_citation(self, bullet: ExperienceBullet) -> dict[str, Any]:
        return {
            "source_type": "experience_bullet",
            "source_id": bullet.bullet_id,
            "source_field": "text",
            "evidence_id": self._bullet_evidence_id(bullet),
        }

    def _build_citation_contract(self, job_id: str) -> dict[str, Any]:
        job_citation = self._job_posting_citation(job_id)
        candidate_citation = self._candidate_profile_citation()
        bullet_required_citations: dict[str, list[dict[str, Any]]] = {}
        for bullet in self._editable_tailoring_bullets()[:2]:
            bullet_required_citations[bullet.bullet_id] = [
                self._experience_bullet_citation(bullet),
                copy.deepcopy(job_citation),
            ]
        return {
            "valid_job_source_fields": list(_TAILOR_RECOMMENDED_JOB_SOURCE_FIELDS),
            "invalid_job_source_fields": list(_TAILOR_INVALID_JOB_SOURCE_FIELDS),
            "valid_candidate_profile_source_fields": sorted(
                CandidateProfile.model_fields
            ),
            "default_job_citation": job_citation,
            "summary_required_citations": [
                copy.deepcopy(job_citation),
                copy.deepcopy(candidate_citation),
            ],
            "bullet_required_citations": bullet_required_citations,
            "copy_identity_fields_exactly": [
                "source_type",
                "source_id",
                "source_field",
                "evidence_id",
            ],
            "instructions": [
                "Copy source_type, source_id, source_field, and evidence_id exactly.",
                "You may add supported_claim, but must not alter identity fields.",
                "Never convert source types into source fields.",
                "Never use an evidence ID as candidate_profile source_id.",
            ],
        }

    def _build_citation_recovery_contract(self, job_id: str) -> dict[str, Any]:
        contract = self._build_citation_contract(job_id)
        return {
            "valid_job_source_fields": contract["valid_job_source_fields"],
            "invalid_job_source_fields": contract["invalid_job_source_fields"],
            "default_job_citation": contract["default_job_citation"],
            "summary_required_citations": contract["summary_required_citations"],
            "bullet_required_citations": contract["bullet_required_citations"],
            "copy_identity_fields_exactly": contract["copy_identity_fields_exactly"],
            "instruction": contract["instructions"],
        }

    @property
    def model_name(self) -> str:
        return str(getattr(self.client, "model_name", self.config.ollama_model))

    def _report_progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    @staticmethod
    def _phase_progress_message(contract: dict[str, Any]) -> str:
        phase = contract["phase"]
        labels = {
            "filtering": "filtering",
            "scoring": "scoring",
            "fit_analysis": "fit analysis",
            "resume_tailoring": "resume tailoring",
            "resume_revision": "resume revision",
            "cover_letter": "cover letter",
        }
        message = f"Agent phase: {labels.get(phase, phase.replace('_', ' '))}"
        rank = contract.get("target_rank")
        if rank is not None and phase in {
            "fit_analysis",
            "resume_tailoring",
            "resume_revision",
            "cover_letter",
        }:
            message += f" {rank}/3"
        return message

    def _target_rank(self, job_id: str | None) -> int | None:
        assert self.state is not None
        if job_id is None:
            return None
        return self.state.top_3_job_ids.index(job_id) + 1

    def _current_professional_summary(self, job_id: str) -> str:
        assert self.state is not None
        draft = self.state.draft_resumes.get(job_id)
        tex_path = draft.draft_tex_path if draft is not None else self.base_resume_tex_path
        try:
            content = tex_path.read_text(encoding="utf-8")
        except OSError:
            return ""
        match = re.search(
            r"% AGENT-EDIT-TARGET: summary\s*(.*?)"
            r"(?=%----------EDUCATION----------)",
            content,
            flags=re.DOTALL,
        )
        if match is None:
            return ""
        return " ".join(
            line.strip()
            for line in match.group(1).splitlines()
            if line.strip()
        )

    @staticmethod
    def _evidence_ids(value: Any) -> list[str]:
        found: set[str] = set()

        def visit(item: Any) -> None:
            if hasattr(item, "model_dump"):
                item = item.model_dump(mode="json")
            if isinstance(item, dict):
                evidence_id = item.get("evidence_id")
                if isinstance(evidence_id, str) and evidence_id:
                    found.add(evidence_id)
                for nested in item.values():
                    visit(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    visit(nested)

        visit(value)
        return sorted(found)

    def _tailoring_context(
        self,
        job_id: str,
        *,
        revision_feedback: str | None = None,
    ) -> dict[str, Any]:
        assert self.state is not None
        assert self.registry is not None
        analysis = self.state.fit_analyses[job_id]
        job = self.registry._job(job_id)
        profile = self.registry.bundle.profile
        editable_bullets = []
        citation_contract = self._build_citation_contract(job_id)
        for bullet in self._editable_tailoring_bullets():
            editable_bullets.append(
                {
                    "bullet_id": bullet.bullet_id,
                    "text": bullet.text,
                    "evidence_ids": list(bullet.evidence_ids),
                    "required_citations": citation_contract[
                        "bullet_required_citations"
                    ].get(
                        bullet.bullet_id,
                        [],
                    ),
                }
            )
        relevant_skills = [
            *analysis.core_skills.aligned_skills,
            *analysis.core_skills.evidenced_elsewhere_skills,
        ]
        relevant_master_skills: list[dict[str, Any]] = []
        seen_skills: set[str] = set()
        for skill in relevant_skills:
            key = skill.casefold()
            if key in seen_skills:
                continue
            seen_skills.add(key)
            records = self.registry.bundle.get_skill_evidence(skill)
            relevant_master_skills.append(
                {
                    "skill": skill,
                    "evidence_ids": sorted(
                        {
                            record.evidence_id
                            for record in records
                            if record.evidence_id
                        }
                    ),
                }
            )
        return {
            "target_job_id": job_id,
            "rank": self._target_rank(job_id),
            "title": job.title,
            "company": job.company,
            "aligned_skills": analysis.core_skills.aligned_skills,
            "evidenced_elsewhere_skills": (
                analysis.core_skills.evidenced_elsewhere_skills
            ),
            "genuine_gaps": analysis.core_skills.genuine_gaps,
            "project_swap": (
                analysis.projects.swap_suggestion.model_dump(mode="json")
                if analysis.projects.swap_suggestion
                else None
            ),
            "editable_experience_bullets": editable_bullets,
            "current_professional_summary": self._current_professional_summary(job_id),
            "relevant_master_skills": relevant_master_skills,
            "current_memory_facts": [
                {
                    "fact_id": fact.fact_id,
                    "fact_type": fact.fact_type,
                    "statement": fact.statement,
                    "normalized_value": fact.normalized_value,
                    "skill_tags": fact.skill_tags,
                }
                for fact in self.registry.memory.facts
            ],
            "revision_feedback": revision_feedback,
            "citation_contract": citation_contract,
        }

    def _tailor_required_shape(
        self,
        target_job_id: str,
        *,
        expected_swap,
        citation_contract: dict[str, Any],
    ) -> dict[str, Any]:
        summary_citations = copy.deepcopy(
            citation_contract["summary_required_citations"]
        )
        bullet_citations = citation_contract["bullet_required_citations"]
        editable_bullets = self._editable_tailoring_bullets()[:2]
        project_swap: dict[str, Any] | None = None
        if expected_swap is not None:
            project_swap = {
                "remove_project_id": expected_swap.remove_project_id,
                "add_project_id": expected_swap.add_project_id,
                "reason": "<reason>",
                "citations": [
                    copy.deepcopy(citation_contract["default_job_citation"]),
                ],
            }
        return {
            "decision_summary": "<concise explanation>",
            "job_id": target_job_id,
            "edit_plan": {
                "job_id": target_job_id,
                "professional_summary": {
                    "new_text": "<tailored summary>",
                    "reason": "<reason>",
                    "citations": summary_citations,
                },
                "experience_bullet_edits": [
                    {
                        "bullet_id": bullet.bullet_id,
                        "new_text": "<tailored bullet text>",
                        "reason": "<reason>",
                        "citations": copy.deepcopy(
                            bullet_citations[bullet.bullet_id]
                        ),
                    }
                    for bullet in editable_bullets
                ],
                "skill_section_edits": [],
                "project_swap": project_swap,
                "plan_rationale": "<concise rationale>",
            },
        }

    def _cover_letter_context(self, job_id: str) -> dict[str, Any]:
        assert self.state is not None
        assert self.registry is not None
        analysis = self.state.fit_analyses[job_id]
        job = self.registry._job(job_id)
        finalized = self.state.finalized_resumes[job_id]
        return {
            "target_job_id": job_id,
            "rank": self._target_rank(job_id),
            "title": job.title,
            "company": job.company,
            "aligned_skills": analysis.core_skills.aligned_skills,
            "evidenced_elsewhere_skills": (
                analysis.core_skills.evidenced_elsewhere_skills
            ),
            "genuine_gaps": analysis.core_skills.genuine_gaps,
            "approved_resume_revision": finalized.approved_revision_round,
            "current_memory_facts": [
                {
                    "fact_id": fact.fact_id,
                    "fact_type": fact.fact_type,
                    "statement": fact.statement,
                    "normalized_value": fact.normalized_value,
                    "skill_tags": fact.skill_tags,
                }
                for fact in self.registry.memory.facts
            ],
            "relevant_evidence_ids": self._evidence_ids(analysis),
        }

    def _next_action_contract(
        self,
        *,
        revision_job_id: str | None = None,
        revision_round: int = 0,
        revision_feedback: str | None = None,
    ) -> dict[str, Any]:
        """Return the one deterministic action permitted on the next model turn."""
        assert self.state is not None
        assert self.registry is not None
        state = self.state
        target_job_id: str | None = None
        target_context: dict[str, Any] | None = None
        initial_draft: bool | None = None

        if revision_job_id is not None:
            allowed_tool = "tailor_resume"
            phase = "resume_revision"
            target_job_id = revision_job_id
            initial_draft = False
        elif state.filtering_result is None:
            allowed_tool = "filter_jobs"
            phase = "filtering"
        elif state.scoring_result is None:
            allowed_tool = "score_jobs"
            phase = "scoring"
        elif len(state.fit_analyses) < 3:
            allowed_tool = "analyze_fit"
            phase = "fit_analysis"
            target_job_id = next(
                job_id
                for job_id in state.top_3_job_ids
                if job_id not in state.fit_analyses
            )
        elif state.human_review is None and len(state.draft_resumes) < 3:
            allowed_tool = "tailor_resume"
            phase = "resume_tailoring"
            target_job_id = next(
                job_id
                for job_id in state.top_3_job_ids
                if job_id not in state.draft_resumes
            )
            initial_draft = True
        elif (
            state.human_review is not None
            and len(state.cover_letters) < len(state.top_3_job_ids)
        ):
            allowed_tool = "generate_cover_letter"
            phase = "cover_letter"
            target_job_id = next(
                job_id
                for job_id in state.top_3_job_ids
                if job_id in state.finalized_resumes
                and job_id not in state.cover_letters
            )
        else:
            raise StateInvariantError("No model action is valid for the current state")

        if allowed_tool == "filter_jobs":
            required_shape: dict[str, Any] = {
                "decision_summary": "<concise explanation>"
            }
            constraints = ["Run deterministic filtering exactly once."]
        elif allowed_tool == "score_jobs":
            required_shape = {"decision_summary": "<concise explanation>"}
            constraints = [
                "Ask Python to calculate scores; never provide or alter a score."
            ]
        elif allowed_tool == "analyze_fit":
            assert target_job_id is not None
            job = self.registry._job(target_job_id)
            required_shape = {
                "decision_summary": "<concise explanation>",
                "job_id": target_job_id,
            }
            constraints = [
                "Use the exact target_job_id.",
                "Analyze only this deterministic Top 3 job.",
            ]
            target_context = {
                "target_job_id": target_job_id,
                "rank": self._target_rank(target_job_id),
                "title": job.title,
                "company": job.company,
            }
        elif allowed_tool == "tailor_resume":
            assert target_job_id is not None
            analysis = state.fit_analyses[target_job_id]
            expected_swap = analysis.projects.swap_suggestion
            target_context = self._tailoring_context(
                target_job_id,
                revision_feedback=revision_feedback,
            )
            required_shape = self._tailor_required_shape(
                target_job_id,
                expected_swap=expected_swap,
                citation_contract=target_context["citation_contract"],
            )
            constraints = [
                item.replace("TARGET_JOB_ID", target_job_id)
                for item in TAILOR_RESUME_CONSTRAINTS
            ] + list(TAILOR_RESUME_PLAN_LIMITS)
        else:
            assert target_job_id is not None
            required_shape = {
                "decision_summary": "<concise explanation>",
                "job_id": target_job_id,
                "plan": {
                    "job_id": target_job_id,
                    "company_hook_phrase": "<evidence-grounded hook>",
                    "company_hook_source_field": "<job field>",
                    "body_paragraphs": ["<1 or 2 CoverLetterParagraph>"],
                    "skills": ["<3 to 8 CoverLetterSkillItem>"],
                    "closing_sentence": "<concise closing>",
                    "plan_rationale": "<concise rationale>",
                },
            }
            constraints = [
                "Outer job_id and plan.job_id must equal the target_job_id.",
                "Use only candidate-side evidence for candidate claims.",
            ] + list(COVER_LETTER_PLAN_LIMITS)
            target_context = self._cover_letter_context(target_job_id)

        return {
            "phase": phase,
            "allowed_tool": allowed_tool,
            "target_job_id": target_job_id,
            "target_rank": self._target_rank(target_job_id),
            "initial_draft": initial_draft,
            "revision_round": revision_round if revision_job_id else None,
            "required_argument_shape": required_shape,
            "constraints": constraints,
            "target_context": target_context,
        }

    def run(self) -> AgentRunResult:
        """Load inputs, run one continuous conversation, and package outputs."""
        self.run_workspace.mkdir(parents=True, exist_ok=True)
        self.final_output_root.mkdir(parents=True, exist_ok=True)
        self.trace = self.tracer.start_trace(
            "agent_run",
            run_id=self.run_id,
            input={
                "jobs_path": self.jobs_path,
                "profile_path": self.profile_path,
                "portfolio_path": self.portfolio_path,
                "evidence_path": self.evidence_path,
                "memory_path": self.memory_path,
                "final_output_root": self.final_output_root,
            },
            metadata={
                "model": self.model_name,
                "system_prompt": SYSTEM_PROMPT,
                "tool_names": list(AssignmentToolRegistry.TOOL_MODELS),
                "ollama_num_ctx": self.config.ollama_num_ctx,
                "ollama_temperature": self.config.ollama_temperature,
            },
        )
        failure: str | None = None
        loop_error: AgentLoopLimitError | None = None
        try:
            self._load_inputs()
            self._reasoning_loop()
        except AgentLoopLimitError as exc:
            failure = str(exc)
            loop_error = exc
            if self.state is not None:
                self.state.phase = AgentPhase.FAILED
                self.state.failure_reason = failure
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if self.state is not None:
                self.state.phase = AgentPhase.FAILED
                self.state.failure_reason = failure
        provisional_result = self._build_result(failure)
        self.tracer.end_trace(
            self.trace,
            output=provisional_result.model_dump(mode="json"),
            error=failure,
        )
        self.tracer.flush()
        if loop_error is not None:
            raise loop_error
        return self._build_result(failure)

    def _load_inputs(self) -> None:
        assert self.trace is not None
        with self.tracer.span(
            self.trace,
            "input_loading",
            input={
                "jobs": self.jobs_path,
                "candidate": [
                    self.profile_path,
                    self.portfolio_path,
                    self.evidence_path,
                ],
                "memory": self.memory_path,
            },
        ) as span:
            jobs = load_jobs(self.jobs_path)
            bundle = load_candidate_bundle(
                self.profile_path,
                self.portfolio_path,
                self.evidence_path,
            )
            memory = load_memory(self.memory_path, bundle.profile.candidate_id)
            if not self.base_resume_tex_path.is_file():
                raise AgentRuntimeError(
                    f"Base resume LaTeX not found: {self.base_resume_tex_path}"
                )
            if not self.base_resume_pdf_path.is_file():
                raise AgentRuntimeError(
                    f"Base resume PDF not found: {self.base_resume_pdf_path}"
                )
            self.state = AgentRunState(
                run_id=self.run_id,
                candidate_id=bundle.profile.candidate_id,
                loaded_memory_candidate_id=memory.candidate_id,
                loaded_memory_fact_ids=[item.fact_id for item in memory.facts],
            )
            self.registry = AssignmentToolRegistry(
                state=self.state,
                jobs=jobs,
                bundle=bundle,
                memory=memory,
                base_resume_tex_path=self.base_resume_tex_path,
                base_resume_pdf_path=self.base_resume_pdf_path,
                run_workspace=self.run_workspace,
                tracer=self.tracer,
                trace=self.trace,
            )
            span.set_output(
                {
                    "job_count": len(jobs),
                    "candidate_id": bundle.profile.candidate_id,
                    "memory_fact_ids": [item.fact_id for item in memory.facts],
                }
            )

    def _reasoning_loop(self) -> None:
        assert self.state is not None
        assert self.registry is not None
        while not self.state.completed:
            if self.state.can_start_human_review():
                self._run_human_review()
                continue
            if (
                len(self.state.cover_letters) == 3
                and set(self.state.cover_letters) == set(self.state.top_3_job_ids)
            ):
                self.state.mark_completed()
                break
            contract = self._next_action_contract()
            if not self._conversation_ends_with_invalid_recovery():
                if self._phase_needs_compaction(contract["phase"]):
                    self._apply_conversation_checkpoint(contract)
                else:
                    self._append_state_snapshot(contract)
            response = self._call_model(self.trace, contract)
            self._append_assistant_message(response)
            valid_count, invalid_count = self._execute_response_calls(
                response,
                contract,
            )
            if valid_count:
                self.state.consecutive_invalid_call_count = 0
            elif invalid_count:
                self.state.consecutive_invalid_call_count += 1
                if (
                    self.state.consecutive_invalid_call_count
                    >= MAX_CONSECUTIVE_INVALID_TURNS
                ):
                    raise AgentLoopLimitError(
                        "Maximum consecutive invalid tool-call turns reached"
                    )

    def _conversation_ends_with_invalid_recovery(self) -> bool:
        if not self.conversation:
            return False
        content = self.conversation[-1].get("content")
        if not isinstance(content, str):
            return False
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return False
        return (
            isinstance(payload, dict)
            and payload.get("status") == "invalid_tool_call"
        )

    def _append_state_snapshot(self, contract: dict[str, Any]) -> None:
        assert self.state is not None
        self.conversation.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "type": "current_state",
                        "state": self.state.snapshot(),
                        "next_action_contract": contract,
                        "instruction": [
                            "Return exactly one tool call.",
                            "Do not return prose-only content.",
                            "Do not call a different tool.",
                            "Do not target another job.",
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )

    def _call_model(
        self,
        parent: TraceContext | SpanContext | None,
        contract: dict[str, Any],
    ) -> NormalizedAssistantMessage:
        assert self.state is not None
        assert self.registry is not None
        assert self.trace is not None
        if self.state.model_call_count >= MAX_MODEL_CALLS:
            raise AgentLoopLimitError("Maximum model-call count reached")
        schemas = self.registry.model_schemas([contract["allowed_tool"]])
        call_number = self.state.model_call_count + 1
        self._report_progress(self._phase_progress_message(contract))
        self._report_progress(
            f"Model call {call_number}/{MAX_MODEL_CALLS}: "
            f"waiting for {self.model_name}"
        )
        with self.tracer.span(
            parent or self.trace,
            f"chat_model:{call_number}",
            input={
                "system_prompt": SYSTEM_PROMPT,
                "messages": self.conversation,
                "tools": schemas,
                "model": self.model_name,
                "configuration": {
                    "think": False,
                    "stream": False,
                    "num_ctx": self.config.ollama_num_ctx,
                    "num_predict": self.config.ollama_num_predict,
                    "temperature": self.config.ollama_temperature,
                    "request_timeout_seconds": (
                        self.config.ollama_request_timeout_seconds
                    ),
                },
            },
            metadata={"model_call_number": call_number},
            observation_type="generation",
        ) as span:
            response = self.client.chat(self.conversation, schemas)
            self._report_progress(f"Model call {call_number}: response received")
            self.state.model_call_count = call_number
            span.set_output(response.model_dump(mode="json"))
        return response

    def _append_assistant_message(
        self,
        response: NormalizedAssistantMessage,
    ) -> None:
        self.conversation.append(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        **({"id": call.id} if call.id else {}),
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in response.tool_calls
                ],
            }
        )

    def _execute_response_calls(
        self,
        response: NormalizedAssistantMessage,
        contract: dict[str, Any],
    ) -> tuple[int, int]:
        assert self.state is not None
        assert self.registry is not None
        self._requested_tool_call_count += len(response.tool_calls)
        if self._requested_tool_call_count > MAX_TOOL_CALLS:
            raise AgentLoopLimitError("Maximum requested tool-call count reached")
        if len(response.tool_calls) != 1:
            error, length_limited = self._empty_tool_call_error(response)
            call = response.tool_calls[0] if response.tool_calls else None
            if not response.tool_calls:
                self._report_no_tool_call_progress(
                    response,
                    length_limited=length_limited,
                )
            self._record_invalid(tool_call=call, error=error)
            self._append_invalid_message(call, error, contract)
            return 0, 1

        call = response.tool_calls[0]
        try:
            self._validate_call_for_contract(call, contract)
            baseline_arguments = self.registry.parse_arguments(
                call.name,
                call.arguments,
            )
            self.state.validate_tool_call(
                call.name,
                job_id=getattr(baseline_arguments, "job_id", None),
            )
            self._execute_model_tool_call(call)
        except (StateInvariantError, ToolRegistryError) as exc:
            error = str(exc)
            if self._is_citation_error(error):
                self._report_citation_rejection_progress()
            self._record_invalid(tool_call=call, error=error)
            self._append_invalid_message(call, error, contract)
            return 0, 1
        return 1, 0

    def _validate_call_for_contract(
        self,
        call: NormalizedToolCall,
        contract: dict[str, Any],
    ) -> None:
        allowed_tool = contract["allowed_tool"]
        target_job_id = contract.get("target_job_id")
        if call.name != allowed_tool:
            raise StateInvariantError(
                f"Expected only {allowed_tool!r}; received {call.name!r}"
            )
        if target_job_id is None:
            return
        outer_job_id = call.arguments.get("job_id")
        if outer_job_id != target_job_id:
            raise StateInvariantError(
                "Outer job_id must equal deterministic target "
                f"{target_job_id!r}; received {outer_job_id!r}"
            )
        if allowed_tool == "tailor_resume":
            edit_plan = call.arguments.get("edit_plan")
            nested_job_id = (
                edit_plan.get("job_id")
                if isinstance(edit_plan, dict)
                else None
            )
            if nested_job_id != target_job_id:
                raise StateInvariantError(
                    "edit_plan.job_id must equal deterministic target "
                    f"{target_job_id!r}; received {nested_job_id!r}"
                )
            assert self.state is not None
            expected = self.state.fit_analyses[
                target_job_id
            ].projects.swap_suggestion
            submitted = edit_plan.get("project_swap")
            if expected is None and submitted is not None:
                raise StateInvariantError(
                    f"project_swap must be null for target {target_job_id!r}; "
                    "never copy a swap from another job"
                )
            if expected is not None:
                if not isinstance(submitted, dict) or (
                    submitted.get("remove_project_id")
                    != expected.remove_project_id
                    or submitted.get("add_project_id")
                    != expected.add_project_id
                ):
                    raise StateInvariantError(
                        "project_swap must exactly match target Fit Analysis: "
                        f"remove {expected.remove_project_id!r}, "
                        f"add {expected.add_project_id!r}"
                    )
        elif allowed_tool == "generate_cover_letter":
            plan = call.arguments.get("plan")
            nested_job_id = plan.get("job_id") if isinstance(plan, dict) else None
            if nested_job_id != target_job_id:
                raise StateInvariantError(
                    "plan.job_id must equal deterministic target "
                    f"{target_job_id!r}; received {nested_job_id!r}"
                )

    @staticmethod
    def _argument_diagnostics(
        call: NormalizedToolCall | None,
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        if call is None:
            return {
                "missing_fields": ["tool_call"],
                "extra_fields": [],
                "misplaced_fields": [],
                "replacement_schema_keys": [],
            }
        if contract["allowed_tool"] != "tailor_resume":
            return {
                "missing_fields": [],
                "extra_fields": [],
                "misplaced_fields": [],
                "replacement_schema_keys": [],
            }
        outer_expected = {"decision_summary", "job_id", "edit_plan"}
        plan_expected = {
            "job_id",
            "professional_summary",
            "experience_bullet_edits",
            "skill_section_edits",
            "project_swap",
            "plan_rationale",
        }
        outer_keys = set(call.arguments)
        raw_plan = call.arguments.get("edit_plan")
        plan_keys = set(raw_plan) if isinstance(raw_plan, dict) else set()
        plan_only = plan_expected - {"job_id", "professional_summary"}
        return {
            "missing_fields": sorted(
                {
                    *(f"outer.{key}" for key in outer_expected - outer_keys),
                    *(f"edit_plan.{key}" for key in plan_expected - plan_keys),
                }
            ),
            "extra_fields": sorted(
                {
                    *(f"outer.{key}" for key in outer_keys - outer_expected),
                    *(f"edit_plan.{key}" for key in plan_keys - plan_expected),
                }
            ),
            "misplaced_fields": sorted(outer_keys & plan_only),
            "replacement_schema_keys": sorted(
                plan_keys & {"education", "experience", "projects", "skills"}
            ),
        }

    def _execute_model_tool_call(
        self,
        call: NormalizedToolCall,
        *,
        revision_round: int = 0,
        review_feedback: str | None = None,
        trace_parent: TraceContext | SpanContext | None = None,
    ):
        assert self.registry is not None
        assert self.trace is not None
        with self.tracer.span(
            trace_parent or self.trace,
            f"tool_call:{call.name}",
            input={
                "tool_call_id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            },
            metadata={
                "decision_summary": call.arguments.get("decision_summary", "")
            },
            observation_type="tool",
        ) as span:
            outcome = self.registry.execute(
                call.name,
                call.arguments,
                tool_call_id=call.id,
                revision_round=revision_round,
                review_feedback=review_feedback,
                trace_parent=span,
            )
            span.set_output(outcome.message_payload)
        model_payload = self._compact_model_tool_result(outcome)
        self.conversation.append(
            {
                "role": "tool",
                "tool_name": call.name,
                **({"tool_call_id": call.id} if call.id else {}),
                "content": json.dumps(
                    model_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
        return outcome

    def _compact_model_tool_result(self, outcome) -> dict[str, Any]:
        """Return a bounded conversation payload while retaining full state/trace data."""
        if outcome.tool_name == "score_jobs":
            return {
                "status": "ok",
                "top_3": [
                    {
                        "job_id": item.job_id,
                        "rank": item.rank,
                    }
                    for item in outcome.result.top_3
                ],
            }
        if outcome.tool_name == "analyze_fit":
            result = outcome.result
            return {
                "status": "ok",
                "job_id": result.job_id,
                "rank": result.score_rank,
                "aligned_skills": result.core_skills.aligned_skills,
                "evidenced_elsewhere_skills": (
                    result.core_skills.evidenced_elsewhere_skills
                ),
                "genuine_gaps": result.core_skills.genuine_gaps,
                "project_swap": (
                    result.projects.swap_suggestion.model_dump(mode="json")
                    if result.projects.swap_suggestion
                    else None
                ),
                "relevant_evidence_ids": self._evidence_ids(result),
            }
        return outcome.message_payload

    def _record_invalid(
        self,
        *,
        tool_call: NormalizedToolCall | None,
        error: str,
    ) -> None:
        assert self.state is not None
        name = tool_call.name if tool_call else "<missing_tool_call>"
        arguments = tool_call.arguments if tool_call else {}
        signature = json.dumps(
            {"name": name, "arguments": arguments},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        self._last_invalid_signature = signature
        self.state.invalid_tool_attempts.append(
            InvalidToolAttempt(
                sequence=len(self.state.invalid_tool_attempts) + 1,
                model_call_number=self.state.model_call_count,
                tool_call_id=tool_call.id if tool_call else None,
                tool_name=name,
                arguments=arguments,
                error=error[:2000],
                valid_state=self.state.snapshot(),
            )
        )

    def _append_invalid_message(
        self,
        tool_call: NormalizedToolCall | None,
        error: str,
        contract: dict[str, Any],
    ) -> None:
        assert self.state is not None
        diagnostics = self._argument_diagnostics(tool_call, contract)
        payload: dict[str, Any] = {
            "status": "invalid_tool_call",
            "requested_tool": tool_call.name if tool_call else None,
            "error": error[:2000],
            "field_diagnostics": diagnostics,
            "target_job_id": contract.get("target_job_id"),
            "allowed_tool": contract["allowed_tool"],
            "required_argument_shape": contract["required_argument_shape"],
            "target_context": contract.get("target_context"),
            "constraints": contract["constraints"],
            "instruction": [
                "Return exactly one tool call.",
                "Do not return prose-only content.",
                "Do not call a different tool.",
                "Do not target another job.",
                "Correct every missing, extra, or misplaced field exactly.",
            ],
        }
        structural_missing = [
            field
            for field in diagnostics["missing_fields"]
            if field != "tool_call"
        ]
        structure_issues = (
            contract["allowed_tool"] == "tailor_resume"
            and (
                structural_missing
                or diagnostics["extra_fields"]
                or diagnostics["misplaced_fields"]
                or diagnostics["replacement_schema_keys"]
            )
        )
        if structure_issues:
            target_job_id = contract.get("target_job_id")
            template = copy.deepcopy(TAILOR_RESUME_ARGUMENT_TEMPLATE)
            if target_job_id:
                template["job_id"] = target_job_id
                template["edit_plan"]["job_id"] = target_job_id
            payload["exact_tailor_resume_structural_template"] = template
        target_job_id = contract.get("target_job_id")
        if (
            contract["allowed_tool"] == "tailor_resume"
            and target_job_id
            and self._is_citation_error(error)
        ):
            payload["citation_recovery_contract"] = (
                self._build_citation_recovery_contract(target_job_id)
            )
            payload["instruction"].append(
                "Copy every supplied citation identity field exactly."
            )
        self.conversation.append(
            {
                "role": "tool" if tool_call else "user",
                **(
                    {
                        "tool_name": tool_call.name,
                        **(
                            {"tool_call_id": tool_call.id}
                            if tool_call.id
                            else {}
                        ),
                    }
                    if tool_call
                    else {}
                ),
                "content": json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )

    def _run_human_review(self) -> None:
        assert self.state is not None
        assert self.registry is not None
        assert self.trace is not None
        self.state.begin_human_review()
        initial_drafts = [
            self.state.draft_resumes[job_id] for job_id in self.state.top_3_job_ids
        ]
        with self.tracer.span(
            self.trace,
            "human_review_pause",
            input={
                "job_ids": self.state.top_3_job_ids,
                "draft_revisions": {
                    item.job_id: item.revision_round for item in initial_drafts
                },
            },
        ) as review_span:

            def traced_provider(pending_drafts, session_state):
                with self.tracer.span(
                    review_span,
                    f"review_decision_round:{session_state.provider_call_count}",
                    input={
                        "pending_job_ids": session_state.pending_job_ids,
                        "pause_count": session_state.pause_count,
                    },
                ) as decision_span:
                    decisions = self.review_decision_provider(
                        pending_drafts, session_state
                    )
                    decision_span.set_output(
                        [
                            item.model_dump(mode="json")
                            if hasattr(item, "model_dump")
                            else str(item)
                            for item in decisions
                        ]
                    )
                    return decisions

            def revision_handler(
                job_id: str,
                previous,
                review_comments: str,
                updated_memory: CandidateMemory,
                next_revision_round: int,
            ):
                self.registry.set_memory(updated_memory)
                with self.tracer.span(
                    review_span,
                    f"revision_call:{job_id}:r{next_revision_round}",
                    input={
                        "job_id": job_id,
                        "review_feedback": review_comments,
                        "memory_fact_ids": [
                            item.fact_id for item in updated_memory.facts
                        ],
                    },
                ) as revision_span:
                    result = self._request_revision(
                        job_id=job_id,
                        review_comments=review_comments,
                        updated_memory=updated_memory,
                        next_revision_round=next_revision_round,
                        trace_parent=revision_span,
                    )
                    revision_span.set_output(
                        {
                            "job_id": result.job_id,
                            "revision_round": result.revision_round,
                            "page_count": result.page_count,
                        }
                    )
                    return result

            result = run_human_review_session(
                initial_drafts=initial_drafts,
                memory=self.registry.memory,
                memory_path=self.memory_path,
                final_output_root=self.final_output_root,
                decision_provider=traced_provider,
                revision_handler=revision_handler,
            )
            with self.tracer.span(
                review_span,
                "memory_write",
                input={"memory_path": self.memory_path},
            ) as memory_span:
                memory_span.set_output(
                    {
                        "learned_fact_ids": result.learned_fact_ids,
                        "final_fact_count": len(result.final_memory.facts),
                    }
                )
            with self.tracer.span(
                review_span,
                "resume_finalization",
                input={"approved_job_ids": result.initial_job_ids},
            ) as finalization_span:
                finalization_span.set_output(
                    {
                        "finalization_count": result.finalization_count,
                        "folders": [
                            item.destination_dir for item in result.finalized_resumes
                        ],
                    }
                )
            self.registry.set_memory(result.final_memory)
            self.state.apply_human_review(result)
            review_span.set_output(
                {
                    "pause_count": result.pause_count,
                    "all_approved": result.all_approved,
                    "learned_fact_ids": result.learned_fact_ids,
                    "revision_count_by_job": result.revision_count_by_job,
                }
            )
        self._package_final_folders()
        self.conversation.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "type": "human_review_outcome",
                        "summary": result.audit_summary,
                        "learned_memory_fact_ids": result.learned_fact_ids,
                        "finalized_job_ids": list(self.state.finalized_resumes),
                        "instruction": (
                            "Continue this same conversation by generating one "
                            "evidence-grounded cover letter for each Top 3 job."
                        ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )

    def _request_revision(
        self,
        *,
        job_id: str,
        review_comments: str,
        updated_memory: CandidateMemory,
        next_revision_round: int,
        trace_parent: SpanContext,
    ):
        assert self.state is not None
        del updated_memory
        contract = self._next_action_contract(
            revision_job_id=job_id,
            revision_round=next_revision_round,
            revision_feedback=review_comments,
        )
        self._apply_conversation_checkpoint(contract)
        invalid_turns = 0
        while True:
            response = self._call_model(trace_parent, contract)
            self._append_assistant_message(response)
            self._requested_tool_call_count += len(response.tool_calls)
            if self._requested_tool_call_count > MAX_TOOL_CALLS:
                raise AgentLoopLimitError(
                    "Maximum requested tool-call count reached"
                )
            if len(response.tool_calls) != 1:
                error, length_limited = self._empty_tool_call_error(response)
                if not response.tool_calls:
                    self._report_no_tool_call_progress(
                        response,
                        length_limited=length_limited,
                    )
                    if length_limited:
                        error = (
                            "Model response reached the generation limit "
                            "before completing a tool call"
                        )
                    else:
                        error = (
                            "A review revision response must contain exactly "
                            "one tailor_resume call"
                        )
                call = response.tool_calls[0] if response.tool_calls else None
                self._record_invalid(tool_call=call, error=error)
                self._append_invalid_message(call, error, contract)
                invalid_turns += 1
            else:
                call = response.tool_calls[0]
                try:
                    self._validate_call_for_contract(call, contract)
                    outcome = self._execute_model_tool_call(
                        call,
                        revision_round=next_revision_round,
                        review_feedback=review_comments,
                        trace_parent=trace_parent,
                    )
                except (StateInvariantError, ToolRegistryError) as exc:
                    error = str(exc)
                    if self._is_citation_error(error):
                        self._report_citation_rejection_progress()
                    self._record_invalid(tool_call=call, error=error)
                    self._append_invalid_message(call, error, contract)
                    invalid_turns += 1
                else:
                    self.state.consecutive_invalid_call_count = 0
                    return outcome.result
            self.state.consecutive_invalid_call_count += 1
            if invalid_turns >= MAX_CONSECUTIVE_INVALID_TURNS:
                raise AgentLoopLimitError(
                    "Maximum invalid revision tool-call turns reached"
                )

    def _package_final_folders(self) -> None:
        assert self.state is not None
        assert self.registry is not None
        assert self.trace is not None
        with self.tracer.span(
            self.trace,
            "output_packaging",
            input={"job_ids": self.state.top_3_job_ids},
        ) as span:
            for job_id in self.state.top_3_job_ids:
                finalized = self.state.finalized_resumes[job_id]
                job = self.registry._job(job_id)
                score = self.registry._score(job_id)
                write_job_details(finalized.destination_dir, job, score)
                write_fit_analysis_files(
                    finalized.destination_dir,
                    self.state.fit_analyses[job_id],
                )
                self._output_folders[job_id] = finalized.destination_dir
            span.set_output(
                {
                    job_id: sorted(path.name for path in folder.iterdir())
                    for job_id, folder in self._output_folders.items()
                }
            )

    def _build_result(self, failure: str | None) -> AgentRunResult:
        state = self.state
        if state is None:
            return AgentRunResult(
                run_id=self.run_id,
                completed=False,
                failure_reason=failure or "Inputs were not loaded",
                model_name=self.model_name,
                model_call_count=0,
                tool_call_count=0,
                invalid_tool_attempt_count=0,
                tool_execution_records=[],
                top_3_job_ids=[],
                top_3_scores={},
                fit_analysis_count=0,
                draft_resume_count=0,
                pause_count=0,
                learned_memory_fact_ids=[],
                finalized_resume_count=0,
                cover_letter_count=0,
                output_folders={},
                trace_id=self.trace.trace_id if self.trace else None,
                trace_url=self.trace.trace_url if self.trace else None,
                state_summary={},
            )
        scores = {
            item.job_id: item.final_score
            for item in (state.scoring_result.top_3 if state.scoring_result else [])
        }
        learned = (
            state.human_review.learned_fact_ids if state.human_review else []
        )
        return AgentRunResult(
            run_id=self.run_id,
            completed=state.completed,
            failure_reason=failure or state.failure_reason,
            model_name=self.model_name,
            model_call_count=state.model_call_count,
            tool_call_count=state.tool_call_count,
            invalid_tool_attempt_count=len(state.invalid_tool_attempts),
            tool_execution_records=state.tool_execution_history,
            top_3_job_ids=state.top_3_job_ids,
            top_3_scores=scores,
            fit_analysis_count=len(state.fit_analyses),
            draft_resume_count=len(state.draft_resumes),
            pause_count=state.pause_count,
            learned_memory_fact_ids=learned,
            finalized_resume_count=len(state.finalized_resumes),
            cover_letter_count=len(state.cover_letters),
            output_folders=self._output_folders,
            trace_id=self.trace.trace_id if self.trace else None,
            trace_url=self.trace.trace_url if self.trace else None,
            state_summary=state.snapshot(),
        )


def run_job_search_agent(
    *,
    review_decision_provider: ReviewDecisionProvider,
    jobs_path: Path = Path("data/AI_ML_Jobs_Dataset_20.csv"),
    profile_path: Path = Path("candidate/profile.json"),
    portfolio_path: Path = Path("candidate/portfolio.json"),
    evidence_path: Path = Path("candidate/evidence_registry.json"),
    memory_path: Path = Path("memory.json"),
    base_resume_tex_path: Path = Path("candidate/sample_resume.tex"),
    base_resume_pdf_path: Path = Path("candidate/sample_resume.pdf"),
    run_workspace: Path = Path(".runtime"),
    final_output_root: Path = Path("outputs"),
    config: AppConfig | None = None,
    client: ChatModelClient | None = None,
    tracer: AgentTracer | None = None,
    run_id: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> AgentRunResult:
    """Convenience entry point for the one production runtime."""
    resolved_config = config or load_config()
    resolved_client = client or OllamaChatModelClient(resolved_config)
    runtime = JobSearchAgentRuntime(
        client=resolved_client,
        review_decision_provider=review_decision_provider,
        config=resolved_config,
        jobs_path=jobs_path,
        profile_path=profile_path,
        portfolio_path=portfolio_path,
        evidence_path=evidence_path,
        memory_path=memory_path,
        base_resume_tex_path=base_resume_tex_path,
        base_resume_pdf_path=base_resume_pdf_path,
        run_workspace=run_workspace,
        final_output_root=final_output_root,
        tracer=tracer,
        run_id=run_id,
        progress_callback=progress_callback,
    )
    return runtime.run()


__all__ = [
    "AgentLoopLimitError",
    "AgentModelResponseError",
    "AgentRuntimeError",
    "JobSearchAgentRuntime",
    "MAX_CONSECUTIVE_INVALID_TURNS",
    "MAX_MODEL_CALLS",
    "MAX_TOOL_CALLS",
    "run_job_search_agent",
]
