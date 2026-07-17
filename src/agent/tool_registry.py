"""Exact five-tool model registry and deterministic guarded executor."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.agent.state import (
    AgentPhase,
    AgentRunState,
    StateInvariantError,
    ToolExecutionRecord,
)
from src.models.bundle import CandidateBundle
from src.models.job import Job
from src.models.memory import CandidateMemory
from src.observability.tracing import AgentTracer, SpanContext, TraceContext
from src.services.resume_finalizer import FinalizedResumeResult
from src.tools.cover_letter import CoverLetterPlan, CoverLetterResult, cover_letter_tool
from src.tools.filtering import FilteringResult, filtering_tool
from src.tools.fit_analysis import FitAnalysisResult, fit_analysis_tool
from src.tools.resume_tailoring import (
    ResumeEditPlan,
    ResumeTailoringResult,
    resume_tailoring_tool,
)
from src.tools.scoring import JobScore, ScoringResult, scoring_tool


class ToolRegistryError(Exception):
    """Base error for tool schema, validation, or execution failures."""


class ToolArgumentsError(ToolRegistryError):
    """Raised when model-supplied arguments fail their typed schema."""


class ToolExecutionError(ToolRegistryError):
    """Raised when an existing assignment tool rejects a requested call."""


_PRIVATE_REASONING_PATTERNS = (
    "chain of thought",
    "chain-of-thought",
    "hidden reasoning",
    "private reasoning",
    "internal monologue",
    "my internal thoughts",
)


class _DecisionArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision_summary: str = Field(min_length=1, max_length=500)

    @field_validator("decision_summary")
    @classmethod
    def validate_decision_summary(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("decision_summary must be nonempty")
        lower = value.casefold()
        if any(pattern in lower for pattern in _PRIVATE_REASONING_PATTERNS):
            raise ValueError(
                "decision_summary must not request or reveal hidden chain-of-thought"
            )
        return value


class FilteringCallArguments(_DecisionArguments):
    """Arguments for assignment tool #1."""


class ScoringCallArguments(_DecisionArguments):
    """Arguments for assignment tool #2."""


class FitAnalysisCallArguments(_DecisionArguments):
    """Arguments for assignment tool #3."""

    job_id: str = Field(min_length=1)


class ResumeTailoringCallArguments(_DecisionArguments):
    """Arguments for assignment tool #4."""

    job_id: str = Field(min_length=1)
    edit_plan: ResumeEditPlan


class CoverLetterCallArguments(_DecisionArguments):
    """Arguments for assignment tool #5."""

    job_id: str = Field(min_length=1)
    plan: CoverLetterPlan


class ToolExecutionOutcome(BaseModel):
    """Successful result returned to the runtime and model conversation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool_name: str
    message_payload: dict[str, Any]
    result: Any
    record: ToolExecutionRecord


class AssignmentToolRegistry:
    """Typed schemas and executor for exactly five assignment-level tools."""

    TOOL_MODELS: ClassVar[dict[str, type[_DecisionArguments]]] = {
        "filter_jobs": FilteringCallArguments,
        "score_jobs": ScoringCallArguments,
        "analyze_fit": FitAnalysisCallArguments,
        "tailor_resume": ResumeTailoringCallArguments,
        "generate_cover_letter": CoverLetterCallArguments,
    }
    _DESCRIPTIONS: ClassVar[dict[str, str]] = {
        "filter_jobs": "Filter all loaded jobs with deterministic candidate preferences.",
        "score_jobs": "Deterministically score accepted jobs and select the Top 3.",
        "analyze_fit": "Analyze one deterministic Top 3 job using candidate evidence.",
        "tailor_resume": "Apply a supplied evidence-grounded resume edit plan.",
        "generate_cover_letter": "Generate one evidence-grounded one-page cover letter.",
    }

    def __init__(
        self,
        *,
        state: AgentRunState,
        jobs: list[Job],
        bundle: CandidateBundle,
        memory: CandidateMemory,
        base_resume_tex_path: Path,
        base_resume_pdf_path: Path,
        run_workspace: Path,
        tracer: AgentTracer,
        trace: TraceContext,
    ) -> None:
        self.state = state
        self.jobs = list(jobs)
        self.bundle = bundle
        self.memory = memory
        self.base_resume_tex_path = base_resume_tex_path.resolve()
        self.base_resume_pdf_path = base_resume_pdf_path.resolve()
        self.run_workspace = run_workspace.resolve()
        self.run_workspace.mkdir(parents=True, exist_ok=True)
        self.tracer = tracer
        self.trace = trace
        self._jobs_by_id = {job.job_id: job for job in jobs}

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self.TOOL_MODELS)

    @classmethod
    def model_schemas(cls) -> list[dict[str, Any]]:
        """Build Ollama function schemas directly from typed Pydantic models."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": cls._DESCRIPTIONS[name],
                    "parameters": model.model_json_schema(),
                },
            }
            for name, model in cls.TOOL_MODELS.items()
        ]

    def parse_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> _DecisionArguments:
        model = self.TOOL_MODELS.get(tool_name)
        if model is None:
            raise ToolArgumentsError(
                f"Unknown model-callable tool {tool_name!r}; valid tools are "
                f"{list(self.TOOL_MODELS)}"
            )
        try:
            return model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolArgumentsError(
                f"Invalid arguments for {tool_name}: {exc}"
            ) from exc

    def set_memory(self, memory: CandidateMemory) -> None:
        if memory.candidate_id != self.bundle.profile.candidate_id:
            raise ToolRegistryError("Updated memory candidate_id does not match bundle")
        self.memory = memory

    def _job(self, job_id: str) -> Job:
        try:
            return self._jobs_by_id[job_id]
        except KeyError as exc:
            raise StateInvariantError(f"Unknown loaded job ID {job_id!r}") from exc

    def _score(self, job_id: str) -> JobScore:
        if self.state.scoring_result is None:
            raise StateInvariantError("Deterministic scores are not available")
        return next(
            item
            for item in self.state.scoring_result.top_3
            if item.job_id == job_id
        )

    @staticmethod
    def _safe_job_folder(job_id: str) -> str:
        return "job-" + re.sub(r"[^A-Za-z0-9_-]", "_", job_id)

    def _domain_span_name(self, tool_name: str, job_id: str | None, round_: int) -> str:
        if tool_name == "filter_jobs":
            return "filtering"
        if tool_name == "score_jobs":
            return "scoring"
        if tool_name == "analyze_fit":
            return f"fit_analysis:{job_id}"
        if tool_name == "tailor_resume":
            return f"resume_tailoring:{job_id}:r{round_}"
        return f"cover_letter:{job_id}"

    def execute(
        self,
        tool_name: str,
        raw_arguments: dict[str, Any],
        *,
        tool_call_id: str | None = None,
        revision_round: int = 0,
        review_feedback: str | None = None,
        trace_parent: TraceContext | SpanContext | None = None,
    ) -> ToolExecutionOutcome:
        """Validate state, invoke one existing tool, then commit valid state."""
        arguments = self.parse_arguments(tool_name, raw_arguments)
        job_id = getattr(arguments, "job_id", None)
        self.state.validate_tool_call(
            tool_name,
            job_id=job_id,
            revision_round=revision_round,
            review_feedback=review_feedback,
        )
        phase_before = self.state.phase
        parent = trace_parent or self.trace
        span_input = arguments.model_dump(mode="json")
        span_name = self._domain_span_name(tool_name, job_id, revision_round)
        try:
            with self.tracer.span(
                parent,
                span_name,
                input=span_input,
                metadata={"assignment_tool": tool_name},
                observation_type="tool",
            ) as span:
                result, payload = self._invoke(
                    tool_name,
                    arguments,
                    revision_round=revision_round,
                    review_feedback=review_feedback,
                )
                span.set_output(payload)
        except (StateInvariantError, ToolRegistryError):
            raise
        except Exception as exc:
            raise ToolExecutionError(
                f"{tool_name} rejected the requested call: {exc}"
            ) from exc

        self._commit(tool_name, result)
        self.state.tool_call_count += 1
        record = ToolExecutionRecord(
            sequence=len(self.state.tool_execution_history) + 1,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            job_id=job_id,
            decision_summary=arguments.decision_summary,
            revision_round=revision_round if tool_name == "tailor_resume" else None,
            phase_before=phase_before,
            phase_after=self.state.phase,
            result_summary=payload,
        )
        self.state.tool_execution_history.append(record)
        return ToolExecutionOutcome(
            tool_name=tool_name,
            message_payload=payload,
            result=result,
            record=record,
        )

    def _invoke(
        self,
        tool_name: str,
        arguments: _DecisionArguments,
        *,
        revision_round: int,
        review_feedback: str | None,
    ) -> tuple[Any, dict[str, Any]]:
        if tool_name == "filter_jobs":
            result = filtering_tool(self.jobs, self.bundle.profile)
            return result, {
                "status": "ok",
                "accepted_count": result.accepted_count,
                "rejected_count": result.rejected_count,
                "accepted_job_ids": [item.job_id for item in result.accepted_jobs],
            }
        if tool_name == "score_jobs":
            assert self.state.filtering_result is not None
            result = scoring_tool(
                self.state.filtering_result.accepted_jobs,
                self.bundle,
                self.memory,
            )
            if len(result.top_3) != 3:
                raise ToolExecutionError(
                    "Scoring must produce exactly three Top 3 jobs for this workflow"
                )
            return result, {
                "status": "ok",
                "formula": result.formula_description,
                "top_3": [
                    {
                        "job_id": item.job_id,
                        "rank": item.rank,
                        "final_score": item.final_score,
                        "matched_required_skills": item.matched_required_skills,
                        "unmatched_required_skills": item.unmatched_required_skills,
                    }
                    for item in result.top_3
                ],
            }
        if tool_name == "analyze_fit":
            assert isinstance(arguments, FitAnalysisCallArguments)
            job = self._job(arguments.job_id)
            result = fit_analysis_tool(
                job,
                self._score(job.job_id),
                self.bundle,
                self.memory,
                self.base_resume_tex_path,
            )
            return result, {
                "status": "ok",
                "job_id": result.job_id,
                "score_rank": result.score_rank,
                "formatted_text": result.formatted_text,
                "core_skills": result.core_skills.model_dump(mode="json"),
                "project_swap": (
                    result.projects.swap_suggestion.model_dump(mode="json")
                    if result.projects.swap_suggestion
                    else None
                ),
                "tailoring_actions": [
                    item.model_dump(mode="json") for item in result.tailoring_actions
                ],
            }
        if tool_name == "tailor_resume":
            assert isinstance(arguments, ResumeTailoringCallArguments)
            job = self._job(arguments.job_id)
            output_dir = (
                self.run_workspace
                / "drafts"
                / self._safe_job_folder(job.job_id)
            )
            result = resume_tailoring_tool(
                job,
                self._score(job.job_id),
                self.state.fit_analyses[job.job_id],
                self.bundle,
                self.memory,
                self.base_resume_tex_path,
                self.base_resume_pdf_path,
                output_dir,
                arguments.edit_plan,
                revision_round=revision_round,
                review_feedback=review_feedback,
            )
            return result, {
                "status": "ok",
                "job_id": result.job_id,
                "revision_round": result.revision_round,
                "change_count": result.change_count,
                "page_count": result.page_count,
                "draft_tex_path": str(result.draft_tex_path),
                "draft_pdf_path": str(result.draft_pdf_path),
                "change_log_path": str(result.change_log_path),
            }
        assert isinstance(arguments, CoverLetterCallArguments)
        job = self._job(arguments.job_id)
        finalized: FinalizedResumeResult = self.state.finalized_resumes[job.job_id]
        result = cover_letter_tool(
            job,
            self._score(job.job_id),
            self.state.fit_analyses[job.job_id],
            self.bundle,
            self.memory,
            finalized,
            finalized.destination_dir,
            arguments.plan,
        )
        return result, {
            "status": "ok",
            "job_id": result.job_id,
            "page_count": result.page_count,
            "paragraph_count": result.paragraph_count,
            "skill_count": result.skill_count,
            "tex_path": str(result.tex_path),
            "pdf_path": str(result.pdf_path),
            "evidence_log_path": str(result.evidence_log_path),
        }

    def _commit(self, tool_name: str, result: Any) -> None:
        if tool_name == "filter_jobs":
            assert isinstance(result, FilteringResult)
            self.state.filtering_result = result
            self.state.phase = AgentPhase.FILTERED
        elif tool_name == "score_jobs":
            assert isinstance(result, ScoringResult)
            self.state.scoring_result = result
            self.state.top_3_job_ids = [item.job_id for item in result.top_3]
            self.state.phase = AgentPhase.SCORED
        elif tool_name == "analyze_fit":
            assert isinstance(result, FitAnalysisResult)
            self.state.fit_analyses[result.job_id] = result
            self.state.phase = (
                AgentPhase.TAILORING
                if len(self.state.fit_analyses) == 3
                else AgentPhase.FIT_ANALYSIS
            )
        elif tool_name == "tailor_resume":
            assert isinstance(result, ResumeTailoringResult)
            self.state.draft_resumes[result.job_id] = result
            if result.revision_round == 0:
                self.state.phase = AgentPhase.TAILORING
        else:
            assert isinstance(result, CoverLetterResult)
            self.state.cover_letters[result.job_id] = result
            self.state.phase = AgentPhase.COVER_LETTERS


__all__ = [
    "AssignmentToolRegistry",
    "CoverLetterCallArguments",
    "FilteringCallArguments",
    "FitAnalysisCallArguments",
    "ResumeTailoringCallArguments",
    "ScoringCallArguments",
    "ToolArgumentsError",
    "ToolExecutionError",
    "ToolExecutionOutcome",
    "ToolRegistryError",
]
