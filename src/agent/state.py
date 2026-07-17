"""Typed state and deterministic workflow invariants for the single runtime."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.services.resume_finalizer import FinalizedResumeResult
from src.tools.cover_letter import CoverLetterResult
from src.tools.filtering import FilteringResult
from src.tools.fit_analysis import FitAnalysisResult
from src.tools.resume_tailoring import ResumeTailoringResult
from src.tools.scoring import ScoringResult
from src.workflow.human_review import HumanReviewSessionResult


class StateInvariantError(Exception):
    """Raised before an invalid workflow transition can mutate state."""


class AgentPhase(StrEnum):
    """Deterministic phases of the one runtime."""

    INITIALIZED = "initialized"
    FILTERED = "filtered"
    SCORED = "scored"
    FIT_ANALYSIS = "fit_analysis"
    TAILORING = "tailoring"
    HUMAN_REVIEW = "human_review"
    COVER_LETTERS = "cover_letters"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolExecutionRecord(BaseModel):
    """Bounded audit record for one successful assignment-tool execution."""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    tool_call_id: str | None = None
    tool_name: str
    job_id: str | None = None
    decision_summary: str
    revision_round: int | None = Field(default=None, ge=0, le=2)
    phase_before: AgentPhase
    phase_after: AgentPhase
    result_summary: dict[str, Any]


class InvalidToolAttempt(BaseModel):
    """One rejected model request that did not mutate valid workflow state."""

    model_config = ConfigDict(extra="forbid")

    sequence: int = Field(ge=1)
    model_call_number: int = Field(ge=1)
    tool_call_id: str | None = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    error: str
    valid_state: dict[str, Any]


class AgentRunState(BaseModel):
    """Mutable typed state owned only by ``JobSearchAgentRuntime``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str
    phase: AgentPhase = AgentPhase.INITIALIZED
    candidate_id: str
    loaded_memory_candidate_id: str
    loaded_memory_fact_ids: list[str] = Field(default_factory=list)
    filtering_result: FilteringResult | None = None
    scoring_result: ScoringResult | None = None
    top_3_job_ids: list[str] = Field(default_factory=list)
    fit_analyses: dict[str, FitAnalysisResult] = Field(default_factory=dict)
    draft_resumes: dict[str, ResumeTailoringResult] = Field(default_factory=dict)
    human_review: HumanReviewSessionResult | None = None
    finalized_resumes: dict[str, FinalizedResumeResult] = Field(default_factory=dict)
    cover_letters: dict[str, CoverLetterResult] = Field(default_factory=dict)
    tool_execution_history: list[ToolExecutionRecord] = Field(default_factory=list)
    invalid_tool_attempts: list[InvalidToolAttempt] = Field(default_factory=list)
    model_call_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    consecutive_invalid_call_count: int = Field(default=0, ge=0)
    pause_count: int = Field(default=0, ge=0, le=1)
    completed: bool = False
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_completion(self) -> "AgentRunState":
        if self.completed:
            if (
                len(self.top_3_job_ids) != 3
                or len(self.fit_analyses) != 3
                or len(self.finalized_resumes) != 3
                or len(self.cover_letters) != 3
                or self.pause_count != 1
            ):
                raise ValueError("completed state requires the full Top 3 workflow")
        return self

    def snapshot(self) -> dict[str, Any]:
        """Return a concise state view suitable for model and error messages."""
        return {
            "phase": self.phase.value,
            "filtered": self.filtering_result is not None,
            "scored": self.scoring_result is not None,
            "top_3_job_ids": list(self.top_3_job_ids),
            "fit_analysis_job_ids": list(self.fit_analyses),
            "draft_resume_job_ids": list(self.draft_resumes),
            "human_review_completed": bool(
                self.human_review and self.human_review.completed
            ),
            "finalized_resume_job_ids": list(self.finalized_resumes),
            "cover_letter_job_ids": list(self.cover_letters),
            "pause_count": self.pause_count,
            "completed": self.completed,
        }

    def require_top_3(self, job_id: str | None) -> str:
        if not job_id or job_id not in self.top_3_job_ids:
            raise StateInvariantError(
                f"Job {job_id!r} is not one of the deterministic Top 3"
            )
        return job_id

    def validate_tool_call(
        self,
        tool_name: str,
        *,
        job_id: str | None = None,
        revision_round: int = 0,
        review_feedback: str | None = None,
    ) -> None:
        """Validate a requested call without mutating prior valid state."""
        if self.completed:
            raise StateInvariantError("The run is already complete")
        if tool_name == "filter_jobs":
            if self.filtering_result is not None:
                raise StateInvariantError("Filtering must run exactly once")
            return
        if tool_name == "score_jobs":
            if self.filtering_result is None:
                raise StateInvariantError("Scoring requires completed filtering")
            if self.scoring_result is not None:
                raise StateInvariantError("Scoring must run exactly once")
            return
        if tool_name == "analyze_fit":
            if self.scoring_result is None:
                raise StateInvariantError("Fit Analysis requires completed scoring")
            scoped_job_id = self.require_top_3(job_id)
            if scoped_job_id in self.fit_analyses:
                raise StateInvariantError(
                    f"Fit Analysis already ran for job {scoped_job_id}"
                )
            return
        if tool_name == "tailor_resume":
            if len(self.fit_analyses) != 3:
                raise StateInvariantError(
                    "Resume Tailoring requires all three Fit Analyses"
                )
            scoped_job_id = self.require_top_3(job_id)
            current = self.draft_resumes.get(scoped_job_id)
            if revision_round == 0:
                if current is not None:
                    raise StateInvariantError(
                        f"Initial Resume Tailoring already ran for job {scoped_job_id}"
                    )
                if self.pause_count:
                    raise StateInvariantError(
                        "Initial Resume Tailoring cannot run after Human Review begins"
                    )
                return
            if self.phase != AgentPhase.HUMAN_REVIEW or self.pause_count != 1:
                raise StateInvariantError(
                    "Resume revision is allowed only inside the one Human Review session"
                )
            if current is None or current.revision_round + 1 != revision_round:
                raise StateInvariantError(
                    f"Revision r{revision_round} is not the next revision for job "
                    f"{scoped_job_id}"
                )
            if revision_round not in {1, 2} or not (review_feedback or "").strip():
                raise StateInvariantError(
                    "Resume revisions require round 1 or 2 and exact review feedback"
                )
            return
        if tool_name == "generate_cover_letter":
            scoped_job_id = self.require_top_3(job_id)
            if not self.human_review or not self.human_review.completed:
                raise StateInvariantError(
                    "Cover Letter generation requires completed Human Review"
                )
            if len(self.finalized_resumes) != 3:
                raise StateInvariantError(
                    "Cover Letter generation requires all three finalized resumes"
                )
            if scoped_job_id not in self.finalized_resumes:
                raise StateInvariantError(
                    f"Job {scoped_job_id} has no approved finalized resume"
                )
            if scoped_job_id in self.cover_letters:
                raise StateInvariantError(
                    f"Cover Letter already ran for job {scoped_job_id}"
                )
            return
        raise StateInvariantError(f"Unknown assignment tool {tool_name!r}")

    def can_start_human_review(self) -> bool:
        return (
            len(self.top_3_job_ids) == 3
            and set(self.draft_resumes) == set(self.top_3_job_ids)
            and all(item.revision_round == 0 for item in self.draft_resumes.values())
            and self.pause_count == 0
            and self.human_review is None
        )

    def begin_human_review(self) -> None:
        if not self.can_start_human_review():
            raise StateInvariantError(
                "Human Review requires exactly three distinct revision-0 drafts"
            )
        self.pause_count = 1
        self.phase = AgentPhase.HUMAN_REVIEW

    def apply_human_review(self, result: HumanReviewSessionResult) -> None:
        if self.pause_count != 1 or not result.completed or result.pause_count != 1:
            raise StateInvariantError("Human Review result is not a completed single pause")
        result_ids = {item.job_id for item in result.finalized_resumes}
        if result_ids != set(self.top_3_job_ids):
            raise StateInvariantError(
                "Human Review finalized resumes do not match the Top 3"
            )
        self.human_review = result
        self.finalized_resumes = {
            item.job_id: item for item in result.finalized_resumes
        }
        self.phase = AgentPhase.COVER_LETTERS

    def mark_completed(self) -> None:
        if (
            set(self.fit_analyses) != set(self.top_3_job_ids)
            or set(self.finalized_resumes) != set(self.top_3_job_ids)
            or set(self.cover_letters) != set(self.top_3_job_ids)
            or self.pause_count != 1
        ):
            raise StateInvariantError(
                "Run completion requires three analyses, final resumes, and cover letters"
            )
        self.completed = True
        self.phase = AgentPhase.COMPLETED


class AgentRunResult(BaseModel):
    """Public, bounded summary of one runtime execution."""

    run_id: str
    completed: bool
    failure_reason: str | None
    model_name: str
    model_call_count: int
    tool_call_count: int
    invalid_tool_attempt_count: int
    tool_execution_records: list[ToolExecutionRecord]
    top_3_job_ids: list[str]
    top_3_scores: dict[str, float]
    fit_analysis_count: int
    draft_resume_count: int
    pause_count: int
    learned_memory_fact_ids: list[str]
    finalized_resume_count: int
    cover_letter_count: int
    output_folders: dict[str, Path]
    trace_id: str | None
    trace_url: str | None
    state_summary: dict[str, Any]


__all__ = [
    "AgentPhase",
    "AgentRunResult",
    "AgentRunState",
    "InvalidToolAttempt",
    "StateInvariantError",
    "ToolExecutionRecord",
]
