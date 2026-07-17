"""Single continuous Human Review session with bounded revisions and memory."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src.models.memory import CandidateMemory, NormalizedValue
from src.services.memory_store import (
    MemoryPersistenceError,
    apply_review_facts,
    deterministic_review_fact_id,
    save_memory_atomic,
)
from src.services.resume_finalizer import (
    FinalizationError,
    FinalizedResumeResult,
    UnapprovedFinalizationError,
    finalize_approved_resume,
)
from src.tools.resume_tailoring import ResumeTailoringResult


class HumanReviewError(Exception):
    """Base error for the single Human Review workflow."""


class InvalidReviewBatchError(HumanReviewError):
    """Raised when drafts or decisions do not form a valid review batch."""


class DuplicatePauseError(HumanReviewError):
    """Raised when a completed review session would be reopened."""


class RevisionLimitError(HumanReviewError):
    """Raised when a resume would require a third revision."""


class RevisionResultMismatchError(HumanReviewError):
    """Raised when a revision handler returns the wrong or unsafe result."""


class ReviewDecisionType(StrEnum):
    """Allowed Human Review decisions."""

    APPROVE = "approve"
    REJECT = "reject"


class ReviewFactType(StrEnum):
    """Only candidate skills and candidate facts may enter memory."""

    SKILL = "skill"
    CANDIDATE_FACT = "candidate_fact"


class ReviewFactInput(BaseModel):
    """Candidate-stated fact learned during the continuous review session."""

    fact_type: ReviewFactType
    statement: str
    normalized_value: NormalizedValue
    skill_tags: list[str]

    @field_validator("statement")
    @classmethod
    def require_statement(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Review fact statement must be nonempty")
        return value

    @field_validator("skill_tags")
    @classmethod
    def normalize_skill_tags(cls, value: list[str]) -> list[str]:
        tags = sorted({" ".join(tag.split()) for tag in value if tag.strip()}, key=str.casefold)
        return tags

    @model_validator(mode="after")
    def validate_skill_content(self) -> "ReviewFactInput":
        if self.fact_type == ReviewFactType.SKILL:
            normalized_nonempty = (
                isinstance(self.normalized_value, str)
                and bool(self.normalized_value.strip())
            ) or (
                isinstance(self.normalized_value, list)
                and any(item.strip() for item in self.normalized_value)
            ) or isinstance(self.normalized_value, (int, float, bool))
            if not normalized_nonempty and not self.skill_tags:
                raise ValueError(
                    "Skill review fact requires a nonempty normalized_value or skill_tags"
                )
        return self


class ResumeReviewDecision(BaseModel):
    """One candidate decision for one currently pending resume."""

    job_id: str
    decision: ReviewDecisionType
    comments: str = ""
    learned_facts: list[ReviewFactInput] = Field(default_factory=list)

    @field_validator("job_id")
    @classmethod
    def require_job_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Review decision job_id must be nonempty")
        return value

    @model_validator(mode="after")
    def require_rejection_comments(self) -> "ResumeReviewDecision":
        self.comments = self.comments.strip()
        if self.decision == ReviewDecisionType.REJECT and not self.comments:
            raise ValueError("Reject decision requires nonempty comments")
        return self


class ReviewRoundRecord(BaseModel):
    """Audit record for one resume decision and optional revision."""

    review_round: int = Field(ge=0, le=2)
    job_id: str
    decision: ReviewDecisionType
    comments: str
    learned_fact_ids: list[str]
    previous_draft_path: Path
    revised_draft_path: Path | None = None
    actions_taken: list[str]
    approved_after_round: bool


class ResumeReviewState(BaseModel):
    """Current review state for one job-specific resume."""

    job_id: str
    current_draft: ResumeTailoringResult
    status: str
    revision_count: int = Field(ge=0, le=2)
    approved_revision: int | None = None


class HumanReviewSessionState(BaseModel):
    """Snapshot passed to an injected decision provider."""

    session_id: str
    pause_count: int
    session_open: bool
    provider_call_count: int
    pending_job_ids: list[str]
    resume_states: list[ResumeReviewState]


class HumanReviewSessionResult(BaseModel):
    """Complete Human Review, memory, revision, and finalization audit."""

    session_id: str
    pause_count: int
    initial_job_ids: list[str]
    completed: bool
    all_approved: bool
    review_round_records: list[ReviewRoundRecord]
    final_memory: CandidateMemory
    memory_path: Path
    learned_fact_ids: list[str]
    revision_count_by_job: dict[str, int]
    approved_revision_by_job: dict[str, int]
    finalized_resumes: list[FinalizedResumeResult]
    finalization_count: int
    max_revision_rounds: int
    decision_provider_call_count: int
    audit_summary: str

    @model_validator(mode="after")
    def validate_success(self) -> "HumanReviewSessionResult":
        if self.completed:
            if self.pause_count != 1 or not self.all_approved:
                raise ValueError("Completed Human Review must have one pause and all approvals")
            if self.finalization_count != 3 or len(self.finalized_resumes) != 3:
                raise ValueError("Completed Human Review must finalize exactly three resumes")
            if set(self.approved_revision_by_job) != set(self.initial_job_ids):
                raise ValueError("Every initial job must have one approved revision")
            if any(count > 2 for count in self.revision_count_by_job.values()):
                raise ValueError("No resume may have more than two revisions")
        return self


class ReviewDecisionProvider(Protocol):
    """Injectable batch provider used only inside the single continuous pause."""

    def __call__(
        self,
        pending_drafts: Sequence[ResumeTailoringResult],
        session_state: HumanReviewSessionState,
    ) -> Sequence[ResumeReviewDecision]:
        """Return one decision for every pending draft."""


class RevisionHandler(Protocol):
    """Injectable handler that builds a revised plan outside this workflow."""

    def __call__(
        self,
        job_id: str,
        previous: ResumeTailoringResult,
        review_comments: str,
        updated_memory: CandidateMemory,
        next_revision_round: int,
    ) -> ResumeTailoringResult:
        """Return the validated tailored result for the requested revision."""


def _session_id(drafts: Sequence[ResumeTailoringResult]) -> str:
    identity = "|".join(
        f"{draft.job_id}:{draft.deterministic_plan_digest}"
        for draft in drafts
    )
    return f"review-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"


def _validate_initial_drafts(drafts: Sequence[ResumeTailoringResult]) -> None:
    if len(drafts) != 3:
        raise InvalidReviewBatchError(
            f"Human Review requires exactly 3 initial drafts; received {len(drafts)}"
        )
    job_ids = [draft.job_id for draft in drafts]
    if len(set(job_ids)) != 3:
        raise InvalidReviewBatchError("Initial resume drafts must have distinct job IDs")
    for draft in drafts:
        if draft.revision_round != 0:
            raise InvalidReviewBatchError("Every initial resume draft must be revision_round 0")
        if not draft.change_log_path.is_file():
            raise InvalidReviewBatchError(
                f"Initial draft change log is missing: {draft.change_log_path}"
            )


def _validate_decisions(
    raw_decisions: Sequence[ResumeReviewDecision],
    pending_job_ids: list[str],
    all_job_ids: set[str],
    approved_job_ids: set[str],
) -> list[ResumeReviewDecision]:
    try:
        decisions = [ResumeReviewDecision.model_validate(decision) for decision in raw_decisions]
    except (ValidationError, TypeError) as exc:
        raise InvalidReviewBatchError(f"Decision provider returned an invalid decision: {exc}") from exc
    decision_ids = [decision.job_id for decision in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise InvalidReviewBatchError("Decision provider returned duplicate job decisions")
    unknown = set(decision_ids) - all_job_ids
    if unknown:
        raise InvalidReviewBatchError(f"Decision provider returned unknown job IDs: {sorted(unknown)}")
    already_approved = set(decision_ids) & approved_job_ids
    if already_approved:
        raise InvalidReviewBatchError(
            f"Decision provider returned already-approved jobs: {sorted(already_approved)}"
        )
    if set(decision_ids) != set(pending_job_ids):
        missing = set(pending_job_ids) - set(decision_ids)
        extra = set(decision_ids) - set(pending_job_ids)
        raise InvalidReviewBatchError(
            f"Provider must decide every pending resume; missing={sorted(missing)}, "
            f"unexpected={sorted(extra)}"
        )
    by_id = {decision.job_id: decision for decision in decisions}
    return [by_id[job_id] for job_id in pending_job_ids]


def _validate_revision(
    revised: ResumeTailoringResult,
    *,
    job_id: str,
    next_round: int,
    comments: str,
) -> None:
    if revised.job_id != job_id:
        raise RevisionResultMismatchError(
            f"Revision handler returned job {revised.job_id!r}; expected {job_id!r}"
        )
    if revised.revision_round != next_round:
        raise RevisionResultMismatchError(
            f"Revision handler returned round {revised.revision_round}; expected {next_round}"
        )
    if (revised.review_feedback or "").strip() != comments.strip():
        raise RevisionResultMismatchError(
            "Revision result does not contain the supplied review feedback"
        )
    if revised.page_count != 1 or revised.compilation.page_count != 1:
        raise RevisionResultMismatchError("Revision result must remain exactly one page")
    if not revised.protected_regions_unchanged:
        raise RevisionResultMismatchError("Revision result failed protected-region verification")
    if revised.summary_change_count != 1:
        raise RevisionResultMismatchError("Revision result must contain one summary change")
    if revised.experience_bullet_change_count != 2:
        raise RevisionResultMismatchError(
            "Revision result must contain exactly two experience-bullet changes"
        )
    if not revised.change_log_path.is_file():
        raise RevisionResultMismatchError("Revision result change log is missing")


def _safe_job_folder(job_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", job_id)
    return f"job-{safe}"


def run_human_review_session(
    initial_drafts: Sequence[ResumeTailoringResult],
    memory: CandidateMemory,
    memory_path: Path,
    final_output_root: Path,
    decision_provider: ReviewDecisionProvider,
    revision_handler: RevisionHandler,
) -> HumanReviewSessionResult:
    """Run exactly one continuous review pause, then finalize three approvals."""
    drafts = list(initial_drafts)
    _validate_initial_drafts(drafts)
    session_id = _session_id(drafts)
    final_output_root = final_output_root.resolve()
    final_output_root.mkdir(parents=True, exist_ok=True)
    marker_path = final_output_root / f".{session_id}.completed.json"
    if marker_path.exists():
        raise DuplicatePauseError(
            f"Human Review session {session_id} is already completed and cannot be reopened"
        )

    initial_job_ids = [draft.job_id for draft in drafts]
    all_job_ids = set(initial_job_ids)
    states = {
        draft.job_id: ResumeReviewState(
            job_id=draft.job_id,
            current_draft=draft,
            status="pending",
            revision_count=0,
            approved_revision=None,
        )
        for draft in drafts
    }
    current_memory = memory.model_copy(deep=True)
    records: list[ReviewRoundRecord] = []
    learned_fact_ids: list[str] = []
    pause_count = 1
    provider_call_count = 0

    while True:
        pending_job_ids = [
            job_id
            for job_id in initial_job_ids
            if states[job_id].status == "pending"
        ]
        if not pending_job_ids:
            break
        provider_call_count += 1
        snapshot = HumanReviewSessionState(
            session_id=session_id,
            pause_count=pause_count,
            session_open=True,
            provider_call_count=provider_call_count,
            pending_job_ids=pending_job_ids,
            resume_states=[
                states[job_id].model_copy(deep=True)
                for job_id in initial_job_ids
            ],
        )
        pending_drafts = [states[job_id].current_draft for job_id in pending_job_ids]
        raw_decisions = decision_provider(tuple(pending_drafts), snapshot)
        decisions = _validate_decisions(
            raw_decisions,
            pending_job_ids,
            all_job_ids,
            {
                job_id
                for job_id, state in states.items()
                if state.status == "approved"
            },
        )

        fact_ids_by_job: dict[str, list[str]] = {}
        updated_memory = current_memory
        for decision in decisions:
            fact_ids = [
                deterministic_review_fact_id(fact)
                for fact in decision.learned_facts
            ]
            fact_ids_by_job[decision.job_id] = fact_ids
            for fact_id in fact_ids:
                if fact_id not in learned_fact_ids:
                    learned_fact_ids.append(fact_id)
            if decision.learned_facts:
                draft_round = states[decision.job_id].current_draft.revision_round
                provenance_round = 1 if draft_round == 0 else 2
                updated_memory = apply_review_facts(
                    updated_memory,
                    decision.learned_facts,
                    provenance_round,
                    [decision.job_id],
                )

        # Mandatory barrier: the complete round's facts are persisted before revisions.
        save_memory_atomic(updated_memory, memory_path)
        current_memory = updated_memory

        for decision in decisions:
            state = states[decision.job_id]
            previous = state.current_draft
            actions = ["review_decision_recorded", "memory_round_persisted"]
            revised_path: Path | None = None
            approved_after = decision.decision == ReviewDecisionType.APPROVE
            if approved_after:
                state.status = "approved"
                state.approved_revision = previous.revision_round
                actions.append(f"approved_r{previous.revision_round}")
            else:
                if previous.revision_round >= 2:
                    raise RevisionLimitError(
                        f"Job {decision.job_id} was rejected at r2; a third revision is forbidden"
                    )
                next_round = previous.revision_round + 1
                revised = revision_handler(
                    decision.job_id,
                    previous,
                    decision.comments,
                    current_memory.model_copy(deep=True),
                    next_round,
                )
                _validate_revision(
                    revised,
                    job_id=decision.job_id,
                    next_round=next_round,
                    comments=decision.comments,
                )
                state.current_draft = revised
                state.revision_count = next_round
                state.status = "pending"
                revised_path = revised.draft_pdf_path
                actions.extend(["rejected", f"revision_r{next_round}_created"])

            records.append(
                ReviewRoundRecord(
                    review_round=previous.revision_round,
                    job_id=decision.job_id,
                    decision=decision.decision,
                    comments=decision.comments,
                    learned_fact_ids=fact_ids_by_job[decision.job_id],
                    previous_draft_path=previous.draft_pdf_path,
                    revised_draft_path=revised_path,
                    actions_taken=actions,
                    approved_after_round=approved_after,
                )
            )

    if any(state.status != "approved" for state in states.values()):
        raise HumanReviewError("Human Review ended without approval for every resume")

    finalized: list[FinalizedResumeResult] = []
    try:
        for job_id in initial_job_ids:
            finalized.append(
                finalize_approved_resume(
                    states[job_id].current_draft,
                    final_output_root / _safe_job_folder(job_id),
                    approved=True,
                )
            )
    except (FinalizationError, OSError) as exc:
        raise FinalizationError(
            f"Human Review approvals were collected but finalization failed: {exc}"
        ) from exc

    revision_counts = {
        job_id: states[job_id].revision_count
        for job_id in initial_job_ids
    }
    approved_revisions = {
        job_id: states[job_id].approved_revision
        for job_id in initial_job_ids
        if states[job_id].approved_revision is not None
    }
    audit_summary = (
        f"One continuous Human Review pause completed for 3 resumes in "
        f"{provider_call_count} decision round(s); all resumes were approved, "
        f"{sum(revision_counts.values())} revision(s) were produced, "
        f"{len(learned_fact_ids)} unique fact(s) were learned, and 3 resumes were finalized."
    )
    result = HumanReviewSessionResult(
        session_id=session_id,
        pause_count=pause_count,
        initial_job_ids=initial_job_ids,
        completed=True,
        all_approved=True,
        review_round_records=records,
        final_memory=current_memory,
        memory_path=memory_path.resolve(),
        learned_fact_ids=learned_fact_ids,
        revision_count_by_job=revision_counts,
        approved_revision_by_job=approved_revisions,
        finalized_resumes=finalized,
        finalization_count=len(finalized),
        max_revision_rounds=2,
        decision_provider_call_count=provider_call_count,
        audit_summary=audit_summary,
    )
    marker_path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "completed": True,
                "pause_count": 1,
                "approved_job_ids": initial_job_ids,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


__all__ = [
    "DuplicatePauseError",
    "FinalizationError",
    "HumanReviewError",
    "HumanReviewSessionResult",
    "HumanReviewSessionState",
    "InvalidReviewBatchError",
    "MemoryPersistenceError",
    "ResumeReviewDecision",
    "ResumeReviewState",
    "ReviewDecisionProvider",
    "ReviewDecisionType",
    "ReviewFactInput",
    "ReviewFactType",
    "ReviewRoundRecord",
    "RevisionHandler",
    "RevisionLimitError",
    "RevisionResultMismatchError",
    "UnapprovedFinalizationError",
    "run_human_review_session",
]
