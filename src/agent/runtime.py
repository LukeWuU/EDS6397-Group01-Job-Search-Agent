"""The one production LLM runtime and its continuous tool-calling loop."""

from __future__ import annotations

import copy
import json
import re
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, field_validator

from src.agent.client import (
    ChatModelClient,
    NormalizedAssistantMessage,
    NormalizedToolCall,
    OllamaChatModelClient,
)
from src.agent.prompts import (
    COVER_LETTER_NORMAL_CONSTRAINTS,
    COVER_LETTER_PLAN_LIMITS,
    SYSTEM_PROMPT,
    TAILOR_RESUME_ARGUMENT_TEMPLATE,
    TAILOR_RESUME_NORMAL_CONSTRAINTS,
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
from src.tools.cover_letter import (
    CoverLetterCitation,
    CoverLetterParagraph,
    CoverLetterPlan,
    CoverLetterSkillItem,
    _normalize_phrase,
    _skill_is_job_relevant,
)
from src.tools.filtering import normalize_title
from src.tools.fit_analysis import FitAnalysisResult
from src.tools.resume_tailoring import (
    _candidate_supported_skills,
    _contains_canonical,
)
from src.tools.scoring import normalize_skill
from src.models.bundle import CandidateBundle
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
)
_TAILOR_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")


def _word_count(value: str) -> int:
    return len(value.split())


class _TailorSemanticTextEdit(BaseModel):
    """Model-authored resume text for one summary or bullet edit."""

    model_config = ConfigDict(extra="forbid")

    new_text: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=200)

    @field_validator("new_text", "reason")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("text fields must be nonempty")
        return value


class _TailorResumeTextDraftBase(BaseModel):
    """Shared compact model-facing adapter fields for assignment tool tailor_resume."""

    model_config = ConfigDict(extra="forbid")

    decision_summary: str = Field(min_length=1, max_length=500)
    job_id: str = Field(min_length=1)
    professional_summary: _TailorSemanticTextEdit
    bullet_1: _TailorSemanticTextEdit
    bullet_2: _TailorSemanticTextEdit
    plan_rationale: str = Field(min_length=1, max_length=200)

    @field_validator("decision_summary", "plan_rationale")
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("required string fields must be nonempty")
        return value

    @field_validator("professional_summary")
    @classmethod
    def validate_summary_word_limit(
        cls,
        value: _TailorSemanticTextEdit,
    ) -> _TailorSemanticTextEdit:
        if _word_count(value.new_text) > 55:
            raise ValueError("professional_summary.new_text must be at most 55 words")
        if _word_count(value.reason) > 18:
            raise ValueError("professional_summary.reason must be at most 18 words")
        return value

    @field_validator("bullet_1", "bullet_2")
    @classmethod
    def validate_named_bullet_word_limits(
        cls,
        value: _TailorSemanticTextEdit,
        info,
    ) -> _TailorSemanticTextEdit:
        field_name = info.field_name or "bullet"
        if _word_count(value.new_text) > 32:
            raise ValueError(f"{field_name}.new_text must be at most 32 words")
        if _word_count(value.reason) > 18:
            raise ValueError(f"{field_name}.reason must be at most 18 words")
        return value

    @field_validator("plan_rationale")
    @classmethod
    def validate_plan_rationale_words(cls, value: str) -> str:
        if _word_count(value) > 25:
            raise ValueError("plan_rationale must be at most 25 words")
        return value


class _TailorResumeTextDraftNoSwap(_TailorResumeTextDraftBase):
    project_swap_reason: Literal[None] = None


class _TailorResumeTextDraftWithSwap(_TailorResumeTextDraftBase):
    project_swap_reason: str = Field(min_length=1, max_length=200)


# Backward-compatible alias for tests importing the draft adapter model.
_TailorResumeTextDraftArguments = _TailorResumeTextDraftNoSwap

_COVER_PATCHABLE_FIELDS = (
    "company_hook_phrase",
    "body_paragraph_1",
    "body_paragraph_2",
    "skills",
    "closing_sentence",
    "plan_rationale",
)

_COVER_MEANINGFUL_HOOK_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

_COVER_NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:\.\d+)?%?(?!\w)")
_COVER_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")

_COVER_CITATION_ERROR_MARKERS = (
    "unknown job citation field",
    "unknown candidate profile source id",
    "unknown experience bullet id",
    "unknown evidence id",
    "unknown evidence-registry id",
    "not authorized for cover letters",
    "does not identify supplied job",
    "company details citation must",
)

_COVER_PREFERRED_SKILLS: dict[str, list[str]] = {
    "Chickasaw Nation Industries": ["Python", "REST APIs", "RAG", "Docker"],
    "Camden Property Trust": ["Python", "REST APIs", "RAG", "MLOps"],
    "Flash AI": ["Python", "RAG", "Embeddings", "NLP"],
}


class _CoverLetterBodyParagraph(BaseModel):
    """Model-authored cover letter paragraph text."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("text", "reason")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("text fields must be nonempty")
        return value


class _CoverLetterTransportDraft(BaseModel):
    """Compact transport layer for model-facing cover letter drafts."""

    model_config = ConfigDict(extra="forbid")

    decision_summary: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    company_hook_phrase: str = Field(min_length=1)
    body_paragraph_1: _CoverLetterBodyParagraph
    body_paragraph_2: _CoverLetterBodyParagraph | None = None
    skills: list[str] = Field(min_length=1)
    closing_sentence: str = Field(min_length=1)
    plan_rationale: str = Field(min_length=1)

    @field_validator(
        "decision_summary",
        "company_hook_phrase",
        "closing_sentence",
        "plan_rationale",
    )
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("required string fields must be nonempty")
        return value

    @field_validator("skills")
    @classmethod
    def validate_skill_strings(cls, value: list[str]) -> list[str]:
        cleaned = [" ".join(skill.split()) for skill in value if skill.strip()]
        if len(cleaned) != len(value):
            raise ValueError("skills must be nonempty strings")
        return cleaned


class _CoverLetterSemanticDraft(BaseModel):
    """Validated semantic cover letter draft after transport and audit."""

    model_config = ConfigDict(extra="forbid")

    decision_summary: str
    job_id: str
    company_hook_phrase: str
    body_paragraph_1: _CoverLetterBodyParagraph
    body_paragraph_2: _CoverLetterBodyParagraph | None = None
    skills: list[str] = Field(min_length=3, max_length=8)
    closing_sentence: str
    plan_rationale: str

    @field_validator("skills")
    @classmethod
    def validate_unique_skills(cls, value: list[str]) -> list[str]:
        canonical = [
            normalize_skill(skill, has_vector_search=True) or skill.casefold()
            for skill in value
        ]
        if len(set(canonical)) != len(value):
            raise ValueError("skills must be unique by canonical form")
        return value


class _CoverLetterTextDraft(_CoverLetterTransportDraft):
    """Backward-compatible alias for tests referencing the compact draft model name."""


class _CoverLetterAuditIssue(BaseModel):
    field: str
    category: str
    message: str


class _CoverLetterAuditResult(BaseModel):
    issues: list[_CoverLetterAuditIssue]

    @property
    def fields(self) -> list[str]:
        return sorted({issue.field for issue in self.issues})

    def raise_if_issues(self) -> None:
        if not self.issues:
            return
        parts = [f"{issue.field}: {issue.message}" for issue in self.issues]
        raise ToolArgumentsError(
            "Cover letter draft audit rejection: " + "; ".join(parts)
        )


class _AllowedCoverSkill(BaseModel):
    display_name: str
    canonical: str
    citation: dict[str, Any]


_TAILOR_PATCHABLE_FIELDS = (
    "professional_summary",
    "bullet_1",
    "bullet_2",
    "project_swap_reason",
)


class _ReconciledSkillBinding(BaseModel):
    skill: str
    source_hint: str


class _ReconciledTailoringEvidence(BaseModel):
    """Concise reconciled skill view for tailoring; raw Fit Analysis is unchanged."""

    aligned_skills: list[str]
    evidenced_elsewhere_skills: list[str]
    genuine_gaps: list[str]
    supported_skill_bindings: list[_ReconciledSkillBinding]
    job_requirements_unsupported: list[str]
    skills_moved_from_gap_to_supported: list[str]
    skills_moved_from_aligned_to_gap: list[str]
    category_conflicts_resolved: int
    reconciliation_applied: bool = True


class _DraftAuditIssue(BaseModel):
    field: str
    category: str
    message: str
    bullet_slot: str | None = None


class _DraftAuditResult(BaseModel):
    issues: list[_DraftAuditIssue]

    @property
    def fields(self) -> list[str]:
        return sorted({issue.field for issue in self.issues})

    def raise_if_issues(self) -> None:
        if not self.issues:
            return
        parts = [f"{issue.field}: {issue.message}" for issue in self.issues]
        raise ToolArgumentsError(
            "Tailor resume draft audit rejection: " + "; ".join(parts)
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
        cover_letter_date: date | None = None,
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
        self._last_generation_span: SpanContext | None = None
        self._tailor_patch_recovery: dict[str, Any] | None = None
        self._cover_letter_patch_recovery: dict[str, Any] | None = None
        self._reconciled_evidence_cache: dict[str, _ReconciledTailoringEvidence] = {}
        self._cover_letter_skill_registry_cache: dict[
            str, dict[str, _AllowedCoverSkill]
        ] = {}
        self._cover_letter_date = cover_letter_date or date.today()

    @staticmethod
    def _phase_needs_compaction(phase: str) -> bool:
        return phase in {"resume_tailoring", "resume_revision", "cover_letter"}

    def _contract_for_model(self, contract: dict[str, Any]) -> dict[str, Any]:
        """Return a model-facing contract without recovery-only fields."""
        return {
            key: value
            for key, value in contract.items()
            if key
            not in {
                "exact_tailor_resume_structural_template",
                "target_context",
                "tailor_patch_fields",
                "tailor_recovery_mode",
                "cover_patch_fields",
                "cover_recovery_mode",
                "reconciliation_metadata",
            }
        }

    def _serialized_conversation_char_count(self) -> int:
        return len(
            json.dumps(
                self.conversation,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    def _model_call_diagnostics(
        self,
        contract: dict[str, Any],
        schemas: list[dict[str, Any]],
    ) -> dict[str, Any]:
        diagnostics = {
            "model_message_count": len(self.conversation),
            "serialized_message_char_count": self._serialized_conversation_char_count(),
            "serialized_tool_schema_char_count": len(
                json.dumps(schemas, ensure_ascii=False, separators=(",", ":"))
            ),
            "phase": contract["phase"],
            "target_rank": contract.get("target_rank"),
        }
        if contract["allowed_tool"] == "tailor_resume":
            diagnostics["model_argument_mode"] = "tailor_resume_text_draft"
            diagnostics["hydration_applied"] = False
            diagnostics["semantic_bullet_slot_mode"] = "named"
            diagnostics["required_role_phrase"] = contract.get("required_role_phrase")
            diagnostics["project_swap_required"] = contract.get("project_swap_required")
            target_job_id = contract.get("target_job_id")
            if target_job_id:
                reconciled = self._reconcile_tailoring_evidence(target_job_id)
                diagnostics["evidence_reconciliation_applied"] = True
                diagnostics["reconciled_aligned_skill_count"] = len(
                    reconciled.aligned_skills
                )
                diagnostics["reconciled_gap_count"] = len(reconciled.genuine_gaps)
                diagnostics["reconciled_conflict_count"] = (
                    reconciled.category_conflicts_resolved
                )
            if contract.get("tailor_recovery_mode") == "patch":
                diagnostics["patch_recovery_applied"] = True
                diagnostics["patch_fields"] = contract.get("tailor_patch_fields", [])
        if contract["allowed_tool"] == "generate_cover_letter":
            diagnostics["model_argument_mode"] = (
                "cover_letter_patch_draft"
                if contract.get("cover_recovery_mode") == "patch"
                else "cover_letter_text_draft"
            )
            diagnostics["cover_letter_compact_draft"] = True
            diagnostics["cover_letter_hydration_applied"] = False
            target_job_id = contract.get("target_job_id")
            if target_job_id:
                diagnostics["cover_letter_allowed_skill_count"] = len(
                    self._select_model_visible_skills(target_job_id)
                )
                diagnostics["cover_letter_visible_skill_count"] = diagnostics[
                    "cover_letter_allowed_skill_count"
                ]
            if contract.get("cover_recovery_mode") == "patch":
                diagnostics["cover_letter_patch_recovery_applied"] = True
                diagnostics["cover_letter_patch_fields"] = contract.get(
                    "cover_patch_fields",
                    [],
                )
                if self._cover_letter_patch_recovery:
                    diagnostics["cover_letter_preserved_field_count"] = len(
                        self._cover_letter_patch_recovery.get("preserved_fields", [])
                    )
                    diagnostics["cover_letter_rejected_category"] = (
                        self._cover_letter_patch_recovery.get("rejected_category")
                    )
            else:
                diagnostics["cover_letter_patch_recovery_applied"] = False
        return diagnostics

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

    def _report_validation_rejection_progress(
        self,
        error: str,
        contract: dict[str, Any],
        *,
        audit: _DraftAuditResult | None = None,
    ) -> None:
        assert self.state is not None
        if contract.get("allowed_tool") == "tailor_resume":
            if audit and audit.issues:
                categories = {issue.category for issue in audit.issues}
                if "semantic_text" in categories:
                    category = "semantic_text"
                elif "evidence" in categories:
                    category = "evidence"
                elif "hydration" in categories:
                    category = "hydration"
                else:
                    category = self._classify_tailor_rejection(error)
            else:
                category = self._classify_tailor_rejection(error)
        elif contract.get("allowed_tool") == "generate_cover_letter":
            if audit and getattr(audit, "issues", None):
                categories = {issue.category for issue in audit.issues}
                if "semantic_text" in categories:
                    category = "semantic_text"
                elif "evidence" in categories:
                    category = "evidence"
                elif "hydration" in categories:
                    category = "hydration"
                elif "target_job" in categories:
                    category = "target_job"
                elif "draft_schema" in categories:
                    category = "draft_schema"
                elif "citation" in categories:
                    category = "citation"
                else:
                    category = self._classify_cover_letter_rejection(error)
            else:
                category = self._classify_cover_letter_rejection(error)
        else:
            category = "validation"
        self._report_progress(
            f"Model call {self.state.model_call_count}: "
            "tool call rejected by validation"
        )
        self._report_progress(f"Validation category: {category}")
        if self._last_generation_span is not None:
            existing = self._last_generation_span.record.metadata or {}
            metadata = {**existing, "rejected_category": category}
            slot = None
            if audit and audit.issues:
                for issue in audit.issues:
                    slot = getattr(issue, "bullet_slot", None)
                    if slot:
                        break
            if slot is None:
                slot = self._infer_rejected_bullet_slot(error)
            if slot is not None:
                metadata["rejected_bullet_slot"] = slot
            if audit and audit.issues:
                metadata["draft_audit_issue_count"] = len(audit.issues)
                metadata["draft_audit_fields"] = audit.fields
            if self._tailor_patch_recovery:
                metadata["patch_recovery_applied"] = True
                metadata["patch_fields"] = self._tailor_patch_recovery["patch_fields"]
                metadata["preserved_field_count"] = len(
                    self._tailor_patch_recovery.get("preserved_fields", [])
                )
            if self._cover_letter_patch_recovery:
                metadata["cover_letter_patch_recovery_applied"] = True
                metadata["cover_letter_patch_fields"] = (
                    self._cover_letter_patch_recovery["patch_fields"]
                )
                metadata["cover_letter_preserved_field_count"] = len(
                    self._cover_letter_patch_recovery.get("preserved_fields", [])
                )
            if audit and getattr(audit, "issues", None):
                metadata["draft_audit_issue_count"] = len(audit.issues)
                metadata["draft_audit_fields"] = audit.fields
                if contract.get("allowed_tool") == "generate_cover_letter":
                    metadata["cover_letter_audit_issue_count"] = len(audit.issues)
                    metadata["cover_letter_audit_fields"] = audit.fields
            metadata["cover_letter_rejected_category"] = (
                category if contract.get("allowed_tool") == "generate_cover_letter" else None
            )
            if metadata["cover_letter_rejected_category"] is None:
                metadata.pop("cover_letter_rejected_category", None)
            self._last_generation_span.record.metadata = metadata

    @staticmethod
    def _derive_required_role_phrase(job_title: str) -> str:
        primary = " ".join(job_title.split("|", 1)[0].split())
        parts = primary.split()
        if len(parts) >= 2:
            return " ".join(parts[:-1] + [parts[-1].lower()])
        return primary.lower()

    @staticmethod
    def _summary_includes_role_phrase(summary_text: str, required_phrase: str) -> bool:
        return required_phrase.casefold() in summary_text.casefold()

    def _candidate_bullet_corpus(self) -> str:
        assert self.registry is not None
        parts: list[str] = []
        for experience in self.registry.bundle.profile.experience:
            for bullet in experience.bullets:
                parts.append(bullet.text)
        return " ".join(parts)

    def _skill_supported_by_candidate(self, skill: str, *, bullet_corpus: str) -> bool:
        assert self.registry is not None
        canonical = normalize_skill(skill, has_vector_search=True)
        if not canonical:
            return False
        supported = _candidate_supported_skills(
            self.registry.bundle,
            self.registry.memory,
        )
        if canonical in supported:
            return True
        if _contains_canonical(bullet_corpus, canonical):
            return True
        for fact in self.registry.memory.facts:
            fact_text = " ".join(
                str(value)
                for value in (
                    fact.statement,
                    fact.normalized_value,
                    " ".join(fact.skill_tags),
                )
                if value
            )
            if _contains_canonical(fact_text, canonical):
                return True
        return False

    def _reconcile_tailoring_evidence(
        self,
        job_id: str,
    ) -> _ReconciledTailoringEvidence:
        if job_id in self._reconciled_evidence_cache:
            return self._reconciled_evidence_cache[job_id]
        assert self.state is not None
        assert self.registry is not None
        analysis = self.state.fit_analyses[job_id]
        job = self.registry._job(job_id)
        bullet_corpus = self._candidate_bullet_corpus()
        moved_gap_to_supported: list[str] = []
        moved_aligned_to_gap: list[str] = []
        conflicts = 0
        seen: set[str] = set()
        aligned: list[str] = []
        evidenced: list[str] = []
        gaps: list[str] = []

        def canonical_key(skill: str) -> str:
            return normalize_skill(skill, has_vector_search=True) or skill.casefold()

        def add_unique(bucket: list[str], skill: str) -> None:
            nonlocal conflicts
            key = canonical_key(skill)
            if key in seen:
                conflicts += 1
                return
            seen.add(key)
            bucket.append(skill)

        def has_support(skill: str) -> bool:
            return self._skill_supported_by_candidate(
                skill,
                bullet_corpus=bullet_corpus,
            )

        for skill in analysis.core_skills.aligned_skills:
            if has_support(skill):
                add_unique(aligned, skill)
            else:
                moved_aligned_to_gap.append(skill)
                add_unique(gaps, skill)

        for skill in analysis.core_skills.evidenced_elsewhere_skills:
            if has_support(skill):
                add_unique(evidenced, skill)
            elif canonical_key(skill) not in seen:
                add_unique(gaps, skill)

        for skill in analysis.core_skills.genuine_gaps:
            if has_support(skill):
                moved_gap_to_supported.append(skill)
                key = canonical_key(skill)
                if key not in seen:
                    add_unique(aligned, skill)
            elif canonical_key(skill) not in seen:
                add_unique(gaps, skill)

        job_requirements_unsupported: list[str] = []
        for skill in job.required_skills:
            if not has_support(skill) and canonical_key(skill) not in seen:
                add_unique(gaps, skill)
                job_requirements_unsupported.append(skill)

        bindings: list[_ReconciledSkillBinding] = []
        for skill in [*aligned, *evidenced]:
            canonical = normalize_skill(skill, has_vector_search=True)
            hint = "candidate evidence"
            if canonical and _contains_canonical(bullet_corpus, canonical):
                hint = "primary experience bullets"
            bindings.append(_ReconciledSkillBinding(skill=skill, source_hint=hint))

        reconciled = _ReconciledTailoringEvidence(
            aligned_skills=aligned,
            evidenced_elsewhere_skills=evidenced,
            genuine_gaps=gaps,
            supported_skill_bindings=bindings,
            job_requirements_unsupported=job_requirements_unsupported,
            skills_moved_from_gap_to_supported=moved_gap_to_supported,
            skills_moved_from_aligned_to_gap=moved_aligned_to_gap,
            category_conflicts_resolved=conflicts,
        )
        self._reconciled_evidence_cache[job_id] = reconciled
        return reconciled

    def _build_resume_execution_fit_analysis(self, job_id: str) -> FitAnalysisResult:
        """Build a private reconciled Fit Analysis copy for resume-tool execution."""
        assert self.state is not None
        raw_analysis = self.state.fit_analyses[job_id]
        reconciled = self._reconcile_tailoring_evidence(job_id)
        execution_copy = raw_analysis.model_copy(deep=True)
        execution_copy.core_skills = execution_copy.core_skills.model_copy(
            update={
                "aligned_skills": list(reconciled.aligned_skills),
                "evidenced_elsewhere_skills": list(
                    reconciled.evidenced_elsewhere_skills
                ),
                "genuine_gaps": list(reconciled.genuine_gaps),
            }
        )
        return execution_copy

    def _is_target_job_skill(self, skill: str, job_id: str) -> bool:
        assert self.registry is not None
        canonical = normalize_skill(skill, has_vector_search=True)
        if not canonical:
            return False
        job = self.registry._job(job_id)
        return any(
            normalize_skill(required, has_vector_search=True) == canonical
            for required in job.required_skills
        )

    def _build_resume_execution_bundle(
        self,
        job_id: str,
    ) -> tuple[CandidateBundle, dict[str, Any]]:
        """Build a private bundle copy with narrowly promoted bullet evidence skills."""
        assert self.registry is not None
        raw_bundle = self.registry.bundle
        reconciled = self._reconcile_tailoring_evidence(job_id)
        execution_copy = raw_bundle.model_copy(deep=True)
        promotions: dict[str, set[str]] = {}

        for skill in reconciled.skills_moved_from_gap_to_supported:
            if not self._is_target_job_skill(skill, job_id):
                continue
            canonical = normalize_skill(skill, has_vector_search=True)
            if not canonical:
                continue
            for experience in raw_bundle.profile.experience:
                for bullet in experience.bullets:
                    if not _contains_canonical(bullet.text, canonical):
                        continue
                    for evidence_id in bullet.evidence_ids:
                        if raw_bundle.get_evidence(evidence_id) is None:
                            continue
                        promotions.setdefault(evidence_id, set()).add(skill)

        if not promotions:
            return execution_copy, {
                "promoted_execution_skill_count": 0,
                "promoted_execution_skills": [],
                "promoted_evidence_record_count": 0,
            }

        updated_records = []
        promoted_skills: set[str] = set()
        promoted_evidence_ids: set[str] = set()
        for record in execution_copy.evidence.evidence_records:
            skills_to_add = promotions.get(record.evidence_id)
            if not skills_to_add:
                updated_records.append(record)
                continue
            existing = {
                normalize_skill(item, has_vector_search=True) or item.casefold()
                for item in record.supported_skills
            }
            new_skills = list(record.supported_skills)
            for skill in sorted(skills_to_add):
                canonical = normalize_skill(skill, has_vector_search=True) or skill.casefold()
                if canonical in existing:
                    continue
                new_skills.append(skill)
                existing.add(canonical)
                promoted_skills.add(skill)
            updated_records.append(
                record.model_copy(update={"supported_skills": new_skills})
            )
            promoted_evidence_ids.add(record.evidence_id)

        execution_copy.evidence = execution_copy.evidence.model_copy(
            update={"evidence_records": updated_records}
        )
        return execution_copy, {
            "promoted_execution_skill_count": len(promoted_skills),
            "promoted_execution_skills": sorted(promoted_skills),
            "promoted_evidence_record_count": len(promoted_evidence_ids),
        }

    @contextmanager
    def _resume_execution_context(
        self,
        job_id: str,
    ) -> Iterator[tuple[FitAnalysisResult, CandidateBundle, dict[str, Any]]]:
        """Temporarily substitute reconciled Fit Analysis and bundle for resume execution."""
        assert self.state is not None
        assert self.registry is not None
        raw_analysis = self.state.fit_analyses[job_id]
        raw_bundle = self.registry.bundle
        execution_analysis = self._build_resume_execution_fit_analysis(job_id)
        execution_bundle, bundle_metadata = self._build_resume_execution_bundle(
            job_id
        )
        self.state.fit_analyses[job_id] = execution_analysis
        self.registry.bundle = execution_bundle
        try:
            yield execution_analysis, execution_bundle, bundle_metadata
        finally:
            self.state.fit_analyses[job_id] = raw_analysis
            self.registry.bundle = raw_bundle

    def _supported_skills_map(self) -> dict[str, set[str]]:
        assert self.registry is not None
        return _candidate_supported_skills(
            self.registry.bundle,
            self.registry.memory,
        )

    def _text_claims_unsupported_required_skill(
        self,
        text: str,
        job_id: str,
        *,
        bullet_corpus: str,
    ) -> str | None:
        assert self.registry is not None
        job = self.registry._job(job_id)
        for skill in job.required_skills:
            canonical = normalize_skill(skill, has_vector_search=True)
            if not canonical or not _contains_canonical(text, canonical):
                continue
            if not self._skill_supported_by_candidate(
                skill,
                bullet_corpus=bullet_corpus,
            ):
                return skill
        return None

    def _text_claims_genuine_gap(
        self,
        text: str,
        reconciled: _ReconciledTailoringEvidence,
    ) -> str | None:
        for gap in reconciled.genuine_gaps:
            canonical = normalize_skill(gap, has_vector_search=True)
            if canonical and _contains_canonical(text, canonical):
                return gap
        return None

    def _unsupported_numbers(self, text: str, allowed_text: str) -> list[str]:
        allowed = set(_TAILOR_NUMBER_PATTERN.findall(allowed_text))
        return sorted(
            set(_TAILOR_NUMBER_PATTERN.findall(text)) - allowed
        )

    def _bullet_slot_transfer_issues(
        self,
        slot: str,
        text: str,
        own_source: dict[str, Any],
        other_source: dict[str, Any],
    ) -> list[_DraftAuditIssue]:
        issues: list[_DraftAuditIssue] = []
        own_allowed = (
            own_source.get("current_text", "")
            + " "
            + " ".join(own_source.get("allowed_numeric_claims", []))
        )
        other_allowed = (
            other_source.get("current_text", "")
            + " "
            + " ".join(other_source.get("allowed_numeric_claims", []))
        )
        for number in self._unsupported_numbers(text, own_allowed):
            if number in _TAILOR_NUMBER_PATTERN.findall(other_allowed):
                issues.append(
                    _DraftAuditIssue(
                        field=slot,
                        category="evidence",
                        message=(
                            f"{slot} uses numeric claim {number!r} allowed only "
                            f"for the other bullet slot"
                        ),
                        bullet_slot=slot,
                    )
                )
            else:
                issues.append(
                    _DraftAuditIssue(
                        field=slot,
                        category="evidence",
                        message=f"{slot} contains unsupported numeric claim {number!r}",
                        bullet_slot=slot,
                    )
                )
        return issues

    def _audit_hydrated_tailor_draft(
        self,
        hydrated_call: NormalizedToolCall,
        contract: dict[str, Any],
    ) -> _DraftAuditResult:
        assert self.registry is not None
        target_job_id = contract.get("target_job_id")
        if target_job_id is None:
            return _DraftAuditResult(issues=[])
        reconciled = self._reconcile_tailoring_evidence(target_job_id)
        edit_plan = hydrated_call.arguments.get("edit_plan", {})
        if not isinstance(edit_plan, dict):
            return _DraftAuditResult(issues=[])
        issues: list[_DraftAuditIssue] = []
        bullet_corpus = self._candidate_bullet_corpus()
        target_context = contract.get("target_context") or {}
        slot_sources = {
            "bullet_1": target_context.get("bullet_1_source") or {},
            "bullet_2": target_context.get("bullet_2_source") or {},
        }

        summary = edit_plan.get("professional_summary")
        if isinstance(summary, dict):
            summary_text = str(summary.get("new_text", ""))
            required_phrase = contract.get("required_role_phrase") or (
                self._derive_required_role_phrase(
                    self.registry._job(target_job_id).title
                )
            )
            if not self._summary_includes_role_phrase(summary_text, required_phrase):
                issues.append(
                    _DraftAuditIssue(
                        field="professional_summary",
                        category="semantic_text",
                        message=(
                            "professional summary must explicitly include role "
                            f'phrase "{required_phrase}"'
                        ),
                    )
                )
            unsupported = self._text_claims_unsupported_required_skill(
                summary_text,
                target_job_id,
                bullet_corpus=bullet_corpus,
            )
            if unsupported is not None:
                issues.append(
                    _DraftAuditIssue(
                        field="professional_summary",
                        category="evidence",
                        message=(
                            f"professional summary claims unsupported required "
                            f"skill {unsupported!r}"
                        ),
                    )
                )
            gap = self._text_claims_genuine_gap(summary_text, reconciled)
            if gap is not None:
                issues.append(
                    _DraftAuditIssue(
                        field="professional_summary",
                        category="evidence",
                        message=(
                            f"professional summary claims genuine-gap skill {gap!r}"
                        ),
                    )
                )

        bullet_edits = edit_plan.get("experience_bullet_edits")
        if isinstance(bullet_edits, list):
            for index, edit in enumerate(bullet_edits[:2], start=1):
                if not isinstance(edit, dict):
                    continue
                slot = f"bullet_{index}"
                text = str(edit.get("new_text", ""))
                other_slot = "bullet_2" if slot == "bullet_1" else "bullet_1"
                issues.extend(
                    self._bullet_slot_transfer_issues(
                        slot,
                        text,
                        slot_sources.get(slot, {}),
                        slot_sources.get(other_slot, {}),
                    )
                )
                gap = self._text_claims_genuine_gap(text, reconciled)
                if gap is not None:
                    issues.append(
                        _DraftAuditIssue(
                            field=slot,
                            category="evidence",
                            message=f"{slot} claims genuine-gap skill {gap!r}",
                            bullet_slot=slot,
                        )
                    )
                unsupported = self._text_claims_unsupported_required_skill(
                    text,
                    target_job_id,
                    bullet_corpus=bullet_corpus,
                )
                if unsupported is not None:
                    issues.append(
                        _DraftAuditIssue(
                            field=slot,
                            category="evidence",
                            message=(
                                f"{slot} claims unsupported required skill "
                                f"{unsupported!r}"
                            ),
                            bullet_slot=slot,
                        )
                    )

        if contract.get("project_swap_required"):
            swap_reason = hydrated_call.arguments.get("project_swap_reason")
            if swap_reason is None:
                plan_swap = edit_plan.get("project_swap")
                if isinstance(plan_swap, dict):
                    swap_reason = plan_swap.get("reason")
            if not isinstance(swap_reason, str) or not swap_reason.strip():
                issues.append(
                    _DraftAuditIssue(
                        field="project_swap_reason",
                        category="hydration",
                        message=(
                            "project_swap_reason must be a nonempty string when "
                            "a project swap is required"
                        ),
                    )
                )
        return _DraftAuditResult(issues=issues)

    def _is_complete_tailor_draft(self, arguments: dict[str, Any]) -> bool:
        required = {
            "decision_summary",
            "job_id",
            "professional_summary",
            "bullet_1",
            "bullet_2",
            "plan_rationale",
            "project_swap_reason",
        }
        return required.issubset(arguments)

    def _merge_tailor_patch(
        self,
        base_draft: dict[str, Any],
        patch: dict[str, Any],
        patch_fields: list[str],
    ) -> dict[str, Any]:
        merged = copy.deepcopy(base_draft)
        for field in patch_fields:
            if field in patch:
                merged[field] = copy.deepcopy(patch[field])
        return merged

    def _patch_model_for_contract(
        self,
        contract: dict[str, Any],
    ) -> type[BaseModel]:
        patch_fields = list(contract.get("tailor_patch_fields") or [])
        field_defs: dict[str, Any] = {
            "job_id": (str, Field(min_length=1)),
        }
        if "professional_summary" in patch_fields:
            field_defs["professional_summary"] = (_TailorSemanticTextEdit, ...)
        if "bullet_1" in patch_fields:
            field_defs["bullet_1"] = (_TailorSemanticTextEdit, ...)
        if "bullet_2" in patch_fields:
            field_defs["bullet_2"] = (_TailorSemanticTextEdit, ...)
        if "project_swap_reason" in patch_fields:
            if contract.get("project_swap_required"):
                field_defs["project_swap_reason"] = (str, Field(min_length=1, max_length=200))
            else:
                field_defs["project_swap_reason"] = (Literal[None], None)
        return create_model(
            "_TailorResumePatchDraft",
            __config__=ConfigDict(extra="forbid"),
            **field_defs,
        )

    @staticmethod
    def _tailor_patch_required_shape(
        target_job_id: str,
        patch_fields: list[str],
        *,
        has_project_swap: bool,
    ) -> dict[str, Any]:
        shape: dict[str, Any] = {"job_id": target_job_id}
        if "professional_summary" in patch_fields:
            shape["professional_summary"] = {
                "new_text": "<revised summary>",
                "reason": "<reason>",
            }
        for slot in ("bullet_1", "bullet_2"):
            if slot in patch_fields:
                shape[slot] = {"new_text": f"<revised {slot} text>", "reason": "<reason>"}
        if "project_swap_reason" in patch_fields:
            shape["project_swap_reason"] = "<swap reason>" if has_project_swap else None
        return shape

    def _prepare_tailor_patch_recovery(
        self,
        raw_call: NormalizedToolCall,
        contract: dict[str, Any],
        *,
        audit: _DraftAuditResult | None = None,
        error: str | None = None,
    ) -> None:
        if not self._is_complete_tailor_draft(raw_call.arguments):
            self._tailor_patch_recovery = None
            return
        patch_fields = audit.fields if audit and audit.issues else []
        if not patch_fields and error:
            patch_fields = self._patch_fields_from_error(error, contract)
        patch_fields = [
            field
            for field in patch_fields
            if field in _TAILOR_PATCHABLE_FIELDS
        ]
        if not patch_fields:
            self._tailor_patch_recovery = None
            return
        base_draft = copy.deepcopy(raw_call.arguments)
        preserved = [
            field
            for field in (
                "decision_summary",
                "professional_summary",
                "bullet_1",
                "bullet_2",
                "project_swap_reason",
                "plan_rationale",
            )
            if field in base_draft and field not in patch_fields
        ]
        self._tailor_patch_recovery = {
            "base_draft": base_draft,
            "patch_fields": patch_fields,
            "preserved_fields": preserved,
        }

    def _patch_fields_from_error(
        self,
        error: str,
        contract: dict[str, Any],
    ) -> list[str]:
        lowered = error.casefold()
        fields: list[str] = []
        if "professional summary" in lowered or "role phrase" in lowered:
            fields.append("professional_summary")
        slot = self._infer_rejected_bullet_slot(error)
        if slot in {"bullet_1", "bullet_2"}:
            fields.append(slot)
        if "project_swap_reason" in lowered:
            fields.append("project_swap_reason")
        if (
            "unsupported required skill" in lowered
            or "genuine-gap" in lowered
        ) and "professional summary" not in lowered and slot is None:
            fields.append("professional_summary")
        return sorted(set(fields))

    def _clear_tailor_patch_recovery(self) -> None:
        self._tailor_patch_recovery = None

    @staticmethod
    def _cover_letter_skill_citation(skill: str) -> dict[str, Any]:
        canonical = normalize_skill(skill, has_vector_search=True)
        if canonical in {"python", "sql", "bash"}:
            source_id = "EV-SKILL-LANG"
        elif canonical in {
            "retrieval augmented generation",
            "embeddings",
            "vector search",
            "prompt engineering",
        }:
            source_id = "EV-SKILL-GENAI"
        elif canonical in {
            "mlops",
            "docker",
            "mlflow",
            "model monitoring",
            "ci cd",
            "aws",
        }:
            source_id = "EV-SKILL-MLOPS"
        elif canonical in {"rest api", "fastapi", "git", "pytest", "postgresql"}:
            source_id = "EV-SKILL-SYSTEMS"
        else:
            source_id = "EV-SKILL-ML"
        return {
            "source_type": "evidence_registry",
            "source_id": source_id,
            "source_field": "supported_skills",
            "evidence_id": source_id,
        }

    @staticmethod
    def _concise_company_details(job: Any) -> str:
        words = job.company_details.split()
        excerpt = " ".join(words[:24]).rstrip(".,;:")
        return excerpt

    @staticmethod
    def _default_company_hook_phrase(job: Any) -> str:
        return " ".join(job.company_details.split()[:12]).rstrip(".,;:")

    def _build_cover_letter_allowed_skill_registry(
        self,
        job_id: str,
    ) -> dict[str, _AllowedCoverSkill]:
        if job_id in self._cover_letter_skill_registry_cache:
            return self._cover_letter_skill_registry_cache[job_id]
        assert self.registry is not None
        job = self.registry._job(job_id)
        reconciled = self._reconcile_tailoring_evidence(job_id)
        gap_canonical = {
            normalize_skill(gap, has_vector_search=True) or gap.casefold()
            for gap in reconciled.genuine_gaps
        }
        registry: dict[str, _AllowedCoverSkill] = {}
        candidates: list[str] = []
        preferred = _COVER_PREFERRED_SKILLS.get(job.company, [])
        candidates.extend(preferred)
        candidates.extend(reconciled.aligned_skills)
        candidates.extend(reconciled.evidenced_elsewhere_skills)
        for fact in self.registry.memory.facts:
            if fact.fact_type != "skill":
                continue
            if isinstance(fact.normalized_value, str):
                candidates.append(fact.normalized_value)
            candidates.extend(fact.skill_tags)
        for skill in candidates:
            cleaned = " ".join(skill.split())
            if not cleaned:
                continue
            canonical = normalize_skill(cleaned, has_vector_search=True) or cleaned.casefold()
            if canonical in gap_canonical:
                continue
            if canonical in registry:
                continue
            if not _skill_is_job_relevant(cleaned, job):
                continue
            registry[canonical] = _AllowedCoverSkill(
                display_name=cleaned,
                canonical=canonical,
                citation=self._cover_letter_skill_citation(cleaned),
            )
        self._cover_letter_skill_registry_cache[job_id] = registry
        return registry

    def _select_model_visible_skills(self, job_id: str) -> list[str]:
        registry = self._build_cover_letter_allowed_skill_registry(job_id)
        assert self.registry is not None
        job = self.registry._job(job_id)
        reconciled = self._reconcile_tailoring_evidence(job_id)
        ordered: list[str] = []
        seen: set[str] = set()

        def add_skill(surface: str) -> None:
            if len(ordered) >= 8:
                return
            cleaned = " ".join(surface.split())
            if not cleaned:
                return
            canonical = (
                normalize_skill(cleaned, has_vector_search=True) or cleaned.casefold()
            )
            if canonical in seen or canonical not in registry:
                return
            ordered.append(registry[canonical].display_name)
            seen.add(canonical)

        for skill in _COVER_PREFERRED_SKILLS.get(job.company, []):
            add_skill(skill)
        for skill in job.required_skills:
            add_skill(skill)
        for skill in reconciled.aligned_skills:
            add_skill(skill)
        for skill in reconciled.evidenced_elsewhere_skills:
            add_skill(skill)
        for fact in self.registry.memory.facts:
            if fact.fact_type != "skill":
                continue
            if isinstance(fact.normalized_value, str):
                add_skill(fact.normalized_value)
            for tag in fact.skill_tags:
                add_skill(tag)
        for item in sorted(registry.values(), key=lambda entry: entry.display_name):
            add_skill(item.display_name)
        return ordered[:8]

    def _cover_letter_checkpoint(self, job_id: str) -> dict[str, Any]:
        assert self.state is not None
        assert self.registry is not None
        job = self.registry._job(job_id)
        reconciled = self._reconcile_tailoring_evidence(job_id)
        finalized = self.state.finalized_resumes[job_id]
        skill_registry = self._build_cover_letter_allowed_skill_registry(job_id)
        bullet_corpus = self._candidate_bullet_corpus()
        allowed_numeric = sorted(set(_COVER_NUMBER_PATTERN.findall(bullet_corpus)))
        memory_facts = [
            {
                "fact_type": fact.fact_type,
                "statement": fact.statement,
                "normalized_value": fact.normalized_value,
                "skill_tags": fact.skill_tags,
            }
            for fact in self.registry.memory.facts
        ]
        strengths = sorted(
            {
                *reconciled.aligned_skills,
                *reconciled.evidenced_elsewhere_skills,
            }
        )
        return {
            "target_job_id": job_id,
            "rank": self._target_rank(job_id),
            "title": job.title,
            "company": job.company,
            "company_details_excerpt": self._concise_company_details(job),
            "approved_resume_revision": finalized.approved_revision_round,
            "finalized_resume_summary": (
                f"Approved revision {finalized.approved_revision_round} resume "
                f"for {job.title} at {job.company}."
            ),
            "supported_strengths": strengths,
            "allowed_skills": self._select_model_visible_skills(job_id),
            "do_not_claim_skills": reconciled.genuine_gaps,
            "allowed_numeric_claims": allowed_numeric,
            "current_memory_facts": memory_facts,
            "paragraph_count_requirement": "1 or 2",
            "one_page_required": True,
        }

    @staticmethod
    def _cover_draft_required_shape(target_job_id: str) -> dict[str, Any]:
        return {
            "decision_summary": "<concise explanation>",
            "job_id": target_job_id,
            "company_hook_phrase": "<evidence-grounded hook>",
            "body_paragraph_1": {"text": "<paragraph 1>", "reason": "<reason>"},
            "body_paragraph_2": None,
            "skills": ["<3 to 8 allowed skills>"],
            "closing_sentence": "<concise closing>",
            "plan_rationale": "<concise rationale>",
        }

    def _parse_cover_letter_transport(
        self,
        arguments: dict[str, Any],
    ) -> _CoverLetterTransportDraft:
        try:
            return _CoverLetterTransportDraft.model_validate(arguments)
        except ValidationError as exc:
            raise ToolArgumentsError(
                f"Cover letter transport rejection: {exc}"
            ) from exc

    @staticmethod
    def _parse_cover_letter_draft(arguments: dict[str, Any]) -> _CoverLetterTransportDraft:
        try:
            return _CoverLetterTransportDraft.model_validate(arguments)
        except ValidationError as exc:
            raise ToolArgumentsError(
                f"Cover letter transport rejection: {exc}"
            ) from exc

    def _resolve_cover_letter_transport_arguments(
        self,
        call: NormalizedToolCall,
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        target_job_id = contract.get("target_job_id")
        if target_job_id is None:
            raise StateInvariantError(
                "Cover letter hydration rejection: missing deterministic target job"
            )
        arguments = call.arguments
        patch_fields = list(contract.get("cover_patch_fields") or [])
        if patch_fields and self._cover_letter_patch_recovery:
            patch_model = self._cover_patch_model_for_contract(contract)
            try:
                patch = patch_model.model_validate(arguments)
            except ValidationError as exc:
                raise ToolArgumentsError(
                    f"Cover letter transport rejection: {exc}"
                ) from exc
            if patch.job_id != target_job_id:
                raise StateInvariantError(
                    "Cover letter draft job_id mismatch: expected "
                    f"{target_job_id!r}; received {patch.job_id!r}"
                )
            allowed = set(patch_fields) | {"job_id"}
            extra = set(arguments) - allowed
            if extra:
                raise ToolArgumentsError(
                    "Cover letter transport rejection: patch response contains "
                    f"extra fields {sorted(extra)}"
                )
            arguments = self._merge_cover_patch(
                self._cover_letter_patch_recovery["base_draft"],
                patch.model_dump(mode="json"),
                patch_fields,
            )
        transport = self._parse_cover_letter_transport(arguments)
        if transport.job_id != target_job_id:
            raise StateInvariantError(
                "Cover letter draft job_id mismatch: expected "
                f"{target_job_id!r}; received {transport.job_id!r}"
            )
        return transport.model_dump(mode="json")

    def _semantic_cover_letter_draft(
        self,
        transport_args: dict[str, Any],
    ) -> _CoverLetterSemanticDraft:
        try:
            return _CoverLetterSemanticDraft.model_validate(transport_args)
        except ValidationError as exc:
            raise ToolArgumentsError(
                f"Cover letter draft schema rejection: {exc}"
            ) from exc

    @staticmethod
    def _is_complete_cover_draft(arguments: dict[str, Any]) -> bool:
        required = {
            "decision_summary",
            "job_id",
            "company_hook_phrase",
            "body_paragraph_1",
            "skills",
            "closing_sentence",
            "plan_rationale",
        }
        return required.issubset(arguments)

    def _merge_cover_patch(
        self,
        base_draft: dict[str, Any],
        patch: dict[str, Any],
        patch_fields: list[str],
    ) -> dict[str, Any]:
        merged = copy.deepcopy(base_draft)
        for field in patch_fields:
            if field in patch:
                merged[field] = copy.deepcopy(patch[field])
        return merged

    def _cover_patch_model_for_contract(
        self,
        contract: dict[str, Any],
    ) -> type[BaseModel]:
        patch_fields = list(contract.get("cover_patch_fields") or [])
        field_defs: dict[str, Any] = {"job_id": (str, Field(min_length=1))}
        if "company_hook_phrase" in patch_fields:
            field_defs["company_hook_phrase"] = (str, Field(min_length=1))
        if "body_paragraph_1" in patch_fields:
            field_defs["body_paragraph_1"] = (_CoverLetterBodyParagraph, ...)
        if "body_paragraph_2" in patch_fields:
            field_defs["body_paragraph_2"] = (_CoverLetterBodyParagraph | None, ...)
        if "skills" in patch_fields:
            field_defs["skills"] = (list[str], Field(min_length=3, max_length=8))
        if "closing_sentence" in patch_fields:
            field_defs["closing_sentence"] = (str, Field(min_length=1))
        if "plan_rationale" in patch_fields:
            field_defs["plan_rationale"] = (str, Field(min_length=1, max_length=200))
        return create_model(
            "_CoverLetterPatchDraft",
            __config__=ConfigDict(extra="forbid"),
            **field_defs,
        )

    @staticmethod
    def _cover_patch_required_shape(
        target_job_id: str,
        patch_fields: list[str],
    ) -> dict[str, Any]:
        shape: dict[str, Any] = {"job_id": target_job_id}
        if "company_hook_phrase" in patch_fields:
            shape["company_hook_phrase"] = "<revised hook>"
        if "body_paragraph_1" in patch_fields:
            shape["body_paragraph_1"] = {"text": "<paragraph 1>", "reason": "<reason>"}
        if "body_paragraph_2" in patch_fields:
            shape["body_paragraph_2"] = {"text": "<paragraph 2>", "reason": "<reason>"}
        if "skills" in patch_fields:
            shape["skills"] = ["<3 to 8 allowed skills>"]
        if "closing_sentence" in patch_fields:
            shape["closing_sentence"] = "<revised closing>"
        if "plan_rationale" in patch_fields:
            shape["plan_rationale"] = "<revised rationale>"
        return shape

    def _prepare_cover_patch_recovery(
        self,
        raw_call: NormalizedToolCall,
        contract: dict[str, Any],
        *,
        audit: _CoverLetterAuditResult | None = None,
        error: str | None = None,
    ) -> None:
        rejected_category = self._cover_letter_rejection_category(error, audit)
        patch_fields = audit.fields if audit and audit.issues else []
        if not patch_fields and error:
            patch_fields = self._cover_patch_fields_from_error(error, audit)
        patch_fields = [
            field for field in patch_fields if field in _COVER_PATCHABLE_FIELDS
        ]
        if contract.get("cover_recovery_mode") == "patch" and self._cover_letter_patch_recovery:
            self._cover_letter_patch_recovery["rejected_category"] = rejected_category
            return
        if not self._is_complete_cover_draft(raw_call.arguments):
            self._cover_letter_patch_recovery = None
            return
        try:
            self._parse_cover_letter_transport(raw_call.arguments)
        except ToolArgumentsError:
            self._cover_letter_patch_recovery = None
            return
        if not patch_fields:
            self._cover_letter_patch_recovery = None
            return
        base_draft = copy.deepcopy(raw_call.arguments)
        preserved = [
            field
            for field in (
                "decision_summary",
                "company_hook_phrase",
                "body_paragraph_1",
                "body_paragraph_2",
                "skills",
                "closing_sentence",
                "plan_rationale",
            )
            if field in base_draft and field not in patch_fields
        ]
        self._cover_letter_patch_recovery = {
            "base_draft": base_draft,
            "patch_fields": patch_fields,
            "preserved_fields": preserved,
            "rejected_category": rejected_category,
        }

    def _cover_patch_fields_from_error(
        self,
        error: str,
        audit: _CoverLetterAuditResult | None = None,
    ) -> list[str]:
        if audit and audit.issues:
            return audit.fields
        lowered = error.casefold()
        fields: list[str] = []
        if "company_hook_phrase" in lowered or "company hook" in lowered:
            fields.append("company_hook_phrase")
        if (
            "skills" in lowered
            or "between 3 and 8" in lowered
            or "duplicate skill" in lowered
            or "unsupported skill" in lowered
        ):
            fields.append("skills")
        for slot in ("body_paragraph_1", "body_paragraph_2"):
            if slot.replace("_", " ") in lowered or slot in lowered:
                fields.append(slot)
        if "closing_sentence" in lowered:
            fields.append("closing_sentence")
        if "plan_rationale" in lowered:
            fields.append("plan_rationale")
        if "body paragraph" in lowered and "body_paragraph_1" not in fields:
            fields.append("body_paragraph_1")
        return sorted(set(fields))

    @staticmethod
    def _cover_letter_rejection_category(
        error: str | None,
        audit: _CoverLetterAuditResult | None,
    ) -> str:
        if audit and audit.issues:
            categories = {issue.category for issue in audit.issues}
            if "semantic_text" in categories:
                return "semantic_text"
            if "evidence" in categories:
                return "evidence"
            if "target_job" in categories:
                return "target_job"
            if "hydration" in categories:
                return "hydration"
            if "citation" in categories:
                return "citation"
            if "draft_schema" in categories:
                return "draft_schema"
        if error is None:
            return "validation"
        return JobSearchAgentRuntime._classify_cover_letter_rejection(error)

    @staticmethod
    def _cover_letter_should_clear_recovery(
        error: str,
        contract: dict[str, Any],
    ) -> bool:
        if contract.get("cover_recovery_mode") == "patch" and (
            "transport rejection" in error.casefold()
        ):
            return False
        if "transport rejection" in error.casefold():
            return True
        return False

    def _clear_cover_patch_recovery(self) -> None:
        self._cover_letter_patch_recovery = None

    def _paragraph_uses_memory_fact(self, text: str, fact: Any) -> bool:
        normalized_text = _normalize_phrase(text)
        if isinstance(fact.normalized_value, str) and fact.normalized_value.strip():
            if _normalize_phrase(fact.normalized_value) in normalized_text:
                return True
        statement_words = [
            word
            for word in _COVER_WORD_PATTERN.findall(fact.statement or "")
            if word.casefold() not in _COVER_MEANINGFUL_HOOK_WORDS
        ]
        if statement_words:
            matches = sum(
                1 for word in statement_words if word.casefold() in normalized_text
            )
            if matches >= min(3, len(statement_words)):
                return True
        if fact.fact_type == "skill":
            for tag in fact.skill_tags:
                canonical = normalize_skill(tag, has_vector_search=True)
                if canonical and _contains_canonical(text, canonical):
                    return True
        return False

    def _memory_fact_citation(self, fact: Any) -> dict[str, Any]:
        source_field = "skill_tags" if fact.fact_type == "skill" else "statement"
        evidence_id = fact.evidence_refs[0] if fact.evidence_refs else None
        return {
            "source_type": "memory_fact",
            "source_id": fact.fact_id,
            "source_field": source_field,
            "evidence_id": evidence_id,
        }

    def _cover_letter_paragraph_citations(
        self,
        job_id: str,
        paragraph_index: int,
        paragraph_text: str,
    ) -> list[dict[str, Any]]:
        assert self.registry is not None
        citations: list[dict[str, Any]] = [
            {
                "source_type": "job_posting",
                "source_id": job_id,
                "source_field": "job_description",
                "evidence_id": None,
            }
        ]
        if paragraph_index == 0:
            citations.extend(
                [
                    {
                        "source_type": "experience_bullet",
                        "source_id": "exp-primary-bullet-2",
                        "source_field": "text",
                        "evidence_id": "EV-EXP-BULLET-002",
                    },
                    {
                        "source_type": "portfolio_project",
                        "source_id": "proj-carepath-rag",
                        "source_field": "short_description",
                        "evidence_id": "EV-PROJ-001",
                    },
                    {
                        "source_type": "finalized_resume",
                        "source_id": job_id,
                        "source_field": "approved_revision_round",
                        "evidence_id": None,
                    },
                ]
            )
        else:
            citations.extend(
                [
                    {
                        "source_type": "experience_bullet",
                        "source_id": "exp-primary-bullet-3",
                        "source_field": "text",
                        "evidence_id": "EV-EXP-BULLET-003",
                    },
                    {
                        "source_type": "portfolio_project",
                        "source_id": "proj-model-watch",
                        "source_field": "short_description",
                        "evidence_id": "EV-PROJ-003",
                    },
                ]
            )
        for fact in self.registry.memory.facts:
            if self._paragraph_uses_memory_fact(paragraph_text, fact):
                citations.append(self._memory_fact_citation(fact))
        citations.sort(
            key=lambda item: (
                str(item.get("source_type", "")),
                str(item.get("source_id", "")),
                str(item.get("source_field", "")),
                str(item.get("evidence_id") or ""),
            )
        )
        return citations

    def _build_hydrated_cover_letter_plan(
        self,
        transport_args: dict[str, Any],
        job_id: str,
    ) -> dict[str, Any]:
        semantic = self._semantic_cover_letter_draft(transport_args)
        skill_registry = self._build_cover_letter_allowed_skill_registry(job_id)
        body_paragraphs: list[dict[str, Any]] = []
        paragraph_specs = [semantic.body_paragraph_1]
        if semantic.body_paragraph_2 is not None:
            paragraph_specs.append(semantic.body_paragraph_2)
        for index, paragraph in enumerate(paragraph_specs):
            body_paragraphs.append(
                {
                    "text": paragraph.text,
                    "reason": paragraph.reason,
                    "citations": self._cover_letter_paragraph_citations(
                        job_id,
                        index,
                        paragraph.text,
                    ),
                }
            )
        skill_items: list[dict[str, Any]] = []
        for skill in semantic.skills:
            canonical = (
                normalize_skill(skill, has_vector_search=True) or skill.casefold()
            )
            allowed = skill_registry.get(canonical)
            if allowed is None:
                raise StateInvariantError(
                    "Cover letter hydration rejection: "
                    f"unsupported skill {skill!r}"
                )
            skill_items.append(
                {
                    "skill": allowed.display_name,
                    "citations": [copy.deepcopy(allowed.citation)],
                }
            )
        return {
            "job_id": job_id,
            "company_hook_phrase": semantic.company_hook_phrase,
            "company_hook_source_field": "company_details",
            "body_paragraphs": body_paragraphs,
            "skills": skill_items,
            "closing_sentence": semantic.closing_sentence,
            "plan_rationale": semantic.plan_rationale,
            "letter_date": self._cover_letter_date.isoformat(),
        }

    def _hydrate_cover_letter_call(
        self,
        call: NormalizedToolCall,
        contract: dict[str, Any],
        *,
        transport_args: dict[str, Any] | None = None,
    ) -> NormalizedToolCall:
        target_job_id = contract.get("target_job_id")
        if target_job_id is None:
            raise StateInvariantError(
                "Cover letter hydration rejection: missing deterministic target job"
            )
        merged_transport = transport_args or self._resolve_cover_letter_transport_arguments(
            call,
            contract,
        )
        plan_dict = self._build_hydrated_cover_letter_plan(
            merged_transport,
            target_job_id,
        )
        return NormalizedToolCall(
            id=call.id,
            name=call.name,
            arguments={
                "decision_summary": merged_transport["decision_summary"],
                "job_id": target_job_id,
                "plan": plan_dict,
            },
        )

    def _audit_cover_letter_transport(
        self,
        transport_args: dict[str, Any],
        contract: dict[str, Any],
    ) -> _CoverLetterAuditResult:
        assert self.registry is not None
        target_job_id = contract.get("target_job_id")
        if target_job_id is None:
            return _CoverLetterAuditResult(issues=[])
        job = self.registry._job(target_job_id)
        reconciled = self._reconcile_tailoring_evidence(target_job_id)
        skill_registry = self._build_cover_letter_allowed_skill_registry(target_job_id)
        issues: list[_CoverLetterAuditIssue] = []
        if transport_args.get("job_id") != target_job_id:
            issues.append(
                _CoverLetterAuditIssue(
                    field="job_id",
                    category="target_job",
                    message="job_id must equal deterministic target job",
                )
            )
        hook_phrase = str(transport_args.get("company_hook_phrase", ""))
        if hook_phrase and _word_count(hook_phrase) > 15:
            issues.append(
                _CoverLetterAuditIssue(
                    field="company_hook_phrase",
                    category="semantic_text",
                    message="company_hook_phrase must be at most 15 words",
                )
            )
        if hook_phrase and not self._hook_is_grounded(hook_phrase, job):
            issues.append(
                _CoverLetterAuditIssue(
                    field="company_hook_phrase",
                    category="semantic_text",
                    message="company hook must be grounded in company_details",
                )
            )
        hook_words = _COVER_WORD_PATTERN.findall(hook_phrase)
        meaningful = [
            word
            for word in hook_words
            if word.casefold() not in _COVER_MEANINGFUL_HOOK_WORDS
        ]
        if hook_phrase and len(meaningful) < 4:
            issues.append(
                _CoverLetterAuditIssue(
                    field="company_hook_phrase",
                    category="semantic_text",
                    message="company hook requires at least 4 meaningful words",
                )
            )
        paragraph_specs: list[tuple[str, dict[str, Any] | None]] = [
            ("body_paragraph_1", transport_args.get("body_paragraph_1")),
            ("body_paragraph_2", transport_args.get("body_paragraph_2")),
        ]
        present_paragraphs = [
            item for item in paragraph_specs if isinstance(item[1], dict)
        ]
        if not 1 <= len(present_paragraphs) <= 2:
            issues.append(
                _CoverLetterAuditIssue(
                    field="body_paragraph_1",
                    category="draft_schema",
                    message="cover letter requires 1 or 2 body paragraphs",
                )
            )
        bullet_corpus = self._candidate_bullet_corpus()
        allowed_numeric = sorted(set(_COVER_NUMBER_PATTERN.findall(bullet_corpus)))
        allowed_claim_text = bullet_corpus + " " + " ".join(allowed_numeric)
        for field, paragraph in present_paragraphs:
            text = str(paragraph.get("text", ""))
            reason = str(paragraph.get("reason", ""))
            if _word_count(text) > 90:
                issues.append(
                    _CoverLetterAuditIssue(
                        field=field,
                        category="semantic_text",
                        message=f"{field} text must be at most 90 words",
                    )
                )
            if _word_count(reason) > 18:
                issues.append(
                    _CoverLetterAuditIssue(
                        field=field,
                        category="semantic_text",
                        message=f"{field} reason must be at most 18 words",
                    )
                )
            words = _word_count(text)
            if not 35 <= words <= 120:
                issues.append(
                    _CoverLetterAuditIssue(
                        field=field,
                        category="semantic_text",
                        message=(
                            f"{field} must contain 35 to 120 words; found {words}"
                        ),
                    )
                )
            if re.search(
                r"14%\s+improvement\s+in\s+model\s+performance",
                text,
                flags=re.IGNORECASE,
            ):
                issues.append(
                    _CoverLetterAuditIssue(
                        field=field,
                        category="evidence",
                        message=(
                            f"{field} contains unsupported numeric claim "
                            "'14% improvement in model performance'"
                        ),
                    )
                )
            gap = self._text_claims_genuine_gap(text, reconciled)
            if gap is not None:
                issues.append(
                    _CoverLetterAuditIssue(
                        field=field,
                        category="evidence",
                        message=f"{field} claims genuine-gap skill {gap!r}",
                    )
                )
            unsupported = self._text_claims_unsupported_required_skill(
                text,
                target_job_id,
                bullet_corpus=bullet_corpus,
            )
            if unsupported is not None:
                issues.append(
                    _CoverLetterAuditIssue(
                        field=field,
                        category="evidence",
                        message=(
                            f"{field} claims unsupported required skill "
                            f"{unsupported!r}"
                        ),
                    )
                )
            for number in self._unsupported_numbers(text, allowed_claim_text):
                issues.append(
                    _CoverLetterAuditIssue(
                        field=field,
                        category="evidence",
                        message=f"{field} contains unsupported numeric claim {number!r}",
                    )
                )
        skills = transport_args.get("skills")
        if not isinstance(skills, list):
            issues.append(
                _CoverLetterAuditIssue(
                    field="skills",
                    category="draft_schema",
                    message="skills must be a list",
                )
            )
            skills = []
        if not 3 <= len(skills) <= 8:
            issues.append(
                _CoverLetterAuditIssue(
                    field="skills",
                    category="draft_schema",
                    message="skills must contain 3 to 8 items",
                )
            )
        seen_canonical: set[str] = set()
        for skill in skills:
            if not isinstance(skill, str):
                continue
            canonical = normalize_skill(skill, has_vector_search=True) or skill.casefold()
            if canonical in seen_canonical:
                issues.append(
                    _CoverLetterAuditIssue(
                        field="skills",
                        category="draft_schema",
                        message=f"duplicate skill {skill!r}",
                    )
                )
            seen_canonical.add(canonical)
            if canonical not in skill_registry:
                issues.append(
                    _CoverLetterAuditIssue(
                        field="skills",
                        category="evidence",
                        message=f"unsupported skill {skill!r}",
                    )
                )
        closing = str(transport_args.get("closing_sentence", ""))
        if _word_count(closing) > 25:
            issues.append(
                _CoverLetterAuditIssue(
                    field="closing_sentence",
                    category="semantic_text",
                    message="closing_sentence must be at most 25 words",
                )
            )
        closing_words = _word_count(closing)
        if closing and not 3 <= closing_words <= 35:
            issues.append(
                _CoverLetterAuditIssue(
                    field="closing_sentence",
                    category="semantic_text",
                    message="closing_sentence must contain 3 to 35 words",
                )
            )
        rationale = str(transport_args.get("plan_rationale", ""))
        if _word_count(rationale) > 25:
            issues.append(
                _CoverLetterAuditIssue(
                    field="plan_rationale",
                    category="semantic_text",
                    message="plan_rationale must be at most 25 words",
                )
            )
        if not rationale.strip():
            issues.append(
                _CoverLetterAuditIssue(
                    field="plan_rationale",
                    category="draft_schema",
                    message="plan_rationale must be nonempty",
                )
            )
        return _CoverLetterAuditResult(issues=issues)

    def _hook_is_grounded(self, hook_phrase: str, job: Any) -> bool:
        normalized_hook = _normalize_phrase(hook_phrase)
        normalized_details = _normalize_phrase(job.company_details)
        return normalized_hook in normalized_details

    def _audit_hydrated_cover_letter_draft(
        self,
        hydrated_call: NormalizedToolCall,
        contract: dict[str, Any],
    ) -> _CoverLetterAuditResult:
        plan = hydrated_call.arguments.get("plan")
        if not isinstance(plan, dict):
            return _CoverLetterAuditResult(issues=[])
        if plan.get("company_hook_source_field") != "company_details":
            return _CoverLetterAuditResult(
                issues=[
                    _CoverLetterAuditIssue(
                        field="company_hook_phrase",
                        category="hydration",
                        message="company_hook_source_field must be company_details",
                    )
                ]
            )
        return _CoverLetterAuditResult(issues=[])

    @staticmethod
    def _classify_cover_letter_rejection(error: str) -> str:
        lowered = error.casefold()
        if "transport rejection" in lowered:
            return "draft_schema"
        if "draft schema rejection" in lowered:
            return "draft_schema"
        if "draft job_id mismatch" in lowered or "outer job_id must equal" in lowered:
            return "target_job"
        if "hydration rejection" in lowered:
            return "hydration"
        if "company hook" in lowered or "company_hook" in lowered:
            if "grounded" in lowered or "meaningful" in lowered:
                return "semantic_text"
        if "draft audit rejection" in lowered:
            if "semantic_text" in lowered or "company hook" in lowered:
                return "semantic_text"
            if "genuine-gap" in lowered or "unsupported" in lowered:
                return "evidence"
            if "duplicate skill" in lowered or "3 to 8" in lowered:
                return "draft_schema"
        if (
            "genuine-gap" in lowered
            or "unsupported numeric" in lowered
            or "unsupported skill" in lowered
            or "unsupported required skill" in lowered
        ):
            return "evidence"
        if any(marker in lowered for marker in _COVER_CITATION_ERROR_MARKERS):
            return "citation"
        if "rejected the requested call" in lowered or "protected" in lowered:
            return "tool_execution"
        if "one page" in lowered or "one-page" in lowered:
            return "tool_execution"
        return "validation"

    def _record_cover_hydration_diagnostics(
        self,
        raw_call: NormalizedToolCall,
        hydrated_call: NormalizedToolCall,
        contract: dict[str, Any],
    ) -> None:
        raw_size = len(
            json.dumps(raw_call.arguments, ensure_ascii=False, separators=(",", ":"))
        )
        hydrated_size = len(
            json.dumps(
                hydrated_call.arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        plan = hydrated_call.arguments.get("plan", {})
        citation_count = 0
        if isinstance(plan, dict):
            for paragraph in plan.get("body_paragraphs", []) or []:
                if isinstance(paragraph, dict):
                    citation_count += len(paragraph.get("citations") or [])
            for skill in plan.get("skills", []) or []:
                if isinstance(skill, dict):
                    citation_count += len(skill.get("citations") or [])
        target_job_id = contract.get("target_job_id")
        allowed_count = 0
        selected_count = 0
        if target_job_id:
            allowed_count = len(
                self._build_cover_letter_allowed_skill_registry(target_job_id)
            )
            if isinstance(plan, dict) and isinstance(plan.get("skills"), list):
                selected_count = len(plan["skills"])
        diagnostics = {
            "cover_letter_compact_draft": True,
            "cover_letter_hydration_applied": True,
            "model_argument_mode": "cover_letter_text_draft",
            "raw_model_argument_char_count": raw_size,
            "hydrated_argument_char_count": hydrated_size,
            "cover_letter_allowed_skill_count": allowed_count,
            "cover_letter_selected_skill_count": selected_count,
            "cover_letter_citation_count": citation_count,
            "phase": contract["phase"],
            "target_rank": contract.get("target_rank"),
        }
        if self._last_generation_span is not None:
            existing = self._last_generation_span.record.metadata or {}
            self._last_generation_span.record.metadata = {**existing, **diagnostics}

    @contextmanager
    def _cover_letter_execution_context(
        self,
        job_id: str,
    ) -> Iterator[FitAnalysisResult]:
        assert self.state is not None
        raw_analysis = self.state.fit_analyses[job_id]
        execution_analysis = self._build_resume_execution_fit_analysis(job_id)
        self.state.fit_analyses[job_id] = execution_analysis
        try:
            yield execution_analysis
        finally:
            self.state.fit_analyses[job_id] = raw_analysis

    @staticmethod
    def _tailor_draft_model_for_contract(
        contract: dict[str, Any],
    ) -> type[_TailorResumeTextDraftBase]:
        if contract.get("project_swap_required"):
            return _TailorResumeTextDraftWithSwap
        return _TailorResumeTextDraftNoSwap

    def _parse_tailor_draft_arguments(
        self,
        arguments: dict[str, Any],
        contract: dict[str, Any],
    ) -> _TailorResumeTextDraftBase:
        draft_model = self._tailor_draft_model_for_contract(contract)
        try:
            return draft_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolArgumentsError(
                f"Tailor resume draft schema rejection: {exc}"
            ) from exc

    @staticmethod
    def _classify_tailor_rejection(error: str) -> str:
        lowered = error.casefold()
        if "draft schema rejection" in lowered:
            return "draft_schema"
        if "draft job_id mismatch" in lowered or "outer job_id must equal" in lowered:
            return "target_job"
        if "hydration rejection" in lowered:
            return "hydration"
        if "must explicitly include" in lowered and "role phrase" in lowered:
            return "semantic_text"
        if "must clearly align with the actual job role" in lowered:
            return "semantic_text"
        if (
            "genuine-gap" in lowered
            or "unsupported numeric" in lowered
            or "introduces capability not supported" in lowered
            or "unsupported required skill" in lowered
            or "draft audit rejection" in lowered
        ):
            return "evidence"
        if any(marker in lowered for marker in _CITATION_ERROR_MARKERS):
            return "citation"
        if "rejected the requested call" in lowered:
            if (
                "unsupported required skill" in lowered
                or "genuine-gap" in lowered
                or "unsupported numeric" in lowered
                or "introduces capability not supported" in lowered
            ):
                return "evidence"
            if "protected" in lowered or "one-page" in lowered or "one page" in lowered:
                return "tool_execution"
            return "tool_execution"
        if "protected" in lowered or "one-page" in lowered or "one page" in lowered:
            return "tool_execution"
        return "validation"

    def _infer_rejected_bullet_slot(self, error: str) -> str | None:
        bullets = self._editable_tailoring_bullets()[:2]
        for index, bullet in enumerate(bullets, start=1):
            if bullet.bullet_id in error:
                return f"bullet_{index}"
        lowered = error.casefold()
        if "bullet_1" in lowered:
            return "bullet_1"
        if "bullet_2" in lowered:
            return "bullet_2"
        return None

    def _validate_hydrated_tailor_semantics(
        self,
        hydrated_call: NormalizedToolCall,
        contract: dict[str, Any],
    ) -> None:
        self._audit_hydrated_tailor_draft(hydrated_call, contract).raise_if_issues()

    def _compact_bullet_slot_source(
        self,
        bullet: ExperienceBullet,
        reconciled: _ReconciledTailoringEvidence,
    ) -> dict[str, Any]:
        allowed_numeric = sorted(
            set(_TAILOR_NUMBER_PATTERN.findall(bullet.text))
        )
        supported_concepts: list[str] = []
        for skill in [*reconciled.aligned_skills, *reconciled.evidenced_elsewhere_skills]:
            canonical = normalize_skill(skill, has_vector_search=True)
            if canonical and _contains_canonical(bullet.text, canonical):
                supported_concepts.append(skill)
        return {
            "current_text": bullet.text,
            "supported_concepts": supported_concepts,
            "allowed_numeric_claims": allowed_numeric,
        }

    def _tailoring_bullet_slot_source(
        self,
        bullet: ExperienceBullet,
        reconciled: _ReconciledTailoringEvidence,
    ) -> dict[str, Any]:
        return self._compact_bullet_slot_source(bullet, reconciled)

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

    @staticmethod
    def _tailor_draft_required_shape(
        target_job_id: str,
        *,
        has_project_swap: bool,
    ) -> dict[str, Any]:
        return {
            "decision_summary": "<concise explanation>",
            "job_id": target_job_id,
            "professional_summary": {
                "new_text": "<tailored summary>",
                "reason": "<reason>",
            },
            "bullet_1": {"new_text": "<tailored bullet 1 text>", "reason": "<reason>"},
            "bullet_2": {"new_text": "<tailored bullet 2 text>", "reason": "<reason>"},
            "project_swap_reason": "<swap reason>" if has_project_swap else None,
            "plan_rationale": "<concise rationale>",
        }

    def _model_schemas_for_contract(
        self,
        contract: dict[str, Any],
    ) -> list[dict[str, Any]]:
        assert self.registry is not None
        allowed_tool = contract["allowed_tool"]
        if allowed_tool == "tailor_resume":
            if contract.get("tailor_patch_fields"):
                draft_model = self._patch_model_for_contract(contract)
            else:
                draft_model = self._tailor_draft_model_for_contract(contract)
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "tailor_resume",
                        "description": AssignmentToolRegistry._DESCRIPTIONS[
                            "tailor_resume"
                        ],
                        "parameters": draft_model.model_json_schema(),
                    },
                }
            ]
        if allowed_tool == "generate_cover_letter":
            if contract.get("cover_patch_fields"):
                draft_model = self._cover_patch_model_for_contract(contract)
            else:
                draft_model = _CoverLetterTransportDraft
            schema = draft_model.model_json_schema()
            if contract.get("cover_patch_fields"):
                schema["required"] = sorted(schema.get("properties", {}).keys())
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "generate_cover_letter",
                        "description": AssignmentToolRegistry._DESCRIPTIONS[
                            "generate_cover_letter"
                        ],
                        "parameters": schema,
                    },
                }
            ]
        return self.registry.model_schemas([allowed_tool])

    def _build_project_swap_citations(
        self,
        job_id: str,
        swap_suggestion: Any,
    ) -> list[dict[str, Any]]:
        assert self.registry is not None
        bundle = self.registry.bundle
        remove_project = next(
            project
            for project in bundle.all_projects()
            if project.project_id == swap_suggestion.remove_project_id
        )
        add_project = next(
            project
            for project in bundle.all_projects()
            if project.project_id == swap_suggestion.add_project_id
        )
        return [
            self._job_posting_citation(job_id),
            {
                "source_type": "portfolio_project",
                "source_id": remove_project.project_id,
                "source_field": "project_id",
                "evidence_id": remove_project.evidence_ids[0],
            },
            {
                "source_type": "portfolio_project",
                "source_id": add_project.project_id,
                "source_field": "project_id",
                "evidence_id": add_project.evidence_ids[0],
            },
            {
                "source_type": "fit_analysis",
                "source_id": job_id,
                "source_field": "projects.swap_suggestion",
                "supported_claim": swap_suggestion.reason,
                "evidence_id": None,
            },
        ]

    def _hydrate_tailor_resume_call(
        self,
        call: NormalizedToolCall,
        contract: dict[str, Any],
    ) -> NormalizedToolCall:
        assert self.state is not None
        assert self.registry is not None
        target_job_id = contract.get("target_job_id")
        if target_job_id is None:
            raise StateInvariantError(
                "Tailor resume hydration rejection: missing deterministic target job"
            )
        arguments = call.arguments
        if contract.get("tailor_patch_fields") and self._tailor_patch_recovery:
            patch_model = self._patch_model_for_contract(contract)
            try:
                patch = patch_model.model_validate(arguments)
            except ValidationError as exc:
                raise ToolArgumentsError(
                    f"Tailor resume draft schema rejection: {exc}"
                ) from exc
            if patch.job_id != target_job_id:
                raise StateInvariantError(
                    "Tailor resume draft job_id mismatch: expected "
                    f"{target_job_id!r}; received {patch.job_id!r}"
                )
            arguments = self._merge_tailor_patch(
                self._tailor_patch_recovery["base_draft"],
                patch.model_dump(mode="json"),
                list(contract["tailor_patch_fields"]),
            )
        try:
            draft = self._parse_tailor_draft_arguments(arguments, contract)
        except ToolArgumentsError:
            raise
        if draft.job_id != target_job_id:
            raise StateInvariantError(
                "Tailor resume draft job_id mismatch: expected "
                f"{target_job_id!r}; received {draft.job_id!r}"
            )
        editable_bullets = self._editable_tailoring_bullets()[:2]
        if len(editable_bullets) != 2:
            raise AgentRuntimeError(
                "Tailor resume hydration rejection: expected exactly two editable bullets"
            )
        citation_contract = self._build_citation_contract(target_job_id)
        expected_swap = self.state.fit_analyses[
            target_job_id
        ].projects.swap_suggestion
        project_swap: dict[str, Any] | None
        if expected_swap is None:
            if draft.project_swap_reason is not None:
                raise ToolArgumentsError(
                    "Tailor resume hydration rejection: project_swap_reason must "
                    "be null when no swap is required"
                )
            project_swap = None
        else:
            if draft.project_swap_reason is None or not draft.project_swap_reason.strip():
                raise ToolArgumentsError(
                    "Tailor resume hydration rejection: project_swap_reason is "
                    "required when Fit Analysis requires a swap"
                )
            project_swap = {
                "remove_project_id": expected_swap.remove_project_id,
                "add_project_id": expected_swap.add_project_id,
                "reason": draft.project_swap_reason.strip(),
                "citations": self._build_project_swap_citations(
                    target_job_id,
                    expected_swap,
                ),
            }
        hydrated_arguments = {
            "decision_summary": draft.decision_summary,
            "job_id": target_job_id,
            "edit_plan": {
                "job_id": target_job_id,
                "professional_summary": {
                    "new_text": draft.professional_summary.new_text,
                    "reason": draft.professional_summary.reason,
                    "citations": copy.deepcopy(
                        citation_contract["summary_required_citations"]
                    ),
                },
                "experience_bullet_edits": [
                    {
                        "bullet_id": bullet.bullet_id,
                        "new_text": (
                            draft.bullet_1.new_text
                            if index == 0
                            else draft.bullet_2.new_text
                        ),
                        "reason": (
                            draft.bullet_1.reason
                            if index == 0
                            else draft.bullet_2.reason
                        ),
                        "citations": copy.deepcopy(
                            citation_contract["bullet_required_citations"][
                                bullet.bullet_id
                            ]
                        ),
                    }
                    for index, bullet in enumerate(editable_bullets)
                ],
                "skill_section_edits": [],
                "project_swap": project_swap,
                "plan_rationale": draft.plan_rationale,
            },
        }
        return NormalizedToolCall(
            id=call.id,
            name=call.name,
            arguments=hydrated_arguments,
        )

    def _record_hydration_diagnostics(
        self,
        raw_call: NormalizedToolCall,
        hydrated_call: NormalizedToolCall,
        contract: dict[str, Any],
    ) -> None:
        raw_size = len(
            json.dumps(raw_call.arguments, ensure_ascii=False, separators=(",", ":"))
        )
        hydrated_size = len(
            json.dumps(
                hydrated_call.arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        diagnostics = {
            "model_argument_mode": "tailor_resume_text_draft",
            "semantic_bullet_slot_mode": "named",
            "raw_model_argument_char_count": raw_size,
            "hydrated_argument_char_count": hydrated_size,
            "hydration_applied": True,
            "phase": contract["phase"],
            "target_rank": contract.get("target_rank"),
            "required_role_phrase": contract.get("required_role_phrase"),
            "project_swap_required": contract.get("project_swap_required"),
        }
        if self._last_generation_span is not None:
            existing = self._last_generation_span.record.metadata or {}
            self._last_generation_span.record.metadata = {
                **existing,
                **diagnostics,
            }

    @staticmethod
    def _is_semantic_text_recovery_error(error: str) -> bool:
        return JobSearchAgentRuntime._classify_tailor_rejection(error) == "semantic_text"

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
        job = self.registry._job(job_id)
        reconciled = self._reconcile_tailoring_evidence(job_id)
        editable_bullets = self._editable_tailoring_bullets()[:2]
        bullet_sources = [
            self._tailoring_bullet_slot_source(bullet, reconciled)
            for bullet in editable_bullets
        ]
        swap = self.state.fit_analyses[job_id].projects.swap_suggestion
        memory_facts = [
            {
                "fact_type": fact.fact_type,
                "normalized_value": fact.normalized_value,
            }
            for fact in self.registry.memory.facts
        ]
        return {
            "target_job_id": job_id,
            "rank": self._target_rank(job_id),
            "title": job.title,
            "company": job.company,
            "required_role_phrase": self._derive_required_role_phrase(job.title),
            "project_swap_required": swap is not None,
            "project_swap_hint": swap.reason if swap is not None else None,
            "aligned_skills": reconciled.aligned_skills,
            "evidenced_elsewhere_skills": reconciled.evidenced_elsewhere_skills,
            "genuine_gaps": reconciled.genuine_gaps,
            "do_not_claim_skills": reconciled.genuine_gaps,
            "bullet_1_source": bullet_sources[0] if bullet_sources else None,
            "bullet_2_source": bullet_sources[1] if len(bullet_sources) > 1 else None,
            "current_memory_facts": memory_facts,
            "revision_feedback": revision_feedback,
        }

    def _cover_letter_context(self, job_id: str) -> dict[str, Any]:
        return self._cover_letter_checkpoint(job_id)

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
            job = self.registry._job(target_job_id)
            expected_swap = analysis.projects.swap_suggestion
            target_context = self._tailoring_context(
                target_job_id,
                revision_feedback=revision_feedback,
            )
            if self._tailor_patch_recovery:
                patch_fields = self._tailor_patch_recovery["patch_fields"]
                required_shape = self._tailor_patch_required_shape(
                    target_job_id,
                    patch_fields,
                    has_project_swap=expected_swap is not None,
                )
            else:
                required_shape = self._tailor_draft_required_shape(
                    target_job_id,
                    has_project_swap=expected_swap is not None,
                )
            required_role_phrase = self._derive_required_role_phrase(job.title)
            project_swap_required = expected_swap is not None
            constraints = [
                item.replace("TARGET_JOB_ID", target_job_id)
                for item in TAILOR_RESUME_NORMAL_CONSTRAINTS
            ]
        else:
            assert target_job_id is not None
            if self._cover_letter_patch_recovery:
                patch_fields = self._cover_letter_patch_recovery["patch_fields"]
                required_shape = self._cover_patch_required_shape(
                    target_job_id,
                    patch_fields,
                )
            else:
                required_shape = self._cover_draft_required_shape(target_job_id)
            constraints = [
                item.replace("TARGET_JOB_ID", target_job_id)
                for item in COVER_LETTER_NORMAL_CONSTRAINTS
            ]
            target_context = self._cover_letter_checkpoint(target_job_id)

        contract: dict[str, Any] = {
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
        if allowed_tool == "tailor_resume":
            assert target_job_id is not None
            job = self.registry._job(target_job_id)
            contract["required_role_phrase"] = self._derive_required_role_phrase(
                job.title
            )
            contract["project_swap_required"] = (
                state.fit_analyses[target_job_id].projects.swap_suggestion is not None
            )
            if self._tailor_patch_recovery:
                contract["tailor_patch_fields"] = list(
                    self._tailor_patch_recovery["patch_fields"]
                )
                contract["tailor_recovery_mode"] = "patch"
            else:
                contract["tailor_recovery_mode"] = "full"
        if allowed_tool == "generate_cover_letter":
            if self._cover_letter_patch_recovery:
                contract["cover_patch_fields"] = list(
                    self._cover_letter_patch_recovery["patch_fields"]
                )
                contract["cover_recovery_mode"] = "patch"
            else:
                contract["cover_recovery_mode"] = "full"
        return contract

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
            if not self._conversation_ends_with_bounded_recovery():
                if self._phase_needs_compaction(contract["phase"]):
                    self._apply_conversation_checkpoint(contract)
                else:
                    self._append_state_snapshot(contract)
            response = self._call_model(self.trace, contract)
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

    def _conversation_ends_with_bounded_recovery(self) -> bool:
        if not self.conversation:
            return False
        content = self.conversation[-1].get("content")
        if not isinstance(content, str):
            return False
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return False
        if not isinstance(payload, dict):
            return False
        return payload.get("type") in {
            "tool_call_retry",
            "invalid_tool_call_recovery",
        }

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
        schemas = self._model_schemas_for_contract(contract)
        call_number = self.state.model_call_count + 1
        payload_diagnostics = self._model_call_diagnostics(contract, schemas)
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
            metadata={
                "model_call_number": call_number,
                **payload_diagnostics,
            },
            observation_type="generation",
        ) as span:
            self._last_generation_span = span
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
            self._rebuild_bounded_recovery(
                contract,
                error=error,
                tool_call=call,
                length_limited=length_limited,
            )
            self._last_generation_span = None
            return 0, 1

        raw_call = response.tool_calls[0]
        execution_call = raw_call
        audit: _DraftAuditResult | _CoverLetterAuditResult | None = None
        try:
            if raw_call.name != contract["allowed_tool"]:
                raise StateInvariantError(
                    f"Expected only {contract['allowed_tool']!r}; "
                    f"received {raw_call.name!r}"
                )
            if contract["allowed_tool"] == "tailor_resume":
                execution_call = self._hydrate_tailor_resume_call(raw_call, contract)
                self._record_hydration_diagnostics(
                    raw_call,
                    execution_call,
                    contract,
                )
                audit = self._audit_hydrated_tailor_draft(execution_call, contract)
                if self._last_generation_span is not None:
                    existing = self._last_generation_span.record.metadata or {}
                    self._last_generation_span.record.metadata = {
                        **existing,
                        "draft_audit_issue_count": len(audit.issues),
                        "draft_audit_fields": audit.fields,
                    }
                audit.raise_if_issues()
            elif contract["allowed_tool"] == "generate_cover_letter":
                transport_args = self._resolve_cover_letter_transport_arguments(
                    raw_call,
                    contract,
                )
                audit = self._audit_cover_letter_transport(transport_args, contract)
                if self._last_generation_span is not None:
                    existing = self._last_generation_span.record.metadata or {}
                    self._last_generation_span.record.metadata = {
                        **existing,
                        "cover_letter_audit_issue_count": len(audit.issues),
                        "cover_letter_audit_fields": audit.fields,
                    }
                audit.raise_if_issues()
                execution_call = self._hydrate_cover_letter_call(
                    raw_call,
                    contract,
                    transport_args=transport_args,
                )
                self._record_cover_hydration_diagnostics(
                    raw_call,
                    execution_call,
                    contract,
                )
            self._validate_call_for_contract(execution_call, contract)
            baseline_arguments = self.registry.parse_arguments(
                execution_call.name,
                execution_call.arguments,
            )
            self.state.validate_tool_call(
                execution_call.name,
                job_id=getattr(baseline_arguments, "job_id", None),
            )
            self._append_assistant_message(response)
            self._execute_model_tool_call(execution_call)
            if contract["allowed_tool"] == "tailor_resume":
                self._clear_tailor_patch_recovery()
            if contract["allowed_tool"] == "generate_cover_letter":
                self._clear_cover_patch_recovery()
        except (StateInvariantError, ToolRegistryError) as exc:
            error = str(exc)
            if contract["allowed_tool"] == "tailor_resume":
                self._report_validation_rejection_progress(
                    error, contract, audit=audit
                )
                if audit is None and raw_call.name == contract["allowed_tool"]:
                    try:
                        hydrated = self._hydrate_tailor_resume_call(
                            raw_call,
                            contract,
                        )
                        audit = self._audit_hydrated_tailor_draft(
                            hydrated,
                            contract,
                        )
                    except (StateInvariantError, ToolRegistryError):
                        audit = None
                self._prepare_tailor_patch_recovery(
                    raw_call,
                    contract,
                    audit=audit,
                    error=error,
                )
                if self._tailor_patch_recovery:
                    contract = {
                        **contract,
                        "tailor_patch_fields": self._tailor_patch_recovery[
                            "patch_fields"
                        ],
                        "tailor_recovery_mode": "patch",
                        "required_argument_shape": self._tailor_patch_required_shape(
                            contract["target_job_id"],
                            self._tailor_patch_recovery["patch_fields"],
                            has_project_swap=contract.get(
                                "project_swap_required",
                                False,
                            ),
                        ),
                    }
            elif contract["allowed_tool"] == "generate_cover_letter":
                self._report_validation_rejection_progress(
                    error, contract, audit=audit
                )
                if audit is None and raw_call.name == contract["allowed_tool"]:
                    try:
                        transport_args = self._resolve_cover_letter_transport_arguments(
                            raw_call,
                            contract,
                        )
                        audit = self._audit_cover_letter_transport(
                            transport_args,
                            contract,
                        )
                    except (StateInvariantError, ToolRegistryError):
                        audit = None
                self._prepare_cover_patch_recovery(
                    raw_call,
                    contract,
                    audit=audit,
                    error=error,
                )
                if self._cover_letter_patch_recovery:
                    contract = {
                        **contract,
                        "cover_patch_fields": self._cover_letter_patch_recovery[
                            "patch_fields"
                        ],
                        "cover_recovery_mode": "patch",
                        "required_argument_shape": self._cover_patch_required_shape(
                            contract["target_job_id"],
                            self._cover_letter_patch_recovery["patch_fields"],
                        ),
                    }
            self._record_invalid(tool_call=raw_call, error=error)
            self._rebuild_bounded_recovery(
                contract,
                error=error,
                tool_call=raw_call,
                audit=audit,
            )
            return 0, 1
        finally:
            self._last_generation_span = None
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
    def _draft_argument_diagnostics(
        call: NormalizedToolCall | None,
    ) -> dict[str, Any]:
        if call is None:
            return {
                "missing_fields": ["tool_call"],
                "extra_fields": [],
                "misplaced_fields": [],
                "invalid_bullet_counts": [],
            }
        expected = {
            "decision_summary",
            "job_id",
            "professional_summary",
            "bullet_1",
            "bullet_2",
            "project_swap_reason",
            "plan_rationale",
        }
        outer_keys = set(call.arguments)
        summary = call.arguments.get("professional_summary")
        summary_keys = set(summary) if isinstance(summary, dict) else set()
        invalid_bullet_counts: list[str] = []
        if "experience_bullet_edits" in outer_keys:
            invalid_bullet_counts.append("experience_bullet_edits")
        for slot in ("bullet_1", "bullet_2"):
            if slot not in outer_keys:
                invalid_bullet_counts.append(f"missing_{slot}")
        return {
            "missing_fields": sorted(expected - outer_keys),
            "extra_fields": sorted(outer_keys - expected),
            "misplaced_fields": sorted(
                key
                for key in outer_keys
                if key
                in {
                    "edit_plan",
                    "plan",
                    "citations",
                    "skill_section_edits",
                    "experience_bullet_edits",
                }
            ),
            "invalid_bullet_counts": invalid_bullet_counts,
            "summary_missing_fields": sorted(
                {"new_text", "reason"} - summary_keys
            ),
        }

    def _argument_diagnostics(
        self,
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
        if contract["allowed_tool"] == "tailor_resume":
            draft = self._draft_argument_diagnostics(call)
            return {
                "missing_fields": draft["missing_fields"],
                "extra_fields": draft["extra_fields"],
                "misplaced_fields": draft["misplaced_fields"],
                "replacement_schema_keys": draft["invalid_bullet_counts"],
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
            if call.name == "tailor_resume":
                job_id = call.arguments.get("job_id")
                if not isinstance(job_id, str) or not job_id:
                    raise StateInvariantError(
                        "tailor_resume execution requires a deterministic job_id"
                    )
                reconciled = self._reconcile_tailoring_evidence(job_id)
                with self._resume_execution_context(job_id) as (
                    _execution_analysis,
                    _execution_bundle,
                    bundle_metadata,
                ):
                    existing = span.record.metadata or {}
                    span.record.metadata = {
                        **existing,
                        "reconciled_execution_evidence": True,
                        "raw_fit_analysis_preserved": True,
                        "reconciled_execution_bundle": True,
                        "raw_candidate_bundle_preserved": True,
                        "execution_aligned_skill_count": len(
                            reconciled.aligned_skills
                        ),
                        "execution_gap_count": len(reconciled.genuine_gaps),
                        "execution_conflict_count": (
                            reconciled.category_conflicts_resolved
                        ),
                        **bundle_metadata,
                    }
                    outcome = self.registry.execute(
                        call.name,
                        call.arguments,
                        tool_call_id=call.id,
                        revision_round=revision_round,
                        review_feedback=review_feedback,
                        trace_parent=span,
                    )
            elif call.name == "generate_cover_letter":
                job_id = call.arguments.get("job_id")
                if not isinstance(job_id, str) or not job_id:
                    raise StateInvariantError(
                        "generate_cover_letter execution requires a deterministic job_id"
                    )
                reconciled = self._reconcile_tailoring_evidence(job_id)
                with self._cover_letter_execution_context(job_id) as _execution_analysis:
                    existing = span.record.metadata or {}
                    span.record.metadata = {
                        **existing,
                        "reconciled_execution_evidence": True,
                        "raw_fit_analysis_preserved": True,
                        "execution_aligned_skill_count": len(
                            reconciled.aligned_skills
                        ),
                        "execution_gap_count": len(reconciled.genuine_gaps),
                        "execution_conflict_count": (
                            reconciled.category_conflicts_resolved
                        ),
                    }
                    outcome = self.registry.execute(
                        call.name,
                        call.arguments,
                        tool_call_id=call.id,
                        revision_round=revision_round,
                        review_feedback=review_feedback,
                        trace_parent=span,
                    )
            else:
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

    def _build_compact_invalid_recovery(
        self,
        tool_call: NormalizedToolCall | None,
        error: str,
        contract: dict[str, Any],
        *,
        audit: _DraftAuditResult | None = None,
    ) -> dict[str, Any]:
        if contract["allowed_tool"] == "tailor_resume":
            diagnostics = self._draft_argument_diagnostics(tool_call)
            error_category = self._classify_tailor_rejection(error)
            if audit and audit.issues:
                categories = {issue.category for issue in audit.issues}
                if "semantic_text" in categories:
                    error_category = "semantic_text"
                elif "evidence" in categories:
                    error_category = "evidence"
                elif "hydration" in categories:
                    error_category = "hydration"
            draft_issues = (
                diagnostics["missing_fields"]
                or diagnostics["extra_fields"]
                or diagnostics["misplaced_fields"]
                or diagnostics["invalid_bullet_counts"]
                or diagnostics.get("summary_missing_fields")
            )
            if draft_issues and error_category == "validation":
                error_category = "draft_schema"
            if error_category == "draft_schema":
                self._clear_tailor_patch_recovery()
            patch_fields = contract.get("tailor_patch_fields") or []
            patch_mode = bool(patch_fields) and error_category != "draft_schema"
            if patch_mode:
                instruction = (
                    "Return exactly one tailor_resume patch call containing only "
                    "the listed invalid fields plus job_id. Python preserves all "
                    "other valid fields and deterministic IDs."
                )
                if "professional_summary" in patch_fields and contract.get(
                    "required_role_phrase"
                ):
                    phrase = contract["required_role_phrase"]
                    instruction += (
                        f' The professional summary must explicitly include the '
                        f'role phrase "{phrase}".'
                    )
                payload: dict[str, Any] = {
                    "type": "invalid_tool_call_recovery",
                    "error_category": error_category,
                    "allowed_tool": "tailor_resume",
                    "target_job_id": contract.get("target_job_id"),
                    "error": error[:500],
                    "instruction": instruction,
                    "tailor_recovery_mode": "patch",
                    "patch_fields": patch_fields,
                    "required_argument_shape": contract["required_argument_shape"],
                }
                if self._tailor_patch_recovery:
                    payload["preserved_field_count"] = len(
                        self._tailor_patch_recovery.get("preserved_fields", [])
                    )
                if audit and audit.issues:
                    payload["draft_audit_fields"] = audit.fields
                    payload["issues"] = [
                        issue.model_dump(mode="json") for issue in audit.issues
                    ]
                target_context = contract.get("target_context") or {}
                for field in patch_fields:
                    if field in {"bullet_1", "bullet_2"}:
                        source = target_context.get(f"{field}_source")
                        if source is not None:
                            payload[f"{field}_source"] = source
                    if field == "professional_summary" and contract.get(
                        "required_role_phrase"
                    ):
                        payload["required_role_phrase"] = contract[
                            "required_role_phrase"
                        ]
                    if field == "project_swap_reason":
                        payload["project_swap_required"] = contract.get(
                            "project_swap_required",
                            False,
                        )
                bullet_slots = [
                    field
                    for field in patch_fields
                    if field in {"bullet_1", "bullet_2"}
                ]
                if len(bullet_slots) == 1:
                    payload["rejected_bullet_slot"] = bullet_slots[0]
                elif audit and audit.issues:
                    for issue in audit.issues:
                        if issue.bullet_slot:
                            payload["rejected_bullet_slot"] = issue.bullet_slot
                            break
                return payload
            if error_category == "semantic_text":
                required_phrase = contract.get("required_role_phrase")
                if "must explicitly include the role phrase" in error and required_phrase:
                    instruction = (
                        f'The professional summary must explicitly include the role '
                        f'phrase "{required_phrase}". Revise semantic text only.'
                    )
                else:
                    instruction = (
                        "Revise only the semantic text fields. Python will supply "
                        "the same exact IDs and citations on retry."
                    )
            elif error_category == "evidence":
                slot = self._infer_rejected_bullet_slot(error)
                if slot:
                    instruction = (
                        f"Revise {slot} only from {slot}_source. Do not transfer "
                        "metrics, technologies, outcomes, or claims between "
                        "bullet_1 and bullet_2."
                    )
                else:
                    instruction = (
                        "Revise only the semantic text fields. Python will supply "
                        "the same exact IDs and citations on retry."
                    )
            elif error_category == "hydration" and "project_swap_reason" in error:
                instruction = (
                    "project_swap_reason must be a nonempty string when Fit "
                    "Analysis requires a project swap. Python supplies the exact "
                    "project IDs."
                )
            elif error_category in {"draft_schema", "hydration", "target_job"}:
                instruction = (
                    "Return exactly one corrected compact tailor_resume call "
                    "using the checkpoint."
                )
            else:
                instruction = (
                    "Return exactly one corrected compact tailor_resume call "
                    "using the checkpoint."
                )
            payload: dict[str, Any] = {
                "type": "invalid_tool_call_recovery",
                "error_category": error_category,
                "allowed_tool": "tailor_resume",
                "target_job_id": contract.get("target_job_id"),
                "error": error[:500],
                "instruction": instruction,
            }
            if error_category == "semantic_text" and contract.get("required_role_phrase"):
                payload["required_role_phrase"] = contract["required_role_phrase"]
            if error_category == "evidence":
                slot = self._infer_rejected_bullet_slot(error)
                if slot:
                    payload["rejected_bullet_slot"] = slot
                    target_context = contract.get("target_context") or {}
                    slot_source = target_context.get(f"{slot}_source")
                    if slot_source is not None:
                        payload[f"{slot}_source"] = slot_source
            if error_category == "hydration" and "project_swap_reason" in error:
                payload["project_swap_required"] = contract.get(
                    "project_swap_required",
                    True,
                )
            if error_category in {"draft_schema", "hydration", "target_job", "validation"} and draft_issues:
                payload["field_diagnostics"] = diagnostics
                payload["required_argument_shape"] = contract["required_argument_shape"]
            return payload

        if contract["allowed_tool"] == "generate_cover_letter":
            error_category = self._classify_cover_letter_rejection(error)
            if audit and getattr(audit, "issues", None):
                categories = {issue.category for issue in audit.issues}
                if "semantic_text" in categories:
                    error_category = "semantic_text"
                elif "evidence" in categories:
                    error_category = "evidence"
                elif "hydration" in categories:
                    error_category = "hydration"
                elif "target_job" in categories:
                    error_category = "target_job"
                elif "draft_schema" in categories:
                    error_category = "draft_schema"
                elif "citation" in categories:
                    error_category = "citation"
            if "draft schema rejection" in error.casefold():
                error_category = "draft_schema"
            if self._cover_letter_should_clear_recovery(error, contract):
                self._clear_cover_patch_recovery()
            patch_fields = list(
                (self._cover_letter_patch_recovery or {}).get("patch_fields")
                or contract.get("cover_patch_fields")
                or []
            )
            patch_mode = bool(self._cover_letter_patch_recovery and patch_fields)
            if patch_mode:
                contract = {
                    **contract,
                    "cover_patch_fields": patch_fields,
                    "cover_recovery_mode": "patch",
                    "required_argument_shape": self._cover_patch_required_shape(
                        contract["target_job_id"],
                        patch_fields,
                    ),
                }
                payload = {
                    "type": "invalid_tool_call_recovery",
                    "error_category": error_category,
                    "allowed_tool": "generate_cover_letter",
                    "target_job_id": contract.get("target_job_id"),
                    "error": error[:500],
                    "instruction": (
                        "Return exactly one generate_cover_letter patch call "
                        "containing only the listed invalid fields plus job_id. "
                        "Python preserves all other valid fields and injects "
                        "citations deterministically."
                    ),
                    "cover_recovery_mode": "patch",
                    "patch_fields": patch_fields,
                    "required_argument_shape": contract["required_argument_shape"],
                }
                if self._cover_letter_patch_recovery:
                    payload["cover_letter_preserved_field_count"] = len(
                        self._cover_letter_patch_recovery.get("preserved_fields", [])
                    )
                if audit and getattr(audit, "issues", None):
                    payload["cover_letter_audit_fields"] = audit.fields
                    payload["issues"] = [
                        issue.model_dump(mode="json") for issue in audit.issues
                    ]
                return payload
            instruction = (
                "Return exactly one corrected compact generate_cover_letter call "
                "using the checkpoint."
            )
            if error_category == "semantic_text":
                instruction = (
                    "Revise only the listed semantic text fields. Python injects "
                    "company_hook_source_field, citations, and IDs."
                )
            elif error_category == "evidence":
                instruction = (
                    "Revise only unsupported claims or skills from allowed_skills. "
                    "Do not claim do_not_claim_skills."
                )
            payload = {
                "type": "invalid_tool_call_recovery",
                "error_category": error_category,
                "allowed_tool": "generate_cover_letter",
                "target_job_id": contract.get("target_job_id"),
                "error": error[:500],
                "instruction": instruction,
            }
            if error_category in {"draft_schema", "hydration", "target_job"}:
                payload["required_argument_shape"] = contract["required_argument_shape"]
            return payload

        diagnostics = self._argument_diagnostics(tool_call, contract)
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
        citation_issues = (
            contract["allowed_tool"] == "tailor_resume"
            and self._is_citation_error(error)
        )
        if structure_issues and citation_issues:
            error_category = "structural_and_citation"
        elif structure_issues:
            error_category = "structural"
        elif citation_issues:
            error_category = "citation"
        else:
            error_category = "validation"

        payload: dict[str, Any] = {
            "type": "invalid_tool_call_recovery",
            "error_category": error_category,
            "allowed_tool": contract["allowed_tool"],
            "target_job_id": contract.get("target_job_id"),
            "error": error[:500],
            "instruction": "Return exactly one corrected tool call using the checkpoint.",
        }
        if structure_issues:
            payload["field_diagnostics"] = diagnostics
            target_job_id = contract.get("target_job_id")
            template = copy.deepcopy(TAILOR_RESUME_ARGUMENT_TEMPLATE)
            if target_job_id:
                template["job_id"] = target_job_id
                template["edit_plan"]["job_id"] = target_job_id
            payload["exact_tailor_resume_structural_template"] = template
        target_job_id = contract.get("target_job_id")
        if citation_issues and target_job_id:
            payload["citation_recovery_contract"] = (
                self._build_citation_recovery_contract(target_job_id)
            )
            payload["instruction"] = (
                "Copy every supplied citation identity field exactly and "
                "return one corrected tool call using the checkpoint."
            )
        return payload

    def _rebuild_bounded_recovery(
        self,
        contract: dict[str, Any],
        *,
        error: str,
        tool_call: NormalizedToolCall | None = None,
        length_limited: bool = False,
        audit: _DraftAuditResult | None = None,
    ) -> None:
        messages: list[dict[str, Any]] = [
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
        if tool_call is None:
            reason = "generation_limit" if length_limited else "missing_tool_call"
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "type": "tool_call_retry",
                            "reason": reason,
                            "allowed_tool": contract["allowed_tool"],
                            "target_job_id": contract.get("target_job_id"),
                            "instruction": (
                                "Return exactly one complete tool call using "
                                "the checkpoint."
                            ),
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        self._build_compact_invalid_recovery(
                            tool_call,
                            error,
                            contract,
                            audit=audit,
                        ),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
        self.conversation = messages

    def _append_invalid_message(
        self,
        tool_call: NormalizedToolCall | None,
        error: str,
        contract: dict[str, Any],
    ) -> None:
        self._rebuild_bounded_recovery(
            contract,
            error=error,
            tool_call=tool_call,
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
                self._rebuild_bounded_recovery(
                    contract,
                    error=error,
                    tool_call=call,
                    length_limited=length_limited,
                )
                self._last_generation_span = None
                invalid_turns += 1
            else:
                raw_call = response.tool_calls[0]
                execution_call = raw_call
                audit: _DraftAuditResult | None = None
                try:
                    if raw_call.name != contract["allowed_tool"]:
                        raise StateInvariantError(
                            f"Expected only {contract['allowed_tool']!r}; "
                            f"received {raw_call.name!r}"
                        )
                    execution_call = self._hydrate_tailor_resume_call(
                        raw_call,
                        contract,
                    )
                    self._record_hydration_diagnostics(
                        raw_call,
                        execution_call,
                        contract,
                    )
                    audit = self._audit_hydrated_tailor_draft(execution_call, contract)
                    if self._last_generation_span is not None:
                        existing = self._last_generation_span.record.metadata or {}
                        self._last_generation_span.record.metadata = {
                            **existing,
                            "draft_audit_issue_count": len(audit.issues),
                            "draft_audit_fields": audit.fields,
                        }
                    audit.raise_if_issues()
                    self._validate_call_for_contract(execution_call, contract)
                    self.registry.parse_arguments(
                        execution_call.name,
                        execution_call.arguments,
                    )
                    self._append_assistant_message(response)
                    outcome = self._execute_model_tool_call(
                        execution_call,
                        revision_round=next_revision_round,
                        review_feedback=review_comments,
                        trace_parent=trace_parent,
                    )
                except (StateInvariantError, ToolRegistryError) as exc:
                    error = str(exc)
                    if audit is None and raw_call.name == contract["allowed_tool"]:
                        try:
                            hydrated = self._hydrate_tailor_resume_call(
                                raw_call,
                                contract,
                            )
                            audit = self._audit_hydrated_tailor_draft(
                                hydrated,
                                contract,
                            )
                        except (StateInvariantError, ToolRegistryError):
                            audit = None
                    self._report_validation_rejection_progress(
                        error, contract, audit=audit
                    )
                    self._prepare_tailor_patch_recovery(
                        raw_call,
                        contract,
                        audit=audit,
                        error=error,
                    )
                    if self._tailor_patch_recovery:
                        contract = {
                            **contract,
                            "tailor_patch_fields": self._tailor_patch_recovery[
                                "patch_fields"
                            ],
                            "tailor_recovery_mode": "patch",
                            "required_argument_shape": self._tailor_patch_required_shape(
                                contract["target_job_id"],
                                self._tailor_patch_recovery["patch_fields"],
                                has_project_swap=contract.get(
                                    "project_swap_required",
                                    False,
                                ),
                            ),
                        }
                    self._record_invalid(tool_call=raw_call, error=error)
                    self._rebuild_bounded_recovery(
                        contract,
                        error=error,
                        tool_call=raw_call,
                        audit=audit,
                    )
                    invalid_turns += 1
                else:
                    self.state.consecutive_invalid_call_count = 0
                    self._clear_tailor_patch_recovery()
                    return outcome.result
                finally:
                    self._last_generation_span = None
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
