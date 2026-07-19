"""The one production LLM runtime and its continuous tool-calling loop."""

from __future__ import annotations

import copy
import hashlib
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
    CoverLetterEvidenceError,
    CoverLetterParagraph,
    CoverLetterPlan,
    CoverLetterSkillItem,
    _MEANINGLESS_HOOK_WORDS,
    _citation_supports_skill,
    _normalize_phrase,
    _skill_is_job_relevant,
    _validate_required_skill_claims,
    validate_cover_letter_plan,
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

_COVER_HOOK_MISSION_TERMS = frozenset(
    {
        "advance",
        "build",
        "create",
        "deliver",
        "develop",
        "drive",
        "enable",
        "help",
        "improve",
        "innovate",
        "lead",
        "leverage",
        "offer",
        "partner",
        "power",
        "provide",
        "serve",
        "support",
        "transform",
    }
)
_COVER_HOOK_IMPACT_TERMS = frozenset(
    {
        "ai",
        "analytics",
        "commercial",
        "customer",
        "customers",
        "data",
        "federal",
        "impact",
        "innovation",
        "mission",
        "ml",
        "platform",
        "product",
        "products",
        "service",
        "services",
        "solution",
        "solutions",
        "technology",
    }
)
_COVER_HOOK_GENERIC_FILLER = frozenset(
    {
        "company",
        "corporation",
        "enterprise",
        "firm",
        "global",
        "group",
        "inc",
        "industries",
        "leader",
        "leading",
        "llc",
        "nation",
        "organization",
        "provider",
        "services",
    }
)
_COVER_HOOK_MAX_OPTIONS = 6
_COVER_HOOK_MIN_OPTIONS = 1

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
    """Assembled cover letter paragraph after deterministic claim insertion."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    selected_candidate_claim: str | None = None

    @field_validator("text", "reason")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("text fields must be nonempty")
        return value


class _CoverLetterBodyParagraphWrapper(BaseModel):
    """Model-authored paragraph wrapper before deterministic assembly."""

    model_config = ConfigDict(extra="forbid")

    lead_in: str = Field(min_length=1)
    selected_candidate_claim: str = Field(min_length=1)
    follow_up: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("lead_in", "follow_up", "reason", "selected_candidate_claim")
    @classmethod
    def strip_wrapper_fields(cls, value: str) -> str:
        value = " ".join(str(value).split())
        if not value:
            raise ValueError("wrapper fields must be nonempty")
        return value


class _CoverLetterTransportDraft(BaseModel):
    """Compact transport layer for assembled cover letter drafts."""

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
        "closing_sentence",
        "plan_rationale",
    )
    @classmethod
    def strip_required_strings(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("required string fields must be nonempty")
        return value

    @field_validator("company_hook_phrase")
    @classmethod
    def strip_hook_exact(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("company_hook_phrase must be nonempty")
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
    code: str | None = None


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


class _AllowedCandidateClaim(BaseModel):
    claim_text: str
    citations: list[dict[str, Any]]
    numeric_tokens: list[str] = Field(default_factory=list)


_COVER_MAX_ALLOWED_CLAIMS = 5
_COVER_LEAD_IN_MIN_WORDS = 3
_COVER_LEAD_IN_MAX_WORDS = 30
_COVER_FOLLOW_UP_MIN_WORDS = 5
_COVER_FOLLOW_UP_MAX_WORDS = 45
_COVER_WRAPPER_BULLET_MARKERS = ("•", "·", "- ", "* ", "1.", "2.")
_COVER_ASSEMBLED_PARAGRAPH_MIN_WORDS = 35
_COVER_ASSEMBLED_PARAGRAPH_MAX_WORDS = 120
_COVER_REASON_MAX_WORDS = 18
_COVER_NORMALIZED_REASON = (
    "Connects verified candidate evidence to the role requirements."
)
_COVER_WRAPPER_LEAD_TIERS = (
    "My background aligns with the practical responsibilities of this role.",
    "I am interested in applying my technical background to the responsibilities of this role.",
    (
        "I am interested in bringing an evidence-driven technical approach to the "
        "collaborative responsibilities of this role."
    ),
)
_COVER_WRAPPER_FOLLOW_TIERS = (
    "This experience is directly relevant to the position.",
    (
        "This evidence demonstrates practical experience relevant to the position "
        "and its technical responsibilities."
    ),
    (
        "This evidence demonstrates a disciplined and practical approach that aligns "
        "with the collaborative technical responsibilities of the position."
    ),
)
_COVER_WRAPPER_REPAIRABLE_CODES = frozenset(
    {
        "lead_in_too_short",
        "lead_in_too_long",
        "follow_up_too_short",
        "follow_up_too_long",
        "reason_too_long",
        "assembled_paragraph_too_short",
        "assembled_paragraph_too_long",
    }
)
_COVER_CITATION_SOURCE_ORDER = {
    "job_posting": 0,
    "experience": 1,
    "experience_bullet": 2,
    "portfolio_project": 3,
    "education": 4,
    "master_skill": 5,
    "evidence_registry": 6,
    "memory_fact": 7,
    "candidate_profile": 8,
    "finalized_resume": 9,
    "company_details": 10,
    "fit_analysis": 11,
}
_COVER_NUMERIC_DRIFT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"14%\s+(?:improvement\s+in\s+)?(?:query\s+response\s+)?accuracy",
            re.IGNORECASE,
        ),
        "14% metric must preserve validated recall at constant false-positive rate",
    ),
    (
        re.compile(r"14%\s+improvement\s+in\s+model\s+performance", re.IGNORECASE),
        "14% metric must preserve validated recall at constant false-positive rate",
    ),
    (
        re.compile(r"36%\s+(?:improvement\s+in\s+)?model\s+performance", re.IGNORECASE),
        "36% metric must preserve median analyst lookup-time reduction",
    ),
    (
        re.compile(r"87%\s+(?:top[- ]three\s+)?precision", re.IGNORECASE),
        "87% metric must preserve top-three retrieval over synthetic questions",
    ),
    (
        re.compile(r"12%\s+(?:prediction\s+)?accuracy\s+improvement", re.IGNORECASE),
        "12% metric must preserve MAE improvement over seasonal-naive baseline",
    ),
]


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


class DuplicateInvalidOutputError(AgentRuntimeError):
    """Raised when identical invalid model output repeats without safe recovery."""


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
        self._cover_letter_hook_cache: dict[str, tuple[list[str], bool]] = {}
        self._cover_letter_claim_catalog_cache: dict[str, list[_AllowedCandidateClaim]] = (
            {}
        )
        self._cover_letter_skill_support_cache: dict[str, dict[str, list[dict[str, Any]]]] = (
            {}
        )
        self._cover_letter_schema_model_cache: dict[tuple[Any, ...], type[BaseModel]] = (
            {}
        )
        self._cover_letter_last_invalid_fingerprint: str | None = None
        self._cover_letter_invalid_repeat_count = 0
        self._cover_letter_wrapper_repair_meta: dict[str, Any] | None = None
        self._cover_letter_preexecution_meta: dict[str, Any] | None = None
        self._cover_letter_claim_skill_validation_map: dict[str, str] | None = None
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
                "allowed_company_hooks",
                "allowed_candidate_claims",
                "cover_letter_hook_extraction_fallback_used",
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
                allowed_hooks = contract.get("allowed_company_hooks")
                if allowed_hooks is None:
                    allowed_hooks = self._select_allowed_company_hooks(target_job_id)
                diagnostics["cover_letter_allowed_hook_count"] = len(allowed_hooks)
                diagnostics["cover_letter_hook_enum_applied"] = bool(allowed_hooks)
                diagnostics["cover_letter_hook_extraction_fallback_used"] = (
                    contract.get("cover_letter_hook_extraction_fallback_used", False)
                )
                diagnostics["cover_letter_allowed_claim_count"] = len(
                    self._build_allowed_candidate_claims(target_job_id)
                )
            if contract.get("cover_recovery_mode") == "patch":
                diagnostics["cover_letter_patch_recovery_applied"] = True
                diagnostics["cover_letter_patch_fields"] = contract.get(
                    "cover_patch_fields",
                    [],
                )
                diagnostics["cover_letter_recovery_affected_fields"] = contract.get(
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
                    diagnostics["cover_letter_recovery_source_category"] = (
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

    def _validated_memory_fact_evidence_id(self, fact: Any) -> str | None:
        assert self.registry is not None
        for ref in fact.evidence_refs or []:
            if not isinstance(ref, str) or not ref.strip():
                continue
            record = self.registry.bundle.get_evidence(ref)
            if record is None:
                continue
            if "cover_letter" not in record.allowed_uses:
                continue
            return ref
        return None

    def _memory_fact_citation(self, fact: Any, paragraph_text: str) -> dict[str, Any]:
        if fact.fact_type == "skill":
            source_field = "skill_tags"
        elif isinstance(fact.normalized_value, str) and fact.normalized_value.strip():
            normalized_value = fact.normalized_value.strip()
            if normalized_value.casefold() in paragraph_text.casefold():
                source_field = "normalized_value"
            else:
                source_field = "statement"
        else:
            source_field = "statement"
        return {
            "source_type": "memory_fact",
            "source_id": fact.fact_id,
            "source_field": source_field,
            "evidence_id": self._validated_memory_fact_evidence_id(fact),
        }

    @staticmethod
    def _citation_identity(citation: dict[str, Any]) -> tuple[Any, ...]:
        return (
            citation.get("source_type"),
            citation.get("source_id"),
            citation.get("source_field"),
            citation.get("evidence_id"),
        )

    @classmethod
    def _sort_cover_citations(
        cls,
        citations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return sorted(citations, key=cls._citation_sort_key)

    @staticmethod
    def _citation_sort_key(citation: dict[str, Any]) -> tuple[Any, ...]:
        return (
            _COVER_CITATION_SOURCE_ORDER.get(str(citation.get("source_type", "")), 99),
            str(citation.get("source_id", "")),
            str(citation.get("source_field", "")),
            str(citation.get("evidence_id") or ""),
        )

    @staticmethod
    def _dedupe_cover_citations(
        citations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        seen: set[tuple[Any, ...]] = set()
        deduped: list[dict[str, Any]] = []
        for citation in citations:
            identity = JobSearchAgentRuntime._citation_identity(citation)
            if identity in seen:
                continue
            seen.add(identity)
            deduped.append(citation)
        return deduped

    def _resolve_validated_skill_citation(self, skill: str) -> dict[str, Any] | None:
        assert self.registry is not None
        canonical = normalize_skill(skill, has_vector_search=True) or skill.casefold()
        candidates: list[dict[str, Any]] = []
        for fact in self.registry.memory.facts:
            if fact.fact_type != "skill":
                continue
            candidates.append(self._memory_fact_citation(fact, skill))
        for experience in self.registry.bundle.profile.experience:
            for bullet in experience.bullets:
                for evidence_id in bullet.evidence_ids:
                    record = self.registry.bundle.get_evidence(evidence_id)
                    if record is None or "cover_letter" not in record.allowed_uses:
                        continue
                    candidates.append(
                        {
                            "source_type": "experience_bullet",
                            "source_id": bullet.bullet_id,
                            "source_field": "text",
                            "evidence_id": evidence_id,
                        }
                    )
        for project in self.registry.bundle.all_projects():
            for evidence_id in project.evidence_ids:
                record = self.registry.bundle.get_evidence(evidence_id)
                if record is None or "cover_letter" not in record.allowed_uses:
                    continue
                for field in ("short_description", "measurable_result"):
                    candidates.append(
                        {
                            "source_type": "portfolio_project",
                            "source_id": project.project_id,
                            "source_field": field,
                            "evidence_id": evidence_id,
                        }
                    )
        candidates.append(self._cover_letter_skill_citation(skill))
        for candidate in candidates:
            citation = CoverLetterCitation.model_validate(candidate)
            if _citation_supports_skill(
                citation,
                canonical,
                bundle=self.registry.bundle,
                memory=self.registry.memory,
            ):
                return candidate
        return None

    def _build_skill_support_catalog(
        self,
        job_id: str,
    ) -> dict[str, list[dict[str, Any]]]:
        assert self.registry is not None
        if job_id in getattr(self, "_cover_letter_skill_support_cache", {}):
            return self._cover_letter_skill_support_cache[job_id]
        catalog: dict[str, list[dict[str, Any]]] = {}
        registry = self._build_cover_letter_allowed_skill_registry(job_id)
        for entry in registry.values():
            canonical = entry.canonical
            validated = self._resolve_validated_skill_citation(entry.display_name)
            if validated is None:
                continue
            catalog.setdefault(canonical, []).append(copy.deepcopy(validated))
        if not hasattr(self, "_cover_letter_skill_support_cache"):
            self._cover_letter_skill_support_cache = {}
        self._cover_letter_skill_support_cache[job_id] = catalog
        return catalog

    def _build_allowed_candidate_claims(
        self,
        job_id: str,
    ) -> list[_AllowedCandidateClaim]:
        if job_id in self._cover_letter_claim_catalog_cache:
            return self._cover_letter_claim_catalog_cache[job_id]
        assert self.registry is not None
        job = self.registry._job(job_id)
        reconciled = self._reconcile_tailoring_evidence(job_id)
        preferred_skills = {
            normalize_skill(skill, has_vector_search=True) or skill.casefold()
            for skill in [
                *job.required_skills,
                *reconciled.aligned_skills,
                *reconciled.evidenced_elsewhere_skills,
            ]
        }
        entries: list[tuple[int, str, list[dict[str, Any]]]] = []

        def relevance_score(claim_text: str) -> int:
            score = 0
            for canonical in preferred_skills:
                if canonical and _contains_canonical(claim_text, canonical):
                    score += 1
            return score

        def add_entry(claim_text: str, citations: list[dict[str, Any]]) -> None:
            cleaned = " ".join(claim_text.split())
            if not cleaned:
                return
            validated: list[dict[str, Any]] = []
            for citation in citations:
                try:
                    CoverLetterCitation.model_validate(citation)
                    validated.append(copy.deepcopy(citation))
                except Exception:
                    continue
            if not validated:
                return
            entries.append((-relevance_score(cleaned), cleaned, validated))

        memory_entries: list[tuple[int, str, list[dict[str, Any]]]] = []
        for fact in sorted(
            self.registry.memory.facts,
            key=lambda item: item.fact_id,
        ):
            if fact.fact_type == "skill":
                continue
            claim_text = None
            if isinstance(fact.normalized_value, str) and fact.normalized_value.strip():
                claim_text = fact.normalized_value.strip()
            elif fact.statement and fact.statement.strip():
                claim_text = fact.statement.strip()
            if claim_text is None:
                continue
            memory_entries.append(
                (0, claim_text, [self._memory_fact_citation(fact, claim_text)])
            )

        for experience in self.registry.bundle.profile.experience:
            for bullet in experience.bullets:
                citations = [
                    {
                        "source_type": "experience_bullet",
                        "source_id": bullet.bullet_id,
                        "source_field": "text",
                        "evidence_id": evidence_id,
                    }
                    for evidence_id in bullet.evidence_ids
                    if self.registry.bundle.get_evidence(evidence_id) is not None
                    and "cover_letter"
                    in self.registry.bundle.get_evidence(evidence_id).allowed_uses
                ]
                if citations:
                    add_entry(bullet.text, citations[:1])

        for project in sorted(
            self.registry.bundle.all_projects(),
            key=lambda item: item.project_id,
        ):
            field = "measurable_result"
            value = getattr(project, field, None)
            if not isinstance(value, str) or not value.strip():
                field = "short_description"
                value = getattr(project, field, None)
            if not isinstance(value, str) or not value.strip():
                continue
            evidence_id = project.evidence_ids[0] if project.evidence_ids else None
            if evidence_id is None:
                continue
            record = self.registry.bundle.get_evidence(evidence_id)
            if record is None or "cover_letter" not in record.allowed_uses:
                continue
            add_entry(
                value,
                [
                    {
                        "source_type": "portfolio_project",
                        "source_id": project.project_id,
                        "source_field": field,
                        "evidence_id": evidence_id,
                    }
                ],
            )

        seen_claims: set[str] = set()
        claims: list[_AllowedCandidateClaim] = []
        ordered_entries = memory_entries + sorted(
            entries,
            key=lambda item: (item[0], item[1]),
        )
        for _, claim_text, citations in ordered_entries:
            if claim_text in seen_claims:
                continue
            seen_claims.add(claim_text)
            numeric_tokens = sorted(set(_COVER_NUMBER_PATTERN.findall(claim_text)))
            claims.append(
                _AllowedCandidateClaim(
                    claim_text=claim_text,
                    citations=citations,
                    numeric_tokens=numeric_tokens,
                )
            )
            if len(claims) >= _COVER_MAX_ALLOWED_CLAIMS:
                break
        self._cover_letter_claim_catalog_cache[job_id] = claims
        return claims

    def _allowed_candidate_claim_texts(self, job_id: str) -> list[str]:
        return [entry.claim_text for entry in self._build_allowed_candidate_claims(job_id)]

    def _detect_allowed_claims_in_text(
        self,
        text: str,
        job_id: str,
    ) -> list[_AllowedCandidateClaim]:
        matches: list[_AllowedCandidateClaim] = []
        for entry in sorted(
            self._build_allowed_candidate_claims(job_id),
            key=lambda item: len(item.claim_text),
            reverse=True,
        ):
            if entry.claim_text in text:
                matches.append(entry)
        return matches

    def _mentioned_required_skills(self, text: str, job: Any) -> list[str]:
        normalized_text = normalize_skill(text, has_vector_search=True)
        mentioned: list[str] = []
        for required in job.required_skills:
            canonical = normalize_skill(required, has_vector_search=True)
            if not canonical:
                continue
            if re.search(rf"(?<!\w){re.escape(canonical)}(?!\w)", normalized_text):
                mentioned.append(required)
        return mentioned

    def _skill_item_citation_lookup(
        self,
        skill_items: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        lookup: dict[str, list[dict[str, Any]]] = {}
        for item in skill_items:
            skill = str(item.get("skill", ""))
            canonical = normalize_skill(skill, has_vector_search=True) or skill.casefold()
            citations = item.get("citations") or []
            if isinstance(citations, list) and citations:
                lookup[canonical] = [copy.deepcopy(citations[0])]
        return lookup

    def _paragraph_skill_support_citation(
        self,
        skill: str,
        job_id: str,
        skill_item_lookup: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        canonical = normalize_skill(skill, has_vector_search=True) or skill.casefold()
        if canonical in skill_item_lookup:
            return copy.deepcopy(skill_item_lookup[canonical][0])
        catalog = self._build_skill_support_catalog(job_id)
        citations = catalog.get(canonical) or []
        return copy.deepcopy(citations[0]) if citations else None

    def _close_paragraph_citations(
        self,
        job_id: str,
        paragraph_text: str,
        skill_items: list[dict[str, Any]],
        *,
        selected_claim: str | None = None,
    ) -> list[dict[str, Any]]:
        assert self.registry is not None
        job = self.registry._job(job_id)
        citations: list[dict[str, Any]] = [
            {
                "source_type": "job_posting",
                "source_id": job_id,
                "source_field": "job_description",
                "evidence_id": None,
            }
        ]
        if selected_claim:
            for entry in self._build_allowed_candidate_claims(job_id):
                if entry.claim_text == selected_claim:
                    citations.extend(copy.deepcopy(entry.citations))
                    break
        else:
            for entry in self._detect_allowed_claims_in_text(paragraph_text, job_id):
                citations.extend(copy.deepcopy(entry.citations))
        for fact in self.registry.memory.facts:
            if self._paragraph_uses_memory_fact(paragraph_text, fact):
                citations.append(self._memory_fact_citation(fact, paragraph_text))
        skill_item_lookup = self._skill_item_citation_lookup(skill_items)
        for required in self._mentioned_required_skills(paragraph_text, job):
            support = self._paragraph_skill_support_citation(
                required,
                job_id,
                skill_item_lookup,
            )
            if support is not None:
                citations.append(support)
        if not any(
            citation.get("source_type") in {"experience", "experience_bullet"}
            for citation in citations
        ):
            for entry in self._detect_allowed_claims_in_text(paragraph_text, job_id):
                for citation in entry.citations:
                    if citation.get("source_type") == "experience_bullet":
                        citations.append(copy.deepcopy(citation))
                        break
                else:
                    continue
                break
        if not any(
            citation.get("source_type")
            in {
                "portfolio_project",
                "education",
                "master_skill",
                "evidence_registry",
                "memory_fact",
            }
            for citation in citations
        ):
            for entry in self._build_allowed_candidate_claims(job_id):
                for citation in entry.citations:
                    if citation.get("source_type") in {
                        "portfolio_project",
                        "education",
                        "master_skill",
                        "evidence_registry",
                        "memory_fact",
                    }:
                        citations.append(copy.deepcopy(citation))
                        break
                else:
                    continue
                break
        return self._sort_cover_citations(self._dedupe_cover_citations(citations))

    def _cover_body_source_types(
        self,
        body_paragraphs: list[dict[str, Any]],
    ) -> set[str]:
        types: set[str] = set()
        for paragraph in body_paragraphs:
            for citation in paragraph.get("citations") or []:
                if isinstance(citation, dict):
                    source_type = citation.get("source_type")
                    if isinstance(source_type, str):
                        types.add(source_type)
        return types

    def _ensure_cover_body_source_requirements(
        self,
        job_id: str,
        body_paragraphs: list[dict[str, Any]],
    ) -> None:
        if not body_paragraphs:
            return
        types = self._cover_body_source_types(body_paragraphs)
        needs_experience = not types.intersection({"experience", "experience_bullet"})
        needs_additional = not types.intersection(
            {
                "portfolio_project",
                "education",
                "master_skill",
                "evidence_registry",
                "memory_fact",
            }
        )
        if not needs_experience and not needs_additional:
            return
        first = body_paragraphs[0]
        citations = list(first.get("citations") or [])
        if needs_experience:
            for entry in self._build_allowed_candidate_claims(job_id):
                for citation in entry.citations:
                    if citation.get("source_type") == "experience_bullet":
                        citations.append(copy.deepcopy(citation))
                        break
                else:
                    continue
                break
        if needs_additional:
            for entry in self._build_allowed_candidate_claims(job_id):
                for citation in entry.citations:
                    if citation.get("source_type") in {
                        "portfolio_project",
                        "education",
                        "master_skill",
                        "evidence_registry",
                        "memory_fact",
                    }:
                        citations.append(copy.deepcopy(citation))
                        break
                else:
                    continue
                break
        first["citations"] = self._sort_cover_citations(
            self._dedupe_cover_citations(citations)
        )

    def _audit_numeric_claim_integrity(
        self,
        text: str,
        job_id: str,
    ) -> str | None:
        numbers = sorted(set(_COVER_NUMBER_PATTERN.findall(text)))
        if not numbers:
            return None
        for pattern, message in _COVER_NUMERIC_DRIFT_PATTERNS:
            if pattern.search(text):
                return message
        catalog = self._build_allowed_candidate_claims(job_id)
        for number in numbers:
            matching = [
                entry
                for entry in catalog
                if number in entry.numeric_tokens or number in entry.claim_text
            ]
            if not matching:
                return f"unsupported numeric claim {number!r}"
            if not any(entry.claim_text in text for entry in matching):
                return (
                    f"numeric claim {number!r} must copy an allowed_candidate_claim exactly"
                )
        return None

    def _paragraph_has_allowed_candidate_claim(self, text: str, job_id: str) -> bool:
        return bool(self._detect_allowed_claims_in_text(text, job_id))

    @staticmethod
    def _count_exact_substring_occurrences(text: str, substring: str) -> int:
        if not substring:
            return 0
        count = 0
        start = 0
        while True:
            index = text.find(substring, start)
            if index == -1:
                return count
            count += 1
            start = index + len(substring)

    @staticmethod
    def _cover_citation_locator_key(citation: dict[str, Any]) -> tuple[Any, ...]:
        return (
            citation.get("source_type"),
            citation.get("source_id"),
            citation.get("source_field"),
            citation.get("evidence_id"),
        )

    @staticmethod
    def _selected_claims_from_transport(
        transport_args: dict[str, Any],
    ) -> list[str | None]:
        claims: list[str | None] = []
        for field in ("body_paragraph_1", "body_paragraph_2"):
            paragraph = transport_args.get(field)
            if not isinstance(paragraph, dict):
                continue
            selected = paragraph.get("selected_candidate_claim")
            if isinstance(selected, str) and selected.strip():
                claims.append(selected)
            else:
                claims.append(None)
        return claims

    def _claim_catalog_entry(
        self,
        selected_claim: str,
        job_id: str,
    ) -> _AllowedCandidateClaim | None:
        for entry in self._build_allowed_candidate_claims(job_id):
            if entry.claim_text == selected_claim:
                return entry
        return None

    def _split_paragraph_exact_claim_span(
        self,
        text: str,
        selected_claim: str,
    ) -> tuple[str, str, str]:
        occurrences = self._count_exact_substring_occurrences(text, selected_claim)
        if occurrences == 0:
            raise StateInvariantError(
                "Cover letter pre-execution validation rejection: exact_claim_missing"
            )
        if occurrences > 1:
            raise StateInvariantError(
                "Cover letter pre-execution validation rejection: exact_claim_repeated"
            )
        index = text.index(selected_claim)
        prefix = text[:index]
        suffix = text[index + len(selected_claim) :]
        return prefix, selected_claim, suffix

    def _paragraph_free_text_for_skill_scan(
        self,
        text: str,
        selected_claim: str,
    ) -> str:
        prefix, _, suffix = self._split_paragraph_exact_claim_span(text, selected_claim)
        return " ".join(part for part in (prefix.strip(), suffix.strip()) if part)

    def _paragraph_claim_evidence_verified(
        self,
        selected_claim: str,
        job_id: str,
        paragraph_citations: list[Any],
    ) -> _AllowedCandidateClaim:
        entry = self._claim_catalog_entry(selected_claim, job_id)
        if entry is None:
            raise StateInvariantError(
                "Cover letter pre-execution validation rejection: "
                "exact_claim_evidence_missing"
            )
        citation_dicts = [
            citation for citation in paragraph_citations if isinstance(citation, dict)
        ]
        paragraph_keys = {
            self._cover_citation_locator_key(citation) for citation in citation_dicts
        }
        candidate_keys = {
            key
            for key in paragraph_keys
            if key[0]
            not in {
                "job_posting",
                "company_details",
                "fit_analysis",
                "finalized_resume",
            }
        }
        for citation in entry.citations:
            key = self._cover_citation_locator_key(citation)
            if key not in paragraph_keys:
                raise StateInvariantError(
                    "Cover letter pre-execution validation rejection: "
                    "exact_claim_citation_missing"
                )
            if key not in candidate_keys:
                raise StateInvariantError(
                    "Cover letter pre-execution validation rejection: "
                    "exact_claim_citation_missing"
                )
        return entry

    def _audit_paragraph_free_text_skills(
        self,
        free_text: str,
        job_id: str,
        *,
        field: str,
        citations: list[Any],
    ) -> _CoverLetterAuditIssue | None:
        if not free_text.strip():
            return None
        assert self.registry is not None
        job = self.registry._job(job_id)
        fit_analysis = self._build_resume_execution_fit_analysis(job_id)
        validated_citations = [
            CoverLetterCitation.model_validate(citation)
            for citation in citations
            if isinstance(citation, dict)
        ]
        try:
            _validate_required_skill_claims(
                free_text,
                validated_citations,
                job=job,
                fit_analysis=fit_analysis,
                bundle=self.registry.bundle,
                memory=self.registry.memory,
            )
        except CoverLetterEvidenceError as exc:
            message = str(exc)
            if (
                "claims required skill" in message
                or "presents genuine-gap capability" in message
            ):
                return _CoverLetterAuditIssue(
                    field=field,
                    category="evidence",
                    code="unsupported_skill_in_free_text",
                    message=f"{field} {message}",
                )
            raise StateInvariantError(
                f"Cover letter pre-execution validation rejection: {message}"
            ) from exc
        return None

    def _validate_cover_letter_exact_claim_spans(
        self,
        plan_dict: dict[str, Any],
        job_id: str,
        selected_claims: list[str | None],
    ) -> dict[str, Any]:
        assert self.registry is not None
        paragraphs = plan_dict.get("body_paragraphs") or []
        if len(selected_claims) < len(paragraphs):
            selected_claims = [
                *selected_claims,
                *([None] * (len(paragraphs) - len(selected_claims))),
            ]
        reconciled = self._reconcile_tailoring_evidence(job_id)
        skill_citations = [
            citation
            for skill in plan_dict.get("skills") or []
            if isinstance(skill, dict)
            for citation in skill.get("citations") or []
            if isinstance(citation, dict)
        ]
        hook_citations = [
            {
                "source_type": "company_details",
                "source_id": job_id,
                "source_field": str(
                    plan_dict.get("company_hook_source_field", "company_details")
                ),
                "evidence_id": None,
            }
        ]
        meta: dict[str, Any] = {
            "cover_letter_exact_claim_span_verified": False,
            "cover_letter_exact_claim_occurrence_count": 0,
            "cover_letter_exact_claim_evidence_verified": False,
            "cover_letter_exact_claim_citation_verified": False,
            "cover_letter_free_text_skill_scan_applied": False,
            "cover_letter_validated_claim_excluded_from_free_text_scan": False,
            "cover_letter_unsupported_skill_outside_claim_detected": False,
            "cover_letter_unsupported_skill_only_inside_validated_claim": False,
        }
        occurrence_total = 0
        for index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                continue
            field = f"body_paragraph_{index + 1}"
            text = str(paragraph.get("text", ""))
            selected_claim = selected_claims[index]
            if not isinstance(selected_claim, str) or not selected_claim:
                raise StateInvariantError(
                    "Cover letter pre-execution validation rejection: exact_claim_missing"
                )
            prefix, _, suffix = self._split_paragraph_exact_claim_span(
                text,
                selected_claim,
            )
            occurrence_total += 1
            self._paragraph_claim_evidence_verified(
                selected_claim,
                job_id,
                list(paragraph.get("citations") or []),
            )
            paragraph_citations = list(paragraph.get("citations") or [])
            meta["cover_letter_free_text_skill_scan_applied"] = True
            for segment_name, segment in (("lead_in", prefix), ("follow_up", suffix)):
                issue = self._audit_paragraph_free_text_skills(
                    segment,
                    job_id,
                    field=field,
                    citations=paragraph_citations,
                )
                if issue is not None:
                    meta["cover_letter_unsupported_skill_outside_claim_detected"] = True
                    raise StateInvariantError(
                        "Cover letter pre-execution validation rejection: "
                        f"{issue.message}"
                    )
            free_text = self._paragraph_free_text_for_skill_scan(text, selected_claim)
            full_scan_failed = False
            free_scan_failed = False
            try:
                _validate_required_skill_claims(
                    text,
                    [
                        CoverLetterCitation.model_validate(citation)
                        for citation in paragraph_citations
                        if isinstance(citation, dict)
                    ],
                    job=self.registry._job(job_id),
                    fit_analysis=self._build_resume_execution_fit_analysis(job_id),
                    bundle=self.registry.bundle,
                    memory=self.registry.memory,
                )
            except CoverLetterEvidenceError:
                full_scan_failed = True
            try:
                _validate_required_skill_claims(
                    free_text,
                    [
                        CoverLetterCitation.model_validate(citation)
                        for citation in paragraph_citations
                        if isinstance(citation, dict)
                    ],
                    job=self.registry._job(job_id),
                    fit_analysis=self._build_resume_execution_fit_analysis(job_id),
                    bundle=self.registry.bundle,
                    memory=self.registry.memory,
                )
            except CoverLetterEvidenceError:
                free_scan_failed = True
            if full_scan_failed and not free_scan_failed:
                meta["cover_letter_validated_claim_excluded_from_free_text_scan"] = True
                meta[
                    "cover_letter_unsupported_skill_only_inside_validated_claim"
                ] = True
            for segment_name, segment in (("lead_in", prefix), ("follow_up", suffix)):
                gap = self._text_claims_genuine_gap(segment, reconciled)
                if gap is not None:
                    raise StateInvariantError(
                        "Cover letter pre-execution validation rejection: "
                        f"{field}.{segment_name} claims genuine-gap skill {gap!r}"
                    )
        hook = str(plan_dict.get("company_hook_phrase", ""))
        closing = str(plan_dict.get("closing_sentence", ""))
        hook_issue = self._audit_paragraph_free_text_skills(
            hook,
            job_id,
            field="company_hook_phrase",
            citations=hook_citations + skill_citations,
        )
        if hook_issue is not None:
            meta["cover_letter_unsupported_skill_outside_claim_detected"] = True
            raise StateInvariantError(
                "Cover letter pre-execution validation rejection: "
                f"{hook_issue.message}"
            )
        closing_issue = self._audit_paragraph_free_text_skills(
            closing,
            job_id,
            field="closing_sentence",
            citations=skill_citations,
        )
        if closing_issue is not None:
            meta["cover_letter_unsupported_skill_outside_claim_detected"] = True
            raise StateInvariantError(
                "Cover letter pre-execution validation rejection: "
                f"{closing_issue.message}"
            )
        meta["cover_letter_exact_claim_span_verified"] = occurrence_total > 0
        meta["cover_letter_exact_claim_occurrence_count"] = occurrence_total
        meta["cover_letter_exact_claim_evidence_verified"] = occurrence_total > 0
        meta["cover_letter_exact_claim_citation_verified"] = occurrence_total > 0
        return meta

    @contextmanager
    def _claim_aware_required_skill_validation(
        self,
        selected_claims_by_text: dict[str, str],
    ) -> Iterator[None]:
        import src.tools.cover_letter as cover_letter_module

        original = cover_letter_module._validate_required_skill_claims

        def patched(
            text: str,
            citations: list[Any],
            *,
            job: Any,
            fit_analysis: Any,
            bundle: Any,
            memory: Any,
        ) -> None:
            selected_claim = selected_claims_by_text.get(text)
            scan_text = (
                self._paragraph_free_text_for_skill_scan(text, selected_claim)
                if selected_claim
                else text
            )
            return original(
                scan_text,
                citations,
                job=job,
                fit_analysis=fit_analysis,
                bundle=bundle,
                memory=memory,
            )

        cover_letter_module._validate_required_skill_claims = patched
        try:
            yield
        finally:
            cover_letter_module._validate_required_skill_claims = original

    def _prevalidate_hydrated_cover_letter_plan(
        self,
        plan_dict: dict[str, Any],
        job_id: str,
        *,
        transport_args: dict[str, Any] | None = None,
        selected_claims: list[str | None] | None = None,
    ) -> None:
        assert self.registry is not None
        assert self.state is not None
        job = self.registry._job(job_id)
        job_score = next(
            score for score in self.state.scoring_result.top_3 if score.job_id == job_id
        )
        fit_analysis = self._build_resume_execution_fit_analysis(job_id)
        finalized = self.state.finalized_resumes[job_id]
        resolved_claims = list(selected_claims or [])
        if not resolved_claims and transport_args is not None:
            resolved_claims = self._selected_claims_from_transport(transport_args)
        preexecution_meta = self._validate_cover_letter_exact_claim_spans(
            plan_dict,
            job_id,
            resolved_claims,
        )
        plan = CoverLetterPlan.model_validate(plan_dict)
        try:
            validate_cover_letter_plan(
                plan,
                job=job,
                job_score=job_score,
                fit_analysis=fit_analysis,
                bundle=self.registry.bundle,
                memory=self.registry.memory,
                finalized_resume=finalized,
            )
        except (
            CoverLetterEvidenceError,
            ToolRegistryError,
            ValidationError,
        ) as exc:
            raise StateInvariantError(
                f"Cover letter pre-execution validation rejection: {exc}"
            ) from exc
        self._cover_letter_preexecution_meta = preexecution_meta

    def _selected_claims_by_paragraph_text(
        self,
        plan_dict: dict[str, Any],
        transport_args: dict[str, Any] | None,
    ) -> dict[str, str]:
        claims = self._selected_claims_from_transport(transport_args or {})
        mapping: dict[str, str] = {}
        for index, paragraph in enumerate(plan_dict.get("body_paragraphs") or []):
            if not isinstance(paragraph, dict):
                continue
            selected = claims[index] if index < len(claims) else None
            if isinstance(selected, str) and selected:
                mapping[str(paragraph.get("text", ""))] = selected
        return mapping

    def _infer_cover_paragraph_field_from_error(self, error: str) -> str | None:
        lowered = error.casefold()
        for slot in ("body_paragraph_2", "body_paragraph_1"):
            if slot.replace("_", " ") in lowered or slot in lowered:
                return slot
        if "body paragraph" in lowered or "body text" in lowered:
            return "body_paragraph_1"
        return None

    @staticmethod
    def _concise_company_details(job: Any) -> str:
        words = job.company_details.split()
        excerpt = " ".join(words[:24]).rstrip(".,;:")
        return excerpt

    @staticmethod
    def _default_company_hook_phrase(job: Any) -> str:
        return " ".join(job.company_details.split()[:12]).rstrip(".,;:")

    @staticmethod
    def _word_spans(text: str) -> list[tuple[str, int, int]]:
        return [
            (match.group(0), match.start(), match.end())
            for match in _COVER_WORD_PATTERN.finditer(text)
        ]

    @staticmethod
    def _phrase_from_word_span(
        text: str,
        spans: list[tuple[str, int, int]],
        start_index: int,
        end_index: int,
    ) -> str:
        return text[spans[start_index][1] : spans[end_index][2]].strip()

    @staticmethod
    def _hook_meaningful_word_count(phrase: str) -> int:
        words = _COVER_WORD_PATTERN.findall(phrase)
        return sum(
            1 for word in words if word.casefold() not in _MEANINGLESS_HOOK_WORDS
        )

    @classmethod
    def _hook_is_generic_only(cls, phrase: str, company: str) -> bool:
        words = _COVER_WORD_PATTERN.findall(phrase)
        meaningful = [
            word
            for word in words
            if word.casefold() not in _MEANINGLESS_HOOK_WORDS
        ]
        if len(meaningful) < 4:
            return True
        company_tokens = {
            token.casefold()
            for token in _COVER_WORD_PATTERN.findall(company)
            if len(token) > 2
        }
        substantive = [
            word
            for word in meaningful
            if word.casefold() not in company_tokens
            and word.casefold() not in _COVER_HOOK_GENERIC_FILLER
        ]
        return len(substantive) < 2

    @classmethod
    def _is_valid_company_hook_candidate(
        cls,
        phrase: str,
        company_details: str,
        company: str,
    ) -> bool:
        candidate = phrase.strip()
        if not candidate or candidate != phrase.strip():
            return False
        if candidate not in company_details:
            return False
        normalized = _normalize_phrase(candidate)
        if normalized not in _normalize_phrase(company_details):
            return False
        words = _COVER_WORD_PATTERN.findall(candidate)
        if not 4 <= len(words) <= 15:
            return False
        if cls._hook_meaningful_word_count(candidate) < 4:
            return False
        if cls._hook_is_generic_only(candidate, company):
            return False
        return True

    @classmethod
    def _rank_company_hook_candidate(cls, phrase: str, company: str) -> tuple[int, int, int, int]:
        words = [word.casefold() for word in _COVER_WORD_PATTERN.findall(phrase)]
        company_tokens = {
            token.casefold()
            for token in _COVER_WORD_PATTERN.findall(company)
            if len(token) > 2
        }
        has_company_subject = int(
            any(token in phrase.casefold() for token in company_tokens if token)
        )
        mission_hits = sum(1 for word in words if word in _COVER_HOOK_MISSION_TERMS)
        impact_hits = sum(1 for word in words if word in _COVER_HOOK_IMPACT_TERMS)
        filler_hits = sum(1 for word in words if word in _MEANINGLESS_HOOK_WORDS)
        newline_penalty = int("\n" in phrase)
        return (
            has_company_subject,
            mission_hits + impact_hits,
            -filler_hits,
            -newline_penalty,
        )

    @staticmethod
    def _sentence_hook_fragments(text: str) -> list[tuple[str, int]]:
        fragments: list[tuple[str, int]] = []
        for match in re.finditer(r"[^.!?]+(?:[.!?]|$)", text):
            fragment = match.group(0).strip()
            if fragment:
                fragments.append((fragment, match.start()))
        if not fragments and text.strip():
            fragments.append((text.strip(), 0))
        return fragments

    @classmethod
    def _clause_hook_fragments(
        cls,
        fragment: str,
        base_position: int,
    ) -> list[tuple[str, int]]:
        fragments: list[tuple[str, int]] = [(fragment, base_position)]
        delimiter = re.compile(r"\s*;\s*|\s*:\s*|\s+—\s+|\s+–\s+")
        search_start = 0
        for part in delimiter.split(fragment):
            part = part.strip()
            if not part:
                continue
            position = base_position + fragment.find(part, search_start)
            if position >= base_position:
                fragments.append((part, position))
            search_start = max(search_start, fragment.find(part, search_start) + len(part))
        return fragments

    def _extract_allowed_company_hooks(self, job: Any) -> tuple[list[str], bool]:
        details = job.company_details
        ranked: dict[str, tuple[tuple[int, int, int], int, str]] = {}
        fallback_used = False

        def consider(phrase: str, position: int) -> None:
            if not self._is_valid_company_hook_candidate(
                phrase,
                details,
                job.company,
            ):
                return
            rank = self._rank_company_hook_candidate(phrase, job.company)
            normalized = _normalize_phrase(phrase)
            existing = ranked.get(normalized)
            if existing is None or (rank, -position) > (existing[0], -existing[1]):
                ranked[normalized] = (rank, position, phrase)

        for sentence, sentence_pos in self._sentence_hook_fragments(details):
            consider(sentence, sentence_pos)
            for clause, clause_pos in self._clause_hook_fragments(sentence, sentence_pos):
                consider(clause, clause_pos)
            for line in sentence.splitlines():
                line = line.strip()
                if line:
                    line_pos = sentence_pos + sentence.find(line)
                    consider(line, line_pos)
                    for clause, clause_pos in self._clause_hook_fragments(line, line_pos):
                        consider(clause, clause_pos)

        spans = self._word_spans(details)
        for start in range(len(spans)):
            for length in range(4, 16):
                end = start + length - 1
                if end >= len(spans):
                    break
                consider(
                    self._phrase_from_word_span(details, spans, start, end),
                    spans[start][1],
                )

        ordered = sorted(
            ranked.values(),
            key=lambda item: (
                -item[0][0],
                -item[0][1],
                -item[0][2],
                -item[0][3],
                item[1],
                item[2],
            ),
        )
        hooks = [item[2] for item in ordered[:_COVER_HOOK_MAX_OPTIONS]]
        if hooks:
            return hooks, fallback_used

        if len(spans) >= 4:
            fallback_length = min(max(8, 4), len(spans), 15)
            fallback = self._phrase_from_word_span(
                details,
                spans,
                0,
                fallback_length - 1,
            )
            if self._is_valid_company_hook_candidate(
                fallback,
                details,
                job.company,
            ):
                return [fallback], True
        return [], False

    def _select_allowed_company_hooks(self, job_id: str) -> list[str]:
        if job_id in self._cover_letter_hook_cache:
            return list(self._cover_letter_hook_cache[job_id][0])
        assert self.registry is not None
        job = self.registry._job(job_id)
        hooks, fallback_used = self._extract_allowed_company_hooks(job)
        self._cover_letter_hook_cache[job_id] = (list(hooks), fallback_used)
        return list(hooks)

    def _ensure_allowed_company_hooks_for_job(self, job_id: str) -> tuple[list[str], bool]:
        hooks, fallback_used = self._cover_letter_hook_cache.get(job_id, ([], False))
        if not hooks:
            assert self.registry is not None
            job = self.registry._job(job_id)
            hooks, fallback_used = self._extract_allowed_company_hooks(job)
            self._cover_letter_hook_cache[job_id] = (list(hooks), fallback_used)
        if not hooks:
            raise StateInvariantError(
                "Cover letter hydration rejection: no allowed company hooks "
                "could be extracted from company_details"
            )
        if not _COVER_HOOK_MIN_OPTIONS <= len(hooks) <= _COVER_HOOK_MAX_OPTIONS:
            raise StateInvariantError(
                "Cover letter hydration rejection: allowed company hook count "
                f"out of bounds: {len(hooks)}"
            )
        return list(hooks), fallback_used

    @staticmethod
    def _apply_cover_hook_enum_to_schema(
        schema: dict[str, Any],
        allowed_hooks: list[str],
    ) -> dict[str, Any]:
        properties = schema.get("properties", {})
        hook_property = properties.get("company_hook_phrase")
        if isinstance(hook_property, dict) and allowed_hooks:
            hook_property["enum"] = list(allowed_hooks)
        return schema

    @staticmethod
    def _apply_cover_claim_enum_to_schema(
        schema: dict[str, Any],
        allowed_claims: list[str],
    ) -> dict[str, Any]:
        if not allowed_claims:
            return schema
        for prop in ("body_paragraph_1", "body_paragraph_2"):
            paragraph = schema.get("properties", {}).get(prop)
            if isinstance(paragraph, dict):
                claim_property = paragraph.get("properties", {}).get(
                    "selected_candidate_claim"
                )
                if isinstance(claim_property, dict):
                    claim_property["enum"] = list(allowed_claims)
        wrapper_def = schema.get("$defs", {}).get("_CoverLetterParagraphWrapper")
        if isinstance(wrapper_def, dict):
            claim_property = wrapper_def.get("properties", {}).get(
                "selected_candidate_claim"
            )
            if isinstance(claim_property, dict):
                claim_property["enum"] = list(allowed_claims)
        return schema

    def _cover_letter_paragraph_wrapper_model(
        self,
        allowed_claims: tuple[str, ...],
    ) -> type[BaseModel]:
        cache_key = ("paragraph_wrapper", allowed_claims)
        cached = self._cover_letter_schema_model_cache.get(cache_key)
        if cached is not None:
            return cached
        claim_field = (
            str,
            Field(
                min_length=1,
                json_schema_extra=(
                    {"enum": list(allowed_claims)} if allowed_claims else {}
                ),
            ),
        )
        model = create_model(
            "_CoverLetterParagraphWrapper",
            __config__=ConfigDict(extra="forbid"),
            lead_in=(str, Field(min_length=1)),
            selected_candidate_claim=claim_field,
            follow_up=(str, Field(min_length=1)),
            reason=(str, Field(min_length=1)),
        )
        self._cover_letter_schema_model_cache[cache_key] = model
        return model

    def _allowed_claims_for_contract(self, contract: dict[str, Any]) -> list[str]:
        allowed = list(contract.get("allowed_candidate_claims") or [])
        target_job_id = contract.get("target_job_id")
        if not allowed and target_job_id:
            allowed = self._allowed_candidate_claim_texts(target_job_id)
        return allowed

    def _cover_letter_schema_model_for_contract(
        self,
        contract: dict[str, Any],
    ) -> type[BaseModel]:
        allowed_hooks = tuple(contract.get("allowed_company_hooks") or ())
        patch_fields = tuple(contract.get("cover_patch_fields") or ())
        allowed_claims = tuple(self._allowed_claims_for_contract(contract))
        cache_key = (allowed_hooks, patch_fields, allowed_claims)
        cached = self._cover_letter_schema_model_cache.get(cache_key)
        if cached is not None:
            return cached

        wrapper_model = self._cover_letter_paragraph_wrapper_model(allowed_claims)

        if patch_fields:
            field_defs: dict[str, Any] = {"job_id": (str, Field(min_length=1))}
            if "company_hook_phrase" in patch_fields:
                field_defs["company_hook_phrase"] = (
                    str,
                    Field(
                        min_length=1,
                        json_schema_extra={"enum": list(allowed_hooks)},
                    ),
                )
            if "body_paragraph_1" in patch_fields:
                field_defs["body_paragraph_1"] = (wrapper_model, ...)
            if "body_paragraph_2" in patch_fields:
                field_defs["body_paragraph_2"] = (wrapper_model | None, ...)
            if "skills" in patch_fields:
                field_defs["skills"] = (list[str], Field(min_length=3, max_length=8))
            if "closing_sentence" in patch_fields:
                field_defs["closing_sentence"] = (str, Field(min_length=1))
            if "plan_rationale" in patch_fields:
                field_defs["plan_rationale"] = (str, Field(min_length=1, max_length=200))

            model = create_model(
                "_CoverLetterPatchDraft",
                __config__=ConfigDict(extra="forbid"),
                **field_defs,
            )
            self._cover_letter_schema_model_cache[cache_key] = model
            return model

        model = create_model(
            "_CoverLetterTransportDraftEnum",
            __config__=ConfigDict(extra="forbid"),
            decision_summary=(str, Field(min_length=1)),
            job_id=(str, Field(min_length=1)),
            company_hook_phrase=(
                str,
                Field(
                    min_length=1,
                    json_schema_extra={"enum": list(allowed_hooks)},
                ),
            ),
            body_paragraph_1=(wrapper_model, ...),
            body_paragraph_2=(wrapper_model | None, None),
            skills=(list[str], Field(min_length=1)),
            closing_sentence=(str, Field(min_length=1)),
            plan_rationale=(str, Field(min_length=1)),
        )
        self._cover_letter_schema_model_cache[cache_key] = model
        return model

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
            validated_citation = self._resolve_validated_skill_citation(cleaned)
            if validated_citation is None:
                continue
            registry[canonical] = _AllowedCoverSkill(
                display_name=cleaned,
                canonical=canonical,
                citation=validated_citation,
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
        memory_facts = [
            {
                "fact_type": fact.fact_type,
                "statement": fact.statement,
                "normalized_value": fact.normalized_value,
                "skill_tags": fact.skill_tags,
            }
            for fact in self.registry.memory.facts
        ]
        allowed_hooks, fallback_used = self._ensure_allowed_company_hooks_for_job(job_id)
        allowed_claims = self._allowed_candidate_claim_texts(job_id)
        return {
            "target_job_id": job_id,
            "rank": self._target_rank(job_id),
            "title": job.title,
            "company": job.company,
            "allowed_company_hooks": allowed_hooks,
            "allowed_candidate_claim_count": len(allowed_claims),
            "approved_resume_revision": finalized.approved_revision_round,
            "finalized_resume_summary": (
                f"Approved revision {finalized.approved_revision_round} resume "
                f"for {job.title} at {job.company}."
            ),
            "allowed_skills": self._select_model_visible_skills(job_id),
            "do_not_claim_skills": reconciled.genuine_gaps,
            "current_memory_facts": memory_facts,
            "paragraph_count_requirement": "1 or 2",
            "one_page_required": True,
        }

    @staticmethod
    def _cover_draft_required_shape(
        target_job_id: str,
        allowed_hooks: list[str] | None = None,
    ) -> dict[str, Any]:
        hook_shape = (
            allowed_hooks[0]
            if allowed_hooks
            else "<exact allowed_company_hooks option>"
        )
        return {
            "decision_summary": "<concise explanation>",
            "job_id": target_job_id,
            "company_hook_phrase": hook_shape,
            "body_paragraph_1": {
                "lead_in": "<bounded introduction>",
                "selected_candidate_claim": "<exact enum value>",
                "follow_up": "<bounded conclusion>",
                "reason": "<reason>",
            },
            "body_paragraph_2": None,
            "skills": ["<3 to 8 allowed skills>"],
            "closing_sentence": "<concise closing>",
            "plan_rationale": "<concise rationale>",
        }

    @staticmethod
    def _validate_cover_hook_enum(
        hook_phrase: str,
        allowed_hooks: list[str],
    ) -> None:
        if hook_phrase not in set(allowed_hooks):
            raise ToolArgumentsError(
                "Cover letter transport rejection: company_hook_phrase must be "
                "one of allowed_company_hooks"
            )

    @staticmethod
    def _parse_cover_letter_transport_structure(
        arguments: dict[str, Any],
    ) -> _CoverLetterTransportDraft:
        try:
            return _CoverLetterTransportDraft.model_validate(arguments)
        except ValidationError as exc:
            raise ToolArgumentsError(
                f"Cover letter transport rejection: {exc}"
            ) from exc

    @staticmethod
    def _assemble_paragraph_text(lead_in: str, claim: str, follow_up: str) -> str:
        lead = lead_in.strip()
        follow = follow_up.strip()
        if lead.endswith((".", "!", "?", ",", ";", ":")):
            separator_before_claim = " "
        else:
            separator_before_claim = " "
        if follow and claim.endswith((".", "!", "?", ",", ";", ":")):
            separator_before_follow = " "
        else:
            separator_before_follow = " "
        return f"{lead}{separator_before_claim}{claim}{separator_before_follow}{follow}".strip()

    @staticmethod
    def _segment_restates_allowed_claim(segment: str, claim: str) -> bool:
        if claim in segment:
            return True
        claim_words = claim.casefold().split()
        segment_norm = segment.casefold()
        for window in range(min(6, len(claim_words)), len(claim_words) + 1):
            for index in range(len(claim_words) - window + 1):
                phrase = " ".join(claim_words[index : index + window])
                if len(phrase) >= 24 and phrase in segment_norm:
                    return True
        return False

    def _audit_paragraph_wrapper_nonrepairable(
        self,
        paragraph: dict[str, Any],
        field: str,
        allowed_claims: list[str],
    ) -> list[_CoverLetterAuditIssue]:
        issues: list[_CoverLetterAuditIssue] = []
        lead_in = str(paragraph.get("lead_in", ""))
        follow_up = str(paragraph.get("follow_up", ""))
        selected = str(paragraph.get("selected_candidate_claim", ""))

        for segment_name, segment in (("lead_in", lead_in), ("follow_up", follow_up)):
            if re.search(r"\d", segment):
                issues.append(
                    _CoverLetterAuditIssue(
                        field=field,
                        category="semantic_text",
                        code="wrapper_digits",
                        message=f"{field}.{segment_name} must not contain digits",
                    )
                )
            if "%" in segment:
                issues.append(
                    _CoverLetterAuditIssue(
                        field=field,
                        category="semantic_text",
                        code="wrapper_percent",
                        message=f"{field}.{segment_name} must not contain percent signs",
                    )
                )
            for marker in _COVER_WRAPPER_BULLET_MARKERS:
                if segment.lstrip().startswith(marker.strip()):
                    issues.append(
                        _CoverLetterAuditIssue(
                            field=field,
                            category="semantic_text",
                            code="wrapper_bullet_marker",
                            message=(
                                f"{field}.{segment_name} must not contain bullet "
                                "markers or headings"
                            ),
                        )
                    )
            for claim in allowed_claims:
                if claim and self._segment_restates_allowed_claim(segment, claim):
                    issues.append(
                        _CoverLetterAuditIssue(
                            field=field,
                            category="semantic_text",
                            code="wrapper_claim_paraphrase",
                            message=(
                                f"{field}.{segment_name} must not repeat or restate "
                                "an allowed candidate claim"
                            ),
                        )
                    )
                    break
            for claim in allowed_claims:
                for number in _COVER_NUMBER_PATTERN.findall(claim):
                    if number in segment:
                        issues.append(
                            _CoverLetterAuditIssue(
                                field=field,
                                category="semantic_text",
                                code="wrapper_metric_restatement",
                                message=(
                                    f"{field}.{segment_name} must not restate "
                                    "candidate metrics"
                                ),
                            )
                        )

        if selected and selected not in set(allowed_claims):
            issues.append(
                _CoverLetterAuditIssue(
                    field=field,
                    category="draft_schema",
                    code="invalid_selected_claim",
                    message=(
                        f"{field}.selected_candidate_claim must be one of the "
                        "allowed claim enum values"
                    ),
                )
            )
        return issues

    def _audit_paragraph_wrapper_length(
        self,
        paragraph: dict[str, Any],
        field: str,
        *,
        assembled_word_count: int | None = None,
    ) -> list[_CoverLetterAuditIssue]:
        issues: list[_CoverLetterAuditIssue] = []
        lead_in = str(paragraph.get("lead_in", ""))
        follow_up = str(paragraph.get("follow_up", ""))
        reason = str(paragraph.get("reason", ""))

        lead_words = _word_count(lead_in)
        if lead_words < _COVER_LEAD_IN_MIN_WORDS:
            issues.append(
                _CoverLetterAuditIssue(
                    field=field,
                    category="semantic_text",
                    code="lead_in_too_short",
                    message=(
                        f"{field}.lead_in must contain {_COVER_LEAD_IN_MIN_WORDS} to "
                        f"{_COVER_LEAD_IN_MAX_WORDS} words; found {lead_words}"
                    ),
                )
            )
        elif lead_words > _COVER_LEAD_IN_MAX_WORDS:
            issues.append(
                _CoverLetterAuditIssue(
                    field=field,
                    category="semantic_text",
                    code="lead_in_too_long",
                    message=(
                        f"{field}.lead_in must contain {_COVER_LEAD_IN_MIN_WORDS} to "
                        f"{_COVER_LEAD_IN_MAX_WORDS} words; found {lead_words}"
                    ),
                )
            )

        follow_words = _word_count(follow_up)
        if follow_words < _COVER_FOLLOW_UP_MIN_WORDS:
            issues.append(
                _CoverLetterAuditIssue(
                    field=field,
                    category="semantic_text",
                    code="follow_up_too_short",
                    message=(
                        f"{field}.follow_up must contain {_COVER_FOLLOW_UP_MIN_WORDS} "
                        f"to {_COVER_FOLLOW_UP_MAX_WORDS} words; found {follow_words}"
                    ),
                )
            )
        elif follow_words > _COVER_FOLLOW_UP_MAX_WORDS:
            issues.append(
                _CoverLetterAuditIssue(
                    field=field,
                    category="semantic_text",
                    code="follow_up_too_long",
                    message=(
                        f"{field}.follow_up must contain {_COVER_FOLLOW_UP_MIN_WORDS} "
                        f"to {_COVER_FOLLOW_UP_MAX_WORDS} words; found {follow_words}"
                    ),
                )
            )

        reason_words = _word_count(reason)
        if reason_words > _COVER_REASON_MAX_WORDS:
            issues.append(
                _CoverLetterAuditIssue(
                    field=field,
                    category="semantic_text",
                    code="reason_too_long",
                    message=(
                        f"{field} reason must be at most {_COVER_REASON_MAX_WORDS} "
                        f"words; found {reason_words}"
                    ),
                )
            )

        if assembled_word_count is not None:
            if assembled_word_count < _COVER_ASSEMBLED_PARAGRAPH_MIN_WORDS:
                issues.append(
                    _CoverLetterAuditIssue(
                        field=field,
                        category="semantic_text",
                        code="assembled_paragraph_too_short",
                        message=(
                            f"{field} must contain {_COVER_ASSEMBLED_PARAGRAPH_MIN_WORDS} "
                            f"to {_COVER_ASSEMBLED_PARAGRAPH_MAX_WORDS} words; "
                            f"found {assembled_word_count}"
                        ),
                    )
                )
            elif assembled_word_count > _COVER_ASSEMBLED_PARAGRAPH_MAX_WORDS:
                issues.append(
                    _CoverLetterAuditIssue(
                        field=field,
                        category="semantic_text",
                        code="assembled_paragraph_too_long",
                        message=(
                            f"{field} must contain {_COVER_ASSEMBLED_PARAGRAPH_MIN_WORDS} "
                            f"to {_COVER_ASSEMBLED_PARAGRAPH_MAX_WORDS} words; "
                            f"found {assembled_word_count}"
                        ),
                    )
                )
        return issues

    def _audit_paragraph_wrapper_segments(
        self,
        paragraph: dict[str, Any],
        field: str,
        job_id: str,
        allowed_claims: list[str],
    ) -> list[_CoverLetterAuditIssue]:
        del job_id
        nonrepairable = self._audit_paragraph_wrapper_nonrepairable(
            paragraph,
            field,
            allowed_claims,
        )
        if nonrepairable:
            return nonrepairable
        claim = str(paragraph.get("selected_candidate_claim", ""))
        assembled = self._assemble_paragraph_text(
            str(paragraph.get("lead_in", "")),
            claim,
            str(paragraph.get("follow_up", "")),
        )
        return self._audit_paragraph_wrapper_length(
            paragraph,
            field,
            assembled_word_count=_word_count(assembled),
        )

    @classmethod
    def _select_wrapper_tiers_for_claim(
        cls,
        claim: str,
    ) -> tuple[int, int]:
        claim_words = _word_count(claim)
        if claim_words > _COVER_ASSEMBLED_PARAGRAPH_MAX_WORDS:
            raise ToolArgumentsError(
                "Cover letter wrapper repair rejection: selected claim exceeds "
                "assembled paragraph maximum"
            )
        best: tuple[int, int, int] | None = None
        for lead_idx, lead in enumerate(_COVER_WRAPPER_LEAD_TIERS):
            lead_words = _word_count(lead)
            if not _COVER_LEAD_IN_MIN_WORDS <= lead_words <= _COVER_LEAD_IN_MAX_WORDS:
                continue
            for follow_idx, follow in enumerate(_COVER_WRAPPER_FOLLOW_TIERS):
                follow_words = _word_count(follow)
                if not _COVER_FOLLOW_UP_MIN_WORDS <= follow_words <= _COVER_FOLLOW_UP_MAX_WORDS:
                    continue
                assembled_words = _word_count(
                    cls._assemble_paragraph_text(lead, claim, follow)
                )
                if not (
                    _COVER_ASSEMBLED_PARAGRAPH_MIN_WORDS
                    <= assembled_words
                    <= _COVER_ASSEMBLED_PARAGRAPH_MAX_WORDS
                ):
                    continue
                tier_sum = lead_idx + follow_idx
                if best is None or tier_sum < best[0] or (
                    tier_sum == best[0] and lead_idx < best[1]
                ):
                    best = (tier_sum, lead_idx, follow_idx)
        if best is None:
            raise ToolArgumentsError(
                "Cover letter wrapper repair rejection: no safe wrapper tier produces "
                "a legal assembled paragraph"
            )
        return best[1], best[2]

    @staticmethod
    def _lead_tier_for_segment_repair(code: str) -> int:
        if code == "lead_in_too_long":
            return 0
        for index, template in enumerate(_COVER_WRAPPER_LEAD_TIERS):
            if _word_count(template) >= _COVER_LEAD_IN_MIN_WORDS:
                return index
        return len(_COVER_WRAPPER_LEAD_TIERS) - 1

    @staticmethod
    def _follow_tier_for_segment_repair(code: str) -> int:
        if code == "follow_up_too_long":
            return 0
        for index, template in enumerate(_COVER_WRAPPER_FOLLOW_TIERS):
            if _word_count(template) >= _COVER_FOLLOW_UP_MIN_WORDS:
                return index
        return len(_COVER_WRAPPER_FOLLOW_TIERS) - 1

    def _repair_paragraph_wrapper_if_needed(
        self,
        paragraph: dict[str, Any],
        field: str,
        allowed_claims: list[str],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        nonrepairable = self._audit_paragraph_wrapper_nonrepairable(
            paragraph,
            field,
            allowed_claims,
        )
        if nonrepairable:
            _CoverLetterAuditResult(issues=nonrepairable).raise_if_issues()

        repaired = copy.deepcopy(paragraph)
        claim = str(repaired["selected_candidate_claim"])
        original_lead = str(repaired.get("lead_in", ""))
        original_follow = str(repaired.get("follow_up", ""))
        original_reason = str(repaired.get("reason", ""))
        meta: dict[str, Any] = {
            "cover_letter_wrapper_length_repair_applied": False,
            "cover_letter_lead_in_repaired": False,
            "cover_letter_follow_up_repaired": False,
            "cover_letter_reason_repaired": False,
            "cover_letter_wrapper_template_tier": None,
            "cover_letter_paragraph_word_count_after_repair": None,
            "cover_letter_same_turn_repair": False,
        }

        assembled = self._assemble_paragraph_text(
            str(repaired.get("lead_in", "")),
            claim,
            str(repaired.get("follow_up", "")),
        )
        length_issues = self._audit_paragraph_wrapper_length(
            repaired,
            field,
            assembled_word_count=_word_count(assembled),
        )
        if not length_issues:
            meta["cover_letter_paragraph_word_count_after_repair"] = _word_count(
                assembled
            )
            return repaired, meta

        meta["cover_letter_wrapper_length_repair_applied"] = True
        meta["cover_letter_same_turn_repair"] = True
        issue_codes = {issue.code for issue in length_issues if issue.code}

        if "reason_too_long" in issue_codes:
            repaired["reason"] = _COVER_NORMALIZED_REASON
            meta["cover_letter_reason_repaired"] = True

        if "lead_in_too_short" in issue_codes:
            lead_idx = self._lead_tier_for_segment_repair("lead_in_too_short")
            repaired["lead_in"] = _COVER_WRAPPER_LEAD_TIERS[lead_idx]
            meta["cover_letter_lead_in_repaired"] = True
            meta["cover_letter_wrapper_template_tier"] = f"lead:{lead_idx}"
        elif "lead_in_too_long" in issue_codes:
            lead_idx = self._lead_tier_for_segment_repair("lead_in_too_long")
            repaired["lead_in"] = _COVER_WRAPPER_LEAD_TIERS[lead_idx]
            meta["cover_letter_lead_in_repaired"] = True
            meta["cover_letter_wrapper_template_tier"] = f"lead:{lead_idx}"

        if "follow_up_too_short" in issue_codes:
            follow_idx = self._follow_tier_for_segment_repair("follow_up_too_short")
            repaired["follow_up"] = _COVER_WRAPPER_FOLLOW_TIERS[follow_idx]
            meta["cover_letter_follow_up_repaired"] = True
            tier = meta["cover_letter_wrapper_template_tier"]
            meta["cover_letter_wrapper_template_tier"] = (
                f"{tier},follow:{follow_idx}" if tier else f"follow:{follow_idx}"
            )
        elif "follow_up_too_long" in issue_codes:
            follow_idx = self._follow_tier_for_segment_repair("follow_up_too_long")
            repaired["follow_up"] = _COVER_WRAPPER_FOLLOW_TIERS[follow_idx]
            meta["cover_letter_follow_up_repaired"] = True
            tier = meta["cover_letter_wrapper_template_tier"]
            meta["cover_letter_wrapper_template_tier"] = (
                f"{tier},follow:{follow_idx}" if tier else f"follow:{follow_idx}"
            )

        assembled = self._assemble_paragraph_text(
            str(repaired.get("lead_in", "")),
            claim,
            str(repaired.get("follow_up", "")),
        )
        assembled_words = _word_count(assembled)
        if (
            assembled_words < _COVER_ASSEMBLED_PARAGRAPH_MIN_WORDS
            or assembled_words > _COVER_ASSEMBLED_PARAGRAPH_MAX_WORDS
            or "assembled_paragraph_too_short" in issue_codes
            or "assembled_paragraph_too_long" in issue_codes
        ):
            lead_idx, follow_idx = self._select_wrapper_tiers_for_claim(claim)
            repaired["lead_in"] = _COVER_WRAPPER_LEAD_TIERS[lead_idx]
            repaired["follow_up"] = _COVER_WRAPPER_FOLLOW_TIERS[follow_idx]
            if repaired["lead_in"] != original_lead:
                meta["cover_letter_lead_in_repaired"] = True
            if repaired["follow_up"] != original_follow:
                meta["cover_letter_follow_up_repaired"] = True
            meta["cover_letter_wrapper_template_tier"] = (
                f"lead:{lead_idx},follow:{follow_idx}"
            )
            assembled = self._assemble_paragraph_text(
                repaired["lead_in"],
                claim,
                repaired["follow_up"],
            )
            assembled_words = _word_count(assembled)

        remaining = self._audit_paragraph_wrapper_length(
            repaired,
            field,
            assembled_word_count=assembled_words,
        )
        if remaining:
            _CoverLetterAuditResult(issues=remaining).raise_if_issues()

        if repaired["reason"] != original_reason:
            meta["cover_letter_reason_repaired"] = True
        meta["cover_letter_paragraph_word_count_after_repair"] = assembled_words
        return repaired, meta

    def _assemble_cover_letter_transport(
        self,
        wrapper_args: dict[str, Any],
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        result = copy.deepcopy(wrapper_args)
        target_job_id = contract.get("target_job_id")
        if target_job_id is None:
            raise StateInvariantError(
                "Cover letter assembly rejection: missing deterministic target job"
            )
        allowed_claims = self._allowed_claims_for_contract(contract)
        combined_repair_meta: dict[str, Any] = {}
        for field in ("body_paragraph_1", "body_paragraph_2"):
            paragraph = result.get(field)
            if not isinstance(paragraph, dict):
                continue
            if "text" in paragraph and "selected_candidate_claim" not in paragraph:
                continue
            repaired, repair_meta = self._repair_paragraph_wrapper_if_needed(
                paragraph,
                field,
                allowed_claims,
            )
            result[field] = repaired
            combined_repair_meta[field] = repair_meta
            claim = str(repaired["selected_candidate_claim"])
            assembled_text = self._assemble_paragraph_text(
                str(repaired["lead_in"]),
                claim,
                str(repaired["follow_up"]),
            )
            if claim not in assembled_text:
                raise ToolArgumentsError(
                    "Cover letter assembly rejection: selected claim is not an exact "
                    "substring of the assembled paragraph"
                )
            result[field] = {
                "text": assembled_text,
                "reason": str(repaired["reason"]),
                "selected_candidate_claim": claim,
            }
        self._cover_letter_wrapper_repair_meta = combined_repair_meta or None
        return result

    def _parse_cover_letter_wrapper_draft(
        self,
        arguments: dict[str, Any],
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        allowed_hooks = list(contract.get("allowed_company_hooks") or [])
        if not allowed_hooks:
            target_job_id = contract.get("target_job_id")
            if target_job_id:
                allowed_hooks = self._select_allowed_company_hooks(target_job_id)
        if "company_hook_phrase" in arguments and allowed_hooks:
            self._validate_cover_hook_enum(
                str(arguments["company_hook_phrase"]),
                allowed_hooks,
            )
        draft_model = self._cover_letter_schema_model_for_contract(contract)
        try:
            validated = draft_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolArgumentsError(
                f"Cover letter transport rejection: {exc}"
            ) from exc
        return validated.model_dump(mode="json")

    def _parse_cover_letter_transport(
        self,
        arguments: dict[str, Any],
        contract: dict[str, Any],
    ) -> dict[str, Any]:
        wrapper = self._parse_cover_letter_wrapper_draft(arguments, contract)
        return self._assemble_cover_letter_transport(wrapper, contract)

    @staticmethod
    def _parse_cover_letter_draft(
        arguments: dict[str, Any],
        contract: dict[str, Any] | None = None,
    ) -> _CoverLetterTransportDraft:
        if contract is None:
            return JobSearchAgentRuntime._parse_cover_letter_transport_structure(
                arguments
            )
        raise NotImplementedError("Use _parse_cover_letter_transport with contract")

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
            patch_model = self._cover_letter_schema_model_for_contract(contract)
            try:
                patch = patch_model.model_validate(arguments)
            except ValidationError as exc:
                raise ToolArgumentsError(
                    f"Cover letter transport rejection: {exc}"
                ) from exc
            if (
                "company_hook_phrase" in patch_fields
                and contract.get("allowed_company_hooks")
            ):
                self._validate_cover_hook_enum(
                    str(getattr(patch, "company_hook_phrase", "")),
                    list(contract["allowed_company_hooks"]),
                )
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
        full_contract = {**contract, "cover_patch_fields": []}
        wrapper = self._parse_cover_letter_wrapper_draft(arguments, full_contract)
        if wrapper.get("job_id") != target_job_id:
            raise StateInvariantError(
                "Cover letter draft job_id mismatch: expected "
                f"{target_job_id!r}; received {wrapper.get('job_id')!r}"
            )
        return self._assemble_cover_letter_transport(wrapper, full_contract)

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
        if not required.issubset(arguments):
            return False
        paragraph = arguments.get("body_paragraph_1")
        if not isinstance(paragraph, dict):
            return False
        wrapper_required = {
            "lead_in",
            "selected_candidate_claim",
            "follow_up",
            "reason",
        }
        return wrapper_required.issubset(paragraph)

    def _merge_cover_patch(
        self,
        base_draft: dict[str, Any],
        patch: dict[str, Any],
        patch_fields: list[str],
    ) -> dict[str, Any]:
        merged = copy.deepcopy(base_draft)
        for field in patch_fields:
            if field not in patch:
                continue
            if field.startswith("body_paragraph") and isinstance(
                merged.get(field), dict
            ) and isinstance(patch[field], dict):
                combined = copy.deepcopy(merged[field])
                combined.update(copy.deepcopy(patch[field]))
                merged[field] = combined
            else:
                merged[field] = copy.deepcopy(patch[field])
        return merged

    def _cover_patch_required_shape(
        self,
        target_job_id: str,
        patch_fields: list[str],
        *,
        allowed_hooks: list[str] | None = None,
    ) -> dict[str, Any]:
        shape: dict[str, Any] = {"job_id": target_job_id}
        if "company_hook_phrase" in patch_fields:
            shape["company_hook_phrase"] = (
                allowed_hooks[0]
                if allowed_hooks
                else "<exact allowed_company_hooks option>"
            )
        if "body_paragraph_1" in patch_fields:
            shape["body_paragraph_1"] = {
                "lead_in": "<bounded introduction>",
                "selected_candidate_claim": "<exact enum value>",
                "follow_up": "<bounded conclusion>",
                "reason": "<reason>",
            }
        if "body_paragraph_2" in patch_fields:
            shape["body_paragraph_2"] = {
                "lead_in": "<bounded introduction>",
                "selected_candidate_claim": "<exact enum value>",
                "follow_up": "<bounded conclusion>",
                "reason": "<reason>",
            }
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
        skills = raw_call.arguments.get("skills")
        if isinstance(skills, list) and not 3 <= len(skills) <= 8:
            patch_fields.append("skills")
        patch_fields = sorted(set(patch_fields))
        if contract.get("cover_recovery_mode") == "patch" and self._cover_letter_patch_recovery:
            if error and (
                "extra fields" in error.casefold()
                or "extra inputs are not permitted" in error.casefold()
            ):
                self._cover_letter_patch_recovery["rejected_category"] = "draft_schema"
                return
            new_fields = patch_fields or (
                self._cover_patch_fields_from_error(error or "", audit)
                if error
                else []
            )
            new_fields = [
                field for field in new_fields if field in _COVER_PATCHABLE_FIELDS
            ]
            if new_fields:
                merged_fields = sorted(
                    {
                        *self._cover_letter_patch_recovery.get("patch_fields", []),
                        *new_fields,
                    }
                )
                self._cover_letter_patch_recovery["patch_fields"] = merged_fields
                base_draft = self._cover_letter_patch_recovery["base_draft"]
                self._cover_letter_patch_recovery["preserved_fields"] = [
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
                    if field in base_draft and field not in merged_fields
                ]
            self._cover_letter_patch_recovery["rejected_category"] = rejected_category
            return
        if not self._is_complete_cover_draft(raw_call.arguments):
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

    def _cover_letter_duplicate_wrapper_failure(
        self,
        contract: dict[str, Any],
        audit: _CoverLetterAuditResult | None,
    ) -> bool:
        patch_fields = list(contract.get("cover_patch_fields") or [])
        if not any(field.startswith("body_paragraph") for field in patch_fields):
            return False
        if not isinstance(audit, _CoverLetterAuditResult) or not audit.issues:
            return False
        return any(
            issue.field.startswith("body_paragraph") for issue in audit.issues
        )

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
            and "required skill" not in lowered
            and "unsupported required skill" not in lowered
            and "claims required skill" not in lowered
        ) or "between 3 and 8" in lowered or "duplicate skill" in lowered:
            if (
                "unsupported skill" in lowered
                or "between 3 and 8" in lowered
                or "duplicate skill" in lowered
                or "at most 8" in lowered
                or ("too many" in lowered and "skills" in lowered)
                or (
                    "skills" in lowered
                    and "body" not in lowered
                    and "paragraph" not in lowered
                )
            ):
                fields.append("skills")
        paragraph_field = self._infer_cover_paragraph_field_from_error(error)
        if paragraph_field is not None:
            fields.append(paragraph_field)
        if (
            "claims required skill" in lowered
            or "without candidate evidence" in lowered
            or "unsupported numeric" in lowered
            or "numeric claim" in lowered
            or "allowed_candidate_claim" in lowered
            or "genuine-gap" in lowered
            or "introduces capability not supported" in lowered
            or "unsupported required skill" in lowered
        ) and paragraph_field is None:
            fields.append("body_paragraph_1")
        if "closing_sentence" in lowered:
            fields.append("closing_sentence")
        if "plan_rationale" in lowered:
            fields.append("plan_rationale")
        for field in _COVER_PATCHABLE_FIELDS:
            if f"{field}\n" in error or f'"{field}"' in error:
                fields.append(field)
        if (
            "cover-letter skill" in lowered
            or "cannot present genuine-gap skill" in lowered
            or "not relevant to supplied job" in lowered
        ):
            fields.append("skills")
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
        if contract.get("cover_recovery_mode") == "patch":
            return False
        if "transport rejection" in error.casefold():
            return True
        if "rejected the requested call" in error.casefold():
            return False
        return False

    def _clear_cover_patch_recovery(self) -> None:
        self._cover_letter_patch_recovery = None
        self._cover_letter_last_invalid_fingerprint = None
        self._cover_letter_invalid_repeat_count = 0

    def _deterministic_wrapper_templates(self, job: Any) -> dict[str, str]:
        title = " ".join(str(job.title or "this role").split())
        return {
            "lead_in": (
                "My background aligns with the applied responsibilities of this role."
            ),
            "follow_up": (
                "This evidence demonstrates practical experience relevant to the position "
                "and the organization's delivery requirements for evidence grounded systems."
            ),
        }

    def _cover_letter_semantic_duplicate_issue_group(
        self,
        contract: dict[str, Any],
        audit: _CoverLetterAuditResult | None,
        error: str | None,
    ) -> str:
        return self._cover_letter_invalid_fingerprint(
            contract,
            audit,
            error,
        )

    def _cover_letter_invalid_fingerprint(
        self,
        contract: dict[str, Any],
        audit: _CoverLetterAuditResult | None,
        error: str | None,
    ) -> str:
        job_key = hashlib.sha256(
            str(contract.get("target_job_id", "")).encode()
        ).hexdigest()[:12]
        patch_fields = ",".join(sorted(contract.get("cover_patch_fields") or []))
        category = self._cover_letter_rejection_category(error, audit)
        issue_codes = "|".join(
            sorted(
                f"{issue.field}:{issue.code or issue.category}"
                for issue in (audit.issues if audit else [])
            )
        )
        return (
            f"{contract.get('phase', '')}|{job_key}|{patch_fields}|{category}|"
            f"{issue_codes}"
        )

    def _try_cover_letter_deterministic_wrapper_recovery(
        self,
        raw_call: NormalizedToolCall,
        contract: dict[str, Any],
        response: NormalizedAssistantMessage,
        audit: _CoverLetterAuditResult | None,
    ) -> bool:
        if not self._cover_letter_patch_recovery:
            return False
        patch_fields = list(
            self._cover_letter_patch_recovery.get("patch_fields") or []
        )
        if not any(field.startswith("body_paragraph") for field in patch_fields):
            return False
        assert self.registry is not None
        assert self.state is not None
        target_job_id = contract.get("target_job_id")
        if target_job_id is None:
            return False
        job = self.registry._job(target_job_id)
        templates = self._deterministic_wrapper_templates(job)
        base = copy.deepcopy(self._cover_letter_patch_recovery["base_draft"])
        for field in patch_fields:
            if not field.startswith("body_paragraph"):
                continue
            paragraph = base.get(field)
            if not isinstance(paragraph, dict):
                return False
            claim = paragraph.get("selected_candidate_claim")
            if not isinstance(claim, str) or not claim.strip():
                return False
            paragraph["lead_in"] = templates["lead_in"]
            paragraph["follow_up"] = templates["follow_up"]
            base[field] = paragraph
        try:
            full_contract = {**contract, "cover_patch_fields": []}
            transport_args = self._assemble_cover_letter_transport(base, full_contract)
            transport_audit = self._audit_cover_letter_transport(
                transport_args,
                contract,
            )
            transport_audit.raise_if_issues()
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
            self._cover_letter_claim_skill_validation_map = (
                self._selected_claims_by_paragraph_text(
                    execution_call.arguments["plan"],
                    transport_args,
                )
            )
            with self._claim_aware_required_skill_validation(
                self._cover_letter_claim_skill_validation_map
            ):
                self._prevalidate_hydrated_cover_letter_plan(
                    execution_call.arguments["plan"],
                    target_job_id,
                    transport_args=transport_args,
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
            self._clear_cover_patch_recovery()
            if self._last_generation_span is not None:
                existing = self._last_generation_span.record.metadata or {}
                self._last_generation_span.record.metadata = {
                    **existing,
                    "cover_letter_deterministic_wrapper_recovery_applied": True,
                    "cover_letter_paragraph_assembled": True,
                    "cover_letter_claim_exact_substring_verified": True,
                }
            return True
        except (StateInvariantError, ToolRegistryError):
            return False

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

    def _build_hydrated_cover_letter_plan(
        self,
        transport_args: dict[str, Any],
        job_id: str,
    ) -> dict[str, Any]:
        semantic = self._semantic_cover_letter_draft(transport_args)
        skill_registry = self._build_cover_letter_allowed_skill_registry(job_id)
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
        body_paragraphs: list[dict[str, Any]] = []
        paragraph_specs = [semantic.body_paragraph_1]
        if semantic.body_paragraph_2 is not None:
            paragraph_specs.append(semantic.body_paragraph_2)
        for paragraph in paragraph_specs:
            selected_claim = getattr(paragraph, "selected_candidate_claim", None)
            body_paragraphs.append(
                {
                    "text": paragraph.text,
                    "reason": paragraph.reason,
                    "citations": self._close_paragraph_citations(
                        job_id,
                        paragraph.text,
                        skill_items,
                        selected_claim=selected_claim,
                    ),
                }
            )
        self._ensure_cover_body_source_requirements(job_id, body_paragraphs)
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
        transport_skills = transport_args.get("skills")
        if not isinstance(transport_skills, list):
            transport_skills = []
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
            selected_claim = paragraph.get("selected_candidate_claim")
            if not isinstance(selected_claim, str) or not selected_claim:
                issues.append(
                    _CoverLetterAuditIssue(
                        field=field,
                        category="draft_schema",
                        message=f"{field} must include selected_candidate_claim",
                    )
                )
            elif selected_claim not in text:
                issues.append(
                    _CoverLetterAuditIssue(
                        field=field,
                        category="semantic_text",
                        message=(
                            f"{field} assembled text must contain the exact "
                            "selected_candidate_claim"
                        ),
                    )
                )
            numeric_issue = self._audit_numeric_claim_integrity(text, target_job_id)
            if numeric_issue is not None:
                issues.append(
                    _CoverLetterAuditIssue(
                        field=field,
                        category="semantic_text",
                        message=f"{field} {numeric_issue}",
                    )
                )
            if isinstance(selected_claim, str) and selected_claim:
                try:
                    prefix, _, suffix = self._split_paragraph_exact_claim_span(
                        text,
                        selected_claim,
                    )
                    skill_items = [
                        {
                            "skill": skill,
                            "citations": [
                                copy.deepcopy(
                                    skill_registry[
                                        normalize_skill(skill, has_vector_search=True)
                                        or skill.casefold()
                                    ].citation
                                )
                            ],
                        }
                        for skill in transport_skills
                        if isinstance(skill, str)
                        and (
                            normalize_skill(skill, has_vector_search=True)
                            or skill.casefold()
                        )
                        in skill_registry
                    ]
                    paragraph_citations = self._close_paragraph_citations(
                        target_job_id,
                        text,
                        skill_items,
                        selected_claim=selected_claim,
                    )
                    for segment in (prefix, suffix):
                        issue = self._audit_paragraph_free_text_skills(
                            segment,
                            target_job_id,
                            field=field,
                            citations=paragraph_citations,
                        )
                        if issue is not None:
                            issues.append(issue)
                            break
                        segment_gap = self._text_claims_genuine_gap(segment, reconciled)
                        if segment_gap is not None:
                            issues.append(
                                _CoverLetterAuditIssue(
                                    field=field,
                                    category="evidence",
                                    message=(
                                        f"{field} claims genuine-gap skill "
                                        f"{segment_gap!r}"
                                    ),
                                )
                            )
                            break
                except StateInvariantError as exc:
                    code = "exact_claim_missing"
                    message = str(exc)
                    if "exact_claim_repeated" in message:
                        code = "exact_claim_repeated"
                    issues.append(
                        _CoverLetterAuditIssue(
                            field=field,
                            category="evidence",
                            code=code,
                            message=message.split("rejection: ", 1)[-1],
                        )
                    )
            else:
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
        if "pre-execution validation rejection" in lowered:
            if "numeric claim" in lowered or "unsupported numeric" in lowered:
                return "semantic_text"
            if any(marker in lowered for marker in _COVER_CITATION_ERROR_MARKERS):
                return "citation"
            if (
                "claims required skill" in lowered
                or "genuine-gap" in lowered
                or "unsupported required skill" in lowered
            ):
                return "evidence"
            return "validation"
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
        memory_citation_count = 0
        memory_without_evidence = 0
        paragraph_skill_citation_count = 0
        exact_claim_match_count = 0
        if isinstance(plan, dict) and target_job_id:
            for paragraph in plan.get("body_paragraphs", []) or []:
                if not isinstance(paragraph, dict):
                    continue
                paragraph_text = str(paragraph.get("text", ""))
                exact_claim_match_count += len(
                    self._detect_allowed_claims_in_text(paragraph_text, target_job_id)
                )
                for citation in paragraph.get("citations") or []:
                    if not isinstance(citation, dict):
                        continue
                    if citation.get("source_type") == "memory_fact":
                        memory_citation_count += 1
                        if not citation.get("evidence_id"):
                            memory_without_evidence += 1
                    if citation.get("source_type") in {
                        "experience",
                        "experience_bullet",
                        "portfolio_project",
                        "education",
                        "master_skill",
                        "evidence_registry",
                        "memory_fact",
                    }:
                        paragraph_skill_citation_count += 1
        allowed_claim_count = (
            len(self._build_allowed_candidate_claims(target_job_id))
            if target_job_id
            else 0
        )
        diagnostics = {
            "cover_letter_compact_draft": True,
            "cover_letter_hydration_applied": True,
            "cover_letter_paragraph_citation_closure_applied": True,
            "model_argument_mode": "cover_letter_wrapper_draft",
            "raw_model_argument_char_count": raw_size,
            "hydrated_argument_char_count": hydrated_size,
            "cover_letter_allowed_skill_count": allowed_count,
            "cover_letter_selected_skill_count": selected_count,
            "cover_letter_citation_count": citation_count,
            "cover_letter_memory_citation_count": memory_citation_count,
            "cover_letter_memory_citations_without_evidence_id": memory_without_evidence,
            "cover_letter_paragraph_skill_citation_count": paragraph_skill_citation_count,
            "cover_letter_allowed_claim_count": allowed_claim_count,
            "cover_letter_exact_claim_match_count": exact_claim_match_count,
            "cover_letter_paragraph_assembled": True,
            "cover_letter_claim_exact_substring_verified": True,
            "phase": contract["phase"],
            "target_rank": contract.get("target_rank"),
        }
        allowed_claims = list(contract.get("allowed_candidate_claims") or [])
        if target_job_id and not allowed_claims:
            allowed_claims = self._allowed_candidate_claim_texts(target_job_id)
        diagnostics["cover_letter_claim_enum_applied"] = bool(allowed_claims)
        paragraph = raw_call.arguments.get("body_paragraph_1")
        if isinstance(paragraph, dict):
            selected = paragraph.get("selected_candidate_claim")
            if isinstance(selected, str) and selected in allowed_claims:
                diagnostics["cover_letter_selected_claim_index"] = allowed_claims.index(
                    selected
                )
        allowed_hooks = list(contract.get("allowed_company_hooks") or [])
        if target_job_id and not allowed_hooks:
            allowed_hooks = self._select_allowed_company_hooks(target_job_id)
        diagnostics["cover_letter_allowed_hook_count"] = len(allowed_hooks)
        diagnostics["cover_letter_hook_enum_applied"] = bool(allowed_hooks)
        diagnostics["cover_letter_hook_extraction_fallback_used"] = contract.get(
            "cover_letter_hook_extraction_fallback_used",
            False,
        )
        selected_hook = raw_call.arguments.get("company_hook_phrase")
        if isinstance(selected_hook, str) and selected_hook in allowed_hooks:
            diagnostics["cover_letter_selected_hook_index"] = allowed_hooks.index(
                selected_hook
            )
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
            target_job_id = contract.get("target_job_id")
            allowed_hooks = list(contract.get("allowed_company_hooks") or [])
            allowed_claims = list(contract.get("allowed_candidate_claims") or [])
            if target_job_id and not allowed_hooks:
                allowed_hooks, fallback_used = self._ensure_allowed_company_hooks_for_job(
                    target_job_id
                )
                contract = {
                    **contract,
                    "allowed_company_hooks": allowed_hooks,
                    "cover_letter_hook_extraction_fallback_used": fallback_used,
                }
            if target_job_id and not allowed_claims:
                allowed_claims = self._allowed_candidate_claim_texts(target_job_id)
                contract = {**contract, "allowed_candidate_claims": allowed_claims}
            draft_model = self._cover_letter_schema_model_for_contract(contract)
            schema = draft_model.model_json_schema()
            schema = self._apply_cover_hook_enum_to_schema(schema, allowed_hooks)
            schema = self._apply_cover_claim_enum_to_schema(schema, allowed_claims)
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
            allowed_hooks, fallback_used = self._ensure_allowed_company_hooks_for_job(
                target_job_id
            )
            if self._cover_letter_patch_recovery:
                patch_fields = self._cover_letter_patch_recovery["patch_fields"]
                required_shape = self._cover_patch_required_shape(
                    target_job_id,
                    patch_fields,
                    allowed_hooks=allowed_hooks,
                )
            else:
                required_shape = self._cover_draft_required_shape(
                    target_job_id,
                    allowed_hooks,
                )
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
        if allowed_tool == "generate_cover_letter":
            contract["allowed_company_hooks"] = allowed_hooks
            contract["cover_letter_hook_extraction_fallback_used"] = fallback_used
            contract["allowed_candidate_claims"] = self._allowed_candidate_claim_texts(
                target_job_id
            )
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
                if self._last_generation_span is not None:
                    existing = self._last_generation_span.record.metadata or {}
                    allowed_claims = self._allowed_claims_for_contract(contract)
                    selected_index = None
                    paragraph = transport_args.get("body_paragraph_1")
                    if isinstance(paragraph, dict):
                        selected = paragraph.get("selected_candidate_claim")
                        if isinstance(selected, str) and selected in allowed_claims:
                            selected_index = allowed_claims.index(selected)
                    repair_meta: dict[str, Any] = {}
                    if self._cover_letter_wrapper_repair_meta:
                        for field_meta in self._cover_letter_wrapper_repair_meta.values():
                            if isinstance(field_meta, dict):
                                repair_meta.update(field_meta)
                    self._last_generation_span.record.metadata = {
                        **existing,
                        "cover_letter_paragraph_assembled": True,
                        "cover_letter_claim_enum_applied": bool(allowed_claims),
                        "cover_letter_selected_claim_index": selected_index,
                        "cover_letter_claim_exact_substring_verified": True,
                        **repair_meta,
                    }
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
                self._cover_letter_claim_skill_validation_map = (
                    self._selected_claims_by_paragraph_text(
                        execution_call.arguments["plan"],
                        transport_args,
                    )
                )
                with self._claim_aware_required_skill_validation(
                    self._cover_letter_claim_skill_validation_map
                ):
                    self._prevalidate_hydrated_cover_letter_plan(
                        execution_call.arguments["plan"],
                        contract["target_job_id"],
                        transport_args=transport_args,
                    )
                if self._last_generation_span is not None:
                    existing = self._last_generation_span.record.metadata or {}
                    preexecution_meta = self._cover_letter_preexecution_meta or {}
                    self._last_generation_span.record.metadata = {
                        **existing,
                        "cover_letter_preexecution_validation_passed": True,
                        **preexecution_meta,
                    }
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
                self._cover_letter_wrapper_repair_meta = None
                self._cover_letter_preexecution_meta = None
                self._cover_letter_claim_skill_validation_map = None
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
                        try:
                            wrapper = self._parse_cover_letter_wrapper_draft(
                                raw_call.arguments,
                                contract,
                            )
                            wrapper_issues: list[_CoverLetterAuditIssue] = []
                            allowed_claims = self._allowed_claims_for_contract(
                                contract
                            )
                            for field in ("body_paragraph_1", "body_paragraph_2"):
                                paragraph = wrapper.get(field)
                                if isinstance(paragraph, dict):
                                    wrapper_issues.extend(
                                        self._audit_paragraph_wrapper_segments(
                                            paragraph,
                                            field,
                                            contract.get("target_job_id", ""),
                                            allowed_claims,
                                        )
                                    )
                            if wrapper_issues:
                                audit = _CoverLetterAuditResult(issues=wrapper_issues)
                            else:
                                transport_args = self._assemble_cover_letter_transport(
                                    wrapper,
                                    {**contract, "cover_patch_fields": []},
                                )
                                audit = self._audit_cover_letter_transport(
                                    transport_args,
                                    contract,
                                )
                        except (StateInvariantError, ToolRegistryError):
                            audit = None
                fingerprint = self._cover_letter_invalid_fingerprint(
                    contract,
                    audit if isinstance(audit, _CoverLetterAuditResult) else None,
                    error,
                )
                if self._last_generation_span is not None:
                    existing = self._last_generation_span.record.metadata or {}
                    rejection_category = self._cover_letter_rejection_category(
                        error,
                        audit if isinstance(audit, _CoverLetterAuditResult) else None,
                    )
                    duplicate_metadata = {
                        **existing,
                        "cover_letter_semantic_duplicate_issue_group": fingerprint,
                    }
                    if rejection_category == "evidence":
                        duplicate_metadata[
                            "cover_letter_evidence_duplicate_issue_group"
                        ] = fingerprint
                    self._last_generation_span.record.metadata = duplicate_metadata
                if fingerprint == self._cover_letter_last_invalid_fingerprint:
                    self._cover_letter_invalid_repeat_count += 1
                else:
                    self._cover_letter_last_invalid_fingerprint = fingerprint
                    self._cover_letter_invalid_repeat_count = 1
                if self._cover_letter_invalid_repeat_count >= 2:
                    if self._cover_letter_duplicate_wrapper_failure(contract, audit):
                        if self._last_generation_span is not None:
                            existing = self._last_generation_span.record.metadata or {}
                            self._last_generation_span.record.metadata = {
                                **existing,
                                "cover_letter_duplicate_invalid_detected": True,
                            }
                        if self._try_cover_letter_deterministic_wrapper_recovery(
                            raw_call,
                            contract,
                            response,
                            audit if isinstance(audit, _CoverLetterAuditResult) else None,
                        ):
                            return 1, 0
                        affected_field = (
                            audit.fields[0]
                            if isinstance(audit, _CoverLetterAuditResult) and audit.fields
                            else "body_paragraph_1"
                        )
                        self._record_invalid(tool_call=raw_call, error=error)
                        if self._last_generation_span is not None:
                            existing = self._last_generation_span.record.metadata or {}
                            self._last_generation_span.record.metadata = {
                                **existing,
                                "cover_letter_duplicate_invalid_stopped": True,
                            }
                        raise DuplicateInvalidOutputError(
                            "Cover letter duplicate invalid output stopped for "
                            f"{affected_field}"
                        )
                if isinstance(audit, _CoverLetterAuditResult) and audit.issues:
                    wrapper_fields = {
                        issue.field
                        for issue in audit.issues
                        if issue.field.startswith("body_paragraph")
                    }
                    if wrapper_fields and self._last_generation_span is not None:
                        existing = self._last_generation_span.record.metadata or {}
                        self._last_generation_span.record.metadata = {
                            **existing,
                            "cover_letter_paragraph_wrapper_rejected": True,
                        }
                self._prepare_cover_patch_recovery(
                    raw_call,
                    contract,
                    audit=audit if isinstance(audit, _CoverLetterAuditResult) else None,
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
                            allowed_hooks=contract.get("allowed_company_hooks"),
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
                    claim_map = self._cover_letter_claim_skill_validation_map or {}
                    with self._claim_aware_required_skill_validation(claim_map):
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
                        allowed_hooks=contract.get("allowed_company_hooks"),
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
