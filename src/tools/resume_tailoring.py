"""Evidence-enforcing deterministic resume tailoring tool (callable tool #4)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pypdf import PdfReader

from src.models.bundle import CandidateBundle
from src.models.candidate import ExperienceBullet, PortfolioProject
from src.models.job import Job
from src.models.memory import CandidateMemory
from src.tools.filtering import normalize_title
from src.tools.fit_analysis import FitAnalysisResult
from src.tools.scoring import JobScore, normalize_skill


class ResumeTailoringError(Exception):
    """Base error for deterministic resume tailoring failures."""


class ResumeInputMismatchError(ResumeTailoringError):
    """Raised when job-scoped inputs do not identify the same job."""


class ResumeEditPlanError(ResumeTailoringError):
    """Raised when an edit plan violates allowed-edit rules."""


class ResumeEvidenceError(ResumeTailoringError):
    """Raised when an edit or citation lacks real candidate evidence."""


class ResumeTemplateError(ResumeTailoringError):
    """Raised when required LaTeX anchors or target blocks are invalid."""


class ProtectedRegionError(ResumeTailoringError):
    """Raised when content outside permitted target blocks changes."""


class ResumeOutputError(ResumeTailoringError):
    """Raised when an output path is unsafe or would overwrite a revision."""


class ResumeCompilationError(ResumeTailoringError):
    """Raised when pdflatex cannot produce a readable PDF."""


class OnePageConstraintError(ResumeTailoringError):
    """Raised when the generated resume is not exactly one page."""


class ResumeEditCategory(StrEnum):
    """The only four permitted resume edit categories."""

    PROFESSIONAL_SUMMARY = "professional_summary"
    EXPERIENCE_BULLET = "experience_bullet"
    SKILLS = "skills"
    PROJECT_SWAP = "project_swap"


class SkillEditOperation(StrEnum):
    """Permitted skills-section operations."""

    SURFACE_ALIGN = "surface_align"
    ADD_EVIDENCED_SKILL = "add_evidenced_skill"


EditSourceType = Literal[
    "job_posting",
    "candidate_profile",
    "education",
    "experience",
    "experience_bullet",
    "portfolio_project",
    "master_skill",
    "evidence_registry",
    "memory_fact",
    "fit_analysis",
    "resume_tex",
]


class EditCitation(BaseModel):
    """Resolvable source citation attached to a proposed or applied edit."""

    source_type: EditSourceType
    source_id: str
    source_field: str
    evidence_id: str | None = None
    supported_claim: str = ""

    @field_validator("source_id", "source_field")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("citation source_id and source_field must be nonempty")
        return value


class SummaryEdit(BaseModel):
    """Replacement text for the professional summary target."""

    new_text: str
    reason: str
    citations: list[EditCitation]

    @field_validator("new_text", "reason")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("summary text and reason must be nonempty")
        return value

    @field_validator("citations")
    @classmethod
    def require_citations(cls, value: list[EditCitation]) -> list[EditCitation]:
        if not value:
            raise ValueError("summary edit requires citations")
        return value


class ExperienceBulletEdit(BaseModel):
    """Replacement text for one designated editable experience bullet."""

    bullet_id: str
    new_text: str
    reason: str
    citations: list[EditCitation]

    @field_validator("bullet_id", "new_text", "reason")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("bullet edit fields must be nonempty")
        return value

    @field_validator("citations")
    @classmethod
    def require_citations(cls, value: list[EditCitation]) -> list[EditCitation]:
        if not value:
            raise ValueError("experience bullet edit requires citations")
        return value


class SkillSectionEdit(BaseModel):
    """One deterministic skills-section surface alignment or evidence-backed addition."""

    operation: SkillEditOperation
    skill: str
    display_skill: str
    category: str | None = None
    reason: str
    citations: list[EditCitation]

    @field_validator("skill", "display_skill", "reason")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("skill edit fields must be nonempty")
        return value

    @field_validator("citations")
    @classmethod
    def require_citations(cls, value: list[EditCitation]) -> list[EditCitation]:
        if not value:
            raise ValueError("skill edit requires citations")
        return value


class ProjectSwapEdit(BaseModel):
    """Project IDs for the exact swap recommended by Fit Analysis."""

    remove_project_id: str
    add_project_id: str
    reason: str
    citations: list[EditCitation]

    @field_validator("remove_project_id", "add_project_id", "reason")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("project swap fields must be nonempty")
        return value

    @field_validator("citations")
    @classmethod
    def require_citations(cls, value: list[EditCitation]) -> list[EditCitation]:
        if not value:
            raise ValueError("project swap requires citations")
        return value


class ResumeEditPlan(BaseModel):
    """Structured plan supplied by the future single runtime LLM agent."""

    job_id: str
    professional_summary: SummaryEdit
    experience_bullet_edits: list[ExperienceBulletEdit]
    skill_section_edits: list[SkillSectionEdit] = Field(default_factory=list)
    project_swap: ProjectSwapEdit | None = None
    plan_rationale: str

    @field_validator("job_id", "plan_rationale")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("job_id and plan_rationale must be nonempty")
        return value

    @model_validator(mode="after")
    def validate_exactly_two_bullets(self) -> "ResumeEditPlan":
        if len(self.experience_bullet_edits) != 2:
            raise ValueError("ResumeEditPlan requires exactly two experience bullet edits")
        ids = [edit.bullet_id for edit in self.experience_bullet_edits]
        if len(set(ids)) != 2:
            raise ValueError("Experience bullet edit IDs must be distinct")
        return self


class ResumeChange(BaseModel):
    """One evidence-cited applied change."""

    category: ResumeEditCategory
    target_id: str
    before: str
    after: str
    reason: str
    citations: list[EditCitation]


class CompilationResult(BaseModel):
    """Bounded pdflatex execution result."""

    command: list[str]
    return_code: int
    pdf_path: Path
    page_count: int
    stdout_tail: str
    stderr_tail: str


class ResumeTailoringResult(BaseModel):
    """Validated result and output paths for one tailored draft revision."""

    job_id: str
    title: str
    company: str
    revision_round: int
    review_feedback: str | None
    edit_categories: list[ResumeEditCategory]
    changes: list[ResumeChange]
    change_count: int
    summary_change_count: int
    experience_bullet_change_count: int
    skill_change_count: int
    project_swap_change_count: int
    base_resume_pdf_path: Path
    draft_tex_path: Path
    draft_pdf_path: Path
    change_log_path: Path
    compilation: CompilationResult
    page_count: int
    evidence_citation_count: int
    base_tex_sha256: str
    tailored_tex_sha256: str
    protected_regions_sha256_before: str
    protected_regions_sha256_after: str
    protected_regions_unchanged: bool
    deterministic_plan_digest: str

    @model_validator(mode="after")
    def validate_required_counts(self) -> "ResumeTailoringResult":
        if self.summary_change_count != 1:
            raise ValueError("Tailored resume must contain exactly one summary change")
        if self.experience_bullet_change_count != 2:
            raise ValueError("Tailored resume must contain exactly two experience bullet changes")
        if self.page_count != 1:
            raise ValueError("Tailored resume must be exactly one page")
        if not self.protected_regions_unchanged:
            raise ValueError("Protected resume regions must remain unchanged")
        return self


_SUMMARY_ANCHOR = "% AGENT-EDIT-TARGET: summary"
_SKILLS_ANCHOR = "% AGENT-EDIT-TARGET: skills"
_BULLET_ANCHORS = {
    "exp-primary-bullet-1": "% AGENT-EDIT-TARGET: experience-bullet-1",
    "exp-primary-bullet-2": "% AGENT-EDIT-TARGET: experience-bullet-2",
}
_PROJECT_ANCHORS = tuple(f"% AGENT-SWAP-TARGET: project-{index}" for index in range(1, 4))
_ALL_REQUIRED_ANCHORS = (_SUMMARY_ANCHOR, *_BULLET_ANCHORS.values(), *_PROJECT_ANCHORS, _SKILLS_ANCHOR)
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?%?")
_SKILL_LINE_PATTERN = re.compile(
    r"^\s*\\small\\item\{\\textbf\{(?P<label>[^}]*)\}\s*(?P<skills>[^}]*)\}\s*$",
    flags=re.MULTILINE,
)
_LATEX_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def latex_escape(text: str) -> str:
    """Escape plain text for safe insertion into LaTeX."""
    return "".join(_LATEX_ESCAPE_MAP.get(character, character) for character in text)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _anchor_content_start(text: str, anchor: str) -> int:
    anchor_start = text.index(anchor)
    line_end = text.find("\n", anchor_start)
    if line_end == -1:
        raise ResumeTemplateError(f"Anchor {anchor!r} is not followed by a target block")
    return line_end + 1


def _require_template_anchors(text: str) -> None:
    for anchor in _ALL_REQUIRED_ANCHORS:
        count = text.count(anchor)
        if count != 1:
            raise ResumeTemplateError(
                f"Required LaTeX anchor {anchor!r} must occur exactly once; found {count}"
            )


def _find_balanced_command(text: str, anchor: str, command: str) -> tuple[int, int]:
    """Find one braced command after an anchor and return its full span."""
    search_start = _anchor_content_start(text, anchor)
    command_start = text.find(command, search_start)
    if command_start == -1:
        raise ResumeTemplateError(f"Target command {command!r} missing after {anchor!r}")
    brace_start = command_start + len(command) - 1
    depth = 0
    escaped = False
    for index in range(brace_start, len(text)):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return command_start, index + 1
    raise ResumeTemplateError(f"Unbalanced target command after {anchor!r}")


def _target_spans(text: str) -> list[tuple[int, int, str]]:
    """Return all permitted mutable spans for protected-region hashing."""
    _require_template_anchors(text)
    spans: list[tuple[int, int, str]] = []

    summary_start = _anchor_content_start(text, _SUMMARY_ANCHOR)
    summary_end = text.index("%----------EDUCATION----------", summary_start)
    spans.append((summary_start, summary_end, "SUMMARY"))

    for bullet_id, anchor in _BULLET_ANCHORS.items():
        start, end = _find_balanced_command(text, anchor, r"\resumeItem{")
        spans.append((start, end, f"BULLET:{bullet_id}"))

    skills_start = _anchor_content_start(text, _SKILLS_ANCHOR)
    skills_end = text.index(r"\end{itemize}", skills_start) + len(r"\end{itemize}")
    spans.append((skills_start, skills_end, "SKILLS"))

    project_section_end = text.index(r"\resumeEntryListEnd", text.index(_PROJECT_ANCHORS[-1]))
    for index, anchor in enumerate(_PROJECT_ANCHORS):
        start = _anchor_content_start(text, anchor)
        end = (
            text.index(_PROJECT_ANCHORS[index + 1], start)
            if index + 1 < len(_PROJECT_ANCHORS)
            else project_section_end
        )
        spans.append((start, end, f"PROJECT:{index + 1}"))
    return spans


def _protected_representation(text: str) -> str:
    protected = text
    for start, end, label in sorted(_target_spans(text), reverse=True):
        protected = protected[:start] + f"<ALLOWED:{label}>" + protected[end:]
    return protected


def _replace_span(text: str, span: tuple[int, int], replacement: str) -> str:
    return text[: span[0]] + replacement + text[span[1] :]


def _extract_command_argument(text: str, span: tuple[int, int]) -> str:
    command = text[span[0] : span[1]]
    return command[command.index("{") + 1 : -1].strip()


def _all_bullets(bundle: CandidateBundle) -> dict[str, tuple[object, ExperienceBullet]]:
    return {
        bullet.bullet_id: (entry, bullet)
        for entry in bundle.profile.experience
        for bullet in entry.bullets
    }


def _candidate_supported_skills(
    bundle: CandidateBundle,
    memory: CandidateMemory,
) -> dict[str, set[str]]:
    """Map canonical skills to real supporting source IDs."""
    supported: dict[str, set[str]] = {}

    def add(skill: str, source_id: str) -> None:
        canonical = normalize_skill(skill, has_vector_search=True)
        if canonical:
            supported.setdefault(canonical, set()).add(source_id)

    for skill in bundle.all_master_skills():
        add(skill, "master_skills")
    for project in bundle.all_projects():
        for skill in [*project.technology_stack, *project.skills_demonstrated]:
            add(skill, project.project_id)
    for evidence in bundle.evidence.evidence_records:
        for skill in evidence.supported_skills:
            add(skill, evidence.evidence_id)
    for fact in memory.facts:
        if fact.fact_type == "skill":
            for skill in fact.skill_tags:
                add(skill, fact.fact_id)
            if isinstance(fact.normalized_value, str):
                add(fact.normalized_value, fact.fact_id)
            elif isinstance(fact.normalized_value, list):
                for skill in fact.normalized_value:
                    add(skill, fact.fact_id)
    return supported


def _contains_canonical(text: str, canonical: str) -> bool:
    normalized = normalize_skill(text, has_vector_search=True)
    return re.search(rf"(?<!\w){re.escape(canonical)}(?!\w)", normalized) is not None


def _citation_evidence_records(
    citations: list[EditCitation],
    bundle: CandidateBundle,
) -> list[object]:
    records = []
    for citation in citations:
        evidence_id = citation.evidence_id
        if evidence_id is None and citation.source_type == "evidence_registry":
            evidence_id = citation.source_id
        if evidence_id:
            record = bundle.get_evidence(evidence_id)
            if record is not None:
                records.append(record)
    return records


def _validate_citation(
    citation: EditCitation,
    *,
    job: Job,
    fit_analysis: FitAnalysisResult,
    bundle: CandidateBundle,
    memory: CandidateMemory,
    base_resume_tex_path: Path,
) -> None:
    """Resolve a citation to an actual supplied object and field."""
    source_type = citation.source_type
    source_id = citation.source_id
    source_field = citation.source_field
    evidence = bundle.get_evidence(citation.evidence_id) if citation.evidence_id else None
    if citation.evidence_id and evidence is None:
        raise ResumeEvidenceError(f"Unknown evidence ID {citation.evidence_id!r}")

    if source_type == "job_posting":
        if source_id not in {job.job_id, "job_posting"}:
            raise ResumeEvidenceError(f"Job citation source ID {source_id!r} does not identify supplied job")
        root_field = source_field.split(".", 1)[0]
        if root_field not in Job.model_fields:
            raise ResumeEvidenceError(f"Unknown job citation field {source_field!r}")
        return

    if source_type == "candidate_profile":
        if source_id != bundle.profile.candidate_id:
            raise ResumeEvidenceError(f"Unknown candidate profile source ID {source_id!r}")
        root_field = source_field.split(".", 1)[0]
        if root_field not in type(bundle.profile).model_fields:
            raise ResumeEvidenceError(f"Unknown candidate profile field {source_field!r}")
        return

    if source_type == "experience_bullet":
        bullets = _all_bullets(bundle)
        if source_id not in bullets:
            raise ResumeEvidenceError(f"Unknown experience bullet ID {source_id!r}")
        bullet = bullets[source_id][1]
        if source_field not in type(bullet).model_fields:
            raise ResumeEvidenceError(f"Unknown experience bullet field {source_field!r}")
        if citation.evidence_id and citation.evidence_id not in bullet.evidence_ids:
            raise ResumeEvidenceError(
                f"Evidence {citation.evidence_id!r} does not belong to bullet {source_id!r}"
            )
        return

    if source_type == "experience":
        entry = next((item for item in bundle.profile.experience if item.experience_id == source_id), None)
        if entry is None or source_field not in type(entry).model_fields:
            raise ResumeEvidenceError(f"Unknown experience citation {source_id!r}.{source_field}")
        if citation.evidence_id and citation.evidence_id not in entry.evidence_ids:
            raise ResumeEvidenceError(f"Evidence does not belong to experience {source_id!r}")
        return

    if source_type == "education":
        entry = next((item for item in bundle.profile.education if item.education_id == source_id), None)
        if entry is None or source_field not in type(entry).model_fields:
            raise ResumeEvidenceError(f"Unknown education citation {source_id!r}.{source_field}")
        if citation.evidence_id and citation.evidence_id not in entry.evidence_ids:
            raise ResumeEvidenceError(f"Evidence does not belong to education {source_id!r}")
        return

    if source_type == "portfolio_project":
        project = next((item for item in bundle.all_projects() if item.project_id == source_id), None)
        if project is None or source_field not in type(project).model_fields:
            raise ResumeEvidenceError(f"Unknown portfolio citation {source_id!r}.{source_field}")
        if citation.evidence_id and citation.evidence_id not in project.evidence_ids:
            raise ResumeEvidenceError(f"Evidence does not belong to project {source_id!r}")
        return

    if source_type == "evidence_registry":
        record = bundle.get_evidence(source_id)
        if record is None or source_field not in type(record).model_fields:
            raise ResumeEvidenceError(f"Unknown evidence-registry citation {source_id!r}.{source_field}")
        if citation.evidence_id and citation.evidence_id != source_id:
            raise ResumeEvidenceError("Evidence registry citation evidence_id must equal source_id")
        return

    if source_type == "master_skill":
        categories = {
            "master_skills.languages",
            "master_skills.ml_and_data",
            "master_skills.generative_ai",
            "master_skills.cloud_and_mlops",
            "master_skills.cloud_and_mLOps",
            "master_skills.systems_and_tools",
        }
        if source_id not in categories:
            raise ResumeEvidenceError(f"Unknown master-skill category {source_id!r}")
        if not source_field.startswith("master_skills"):
            raise ResumeEvidenceError(f"Invalid master-skill field {source_field!r}")
        return

    if source_type == "memory_fact":
        fact = next((item for item in memory.facts if item.fact_id == source_id), None)
        if fact is None or source_field not in type(fact).model_fields:
            raise ResumeEvidenceError(f"Unknown memory citation {source_id!r}.{source_field}")
        return

    if source_type == "fit_analysis":
        if source_id not in {fit_analysis.job_id, "fit_analysis"}:
            raise ResumeEvidenceError(f"Fit Analysis citation {source_id!r} does not identify supplied result")
        if source_field.split(".", 1)[0] not in FitAnalysisResult.model_fields:
            raise ResumeEvidenceError(f"Unknown Fit Analysis field {source_field!r}")
        return

    if source_type == "resume_tex":
        if source_id not in {base_resume_tex_path.name, str(base_resume_tex_path)}:
            raise ResumeEvidenceError(f"Resume citation source ID {source_id!r} is not the base resume")
        return

    raise ResumeEvidenceError(f"Unsupported citation source type {source_type!r}")


def _validate_edit_citations(
    citations: list[EditCitation],
    *,
    job: Job,
    fit_analysis: FitAnalysisResult,
    bundle: CandidateBundle,
    memory: CandidateMemory,
    base_resume_tex_path: Path,
) -> None:
    if not citations:
        raise ResumeEvidenceError("Every edit requires at least one citation")
    seen: set[tuple[str, str, str, str | None]] = set()
    for citation in citations:
        key = (
            citation.source_type,
            citation.source_id,
            citation.source_field,
            citation.evidence_id,
        )
        if key in seen:
            raise ResumeEvidenceError("Duplicate citation within an edit")
        seen.add(key)
        _validate_citation(
            citation,
            job=job,
            fit_analysis=fit_analysis,
            bundle=bundle,
            memory=memory,
            base_resume_tex_path=base_resume_tex_path,
        )


def _require_source_types(
    citations: list[EditCitation],
    required: set[str],
    *,
    context: str,
) -> None:
    present = {citation.source_type for citation in citations}
    missing = required - present
    if missing:
        raise ResumeEvidenceError(f"{context} is missing required citation types: {sorted(missing)}")


def _validate_no_genuine_gap_claims(text: str, fit_analysis: FitAnalysisResult) -> None:
    for gap in fit_analysis.core_skills.genuine_gaps:
        canonical = normalize_skill(gap, has_vector_search=True)
        if canonical and _contains_canonical(text, canonical):
            raise ResumeEvidenceError(f"Edited candidate text claims genuine-gap skill {gap!r}")


def _validate_target_company_claim(text: str, job: Job) -> None:
    normalized_company = normalize_title(job.company)
    normalized_text = normalize_title(text)
    if normalized_company and re.search(
        rf"(?<!\w){re.escape(normalized_company)}(?!\w)",
        normalized_text,
    ):
        raise ResumeEvidenceError(
            f"Edited candidate text must not claim or imply employment at target company {job.company!r}"
        )


def _supported_numeric_tokens(bundle: CandidateBundle, memory: CandidateMemory) -> set[str]:
    candidate_text = json.dumps(bundle.model_dump(mode="json"), sort_keys=True)
    memory_text = json.dumps(memory.model_dump(mode="json"), sort_keys=True)
    return set(_NUMBER_PATTERN.findall(candidate_text + " " + memory_text))


def _validate_required_skill_claims(
    text: str,
    job: Job,
    supported_skills: dict[str, set[str]],
) -> None:
    for skill in job.required_skills:
        canonical = normalize_skill(skill, has_vector_search=True)
        if canonical and _contains_canonical(text, canonical) and canonical not in supported_skills:
            raise ResumeEvidenceError(f"Candidate text claims unsupported required skill {skill!r}")


def _validate_summary(
    edit: SummaryEdit,
    *,
    job: Job,
    fit_analysis: FitAnalysisResult,
    bundle: CandidateBundle,
    memory: CandidateMemory,
    supported_skills: dict[str, set[str]],
) -> None:
    if len(edit.new_text) > 600:
        raise ResumeEditPlanError("Professional summary must be concise (600 characters or fewer)")
    job_citations = [c for c in edit.citations if c.source_type == "job_posting"]
    candidate_citations = [
        c
        for c in edit.citations
        if c.source_type
        in {
            "candidate_profile",
            "education",
            "experience",
            "experience_bullet",
            "portfolio_project",
            "master_skill",
            "evidence_registry",
            "memory_fact",
        }
    ]
    if not job_citations or not candidate_citations:
        raise ResumeEvidenceError(
            "Summary requires at least one job-posting and one candidate-evidence citation"
        )
    role = normalize_title(job.title.split("|", 1)[0].strip())
    normalized_summary = normalize_title(edit.new_text)
    role_tokens = [token for token in role.split() if token not in {"remote", "position", "office"}]
    if role_tokens and not all(token in normalized_summary for token in role_tokens):
        raise ResumeEditPlanError("Professional summary must clearly align with the actual job role")
    unsupported_numbers = set(_NUMBER_PATTERN.findall(edit.new_text)) - _supported_numeric_tokens(bundle, memory)
    if unsupported_numbers:
        raise ResumeEvidenceError(
            f"Summary contains unsupported numeric candidate claims: {sorted(unsupported_numbers)}"
        )
    _validate_no_genuine_gap_claims(edit.new_text, fit_analysis)
    _validate_target_company_claim(edit.new_text, job)
    _validate_required_skill_claims(edit.new_text, job, supported_skills)


def _validate_bullet_edit(
    edit: ExperienceBulletEdit,
    bullet: ExperienceBullet,
    *,
    job: Job,
    fit_analysis: FitAnalysisResult,
    bundle: CandidateBundle,
    supported_skills: dict[str, set[str]],
) -> None:
    bullet_citations = [
        citation
        for citation in edit.citations
        if citation.source_type == "experience_bullet" and citation.source_id == bullet.bullet_id
    ]
    job_citations = [citation for citation in edit.citations if citation.source_type == "job_posting"]
    if not bullet_citations or not job_citations:
        raise ResumeEvidenceError(
            f"Bullet {bullet.bullet_id!r} requires its bullet citation and a job-posting citation"
        )
    if not any(
        citation.evidence_id in bullet.evidence_ids
        for citation in bullet_citations
    ):
        raise ResumeEvidenceError(
            f"Bullet {bullet.bullet_id!r} citation must include its actual evidence ID"
        )

    evidence_records = _citation_evidence_records(edit.citations, bundle)
    allowed_numeric_text = bullet.text + " " + " ".join(record.claim for record in evidence_records)
    unsupported_numbers = set(_NUMBER_PATTERN.findall(edit.new_text)) - set(
        _NUMBER_PATTERN.findall(allowed_numeric_text)
    )
    if unsupported_numbers:
        raise ResumeEvidenceError(
            f"Bullet {bullet.bullet_id!r} contains unsupported numeric claims: "
            f"{sorted(unsupported_numbers)}"
        )

    bullet_supported: set[str] = set()
    for evidence_id in bullet.evidence_ids:
        evidence = bundle.get_evidence(evidence_id)
        if evidence:
            bullet_supported.update(
                normalize_skill(skill, has_vector_search=True)
                for skill in evidence.supported_skills
            )
    for candidate_skill in supported_skills:
        if _contains_canonical(edit.new_text, candidate_skill) and candidate_skill not in bullet_supported:
            raise ResumeEvidenceError(
                f"Bullet {bullet.bullet_id!r} introduces capability not supported by its evidence: "
                f"{candidate_skill!r}"
            )
    _validate_no_genuine_gap_claims(edit.new_text, fit_analysis)
    _validate_target_company_claim(edit.new_text, job)
    _validate_required_skill_claims(edit.new_text, job, supported_skills)


def _master_skill_category(bundle: CandidateBundle, canonical: str) -> str:
    category_map = (
        ("Languages:", bundle.profile.master_skills.languages),
        ("ML, Data \\& GenAI:", [*bundle.profile.master_skills.ml_and_data, *bundle.profile.master_skills.generative_ai]),
        (
            "Cloud, MLOps \\& Systems:",
            [*bundle.profile.master_skills.cloud_and_mlops, *bundle.profile.master_skills.systems_and_tools],
        ),
    )
    for label, skills in category_map:
        if any(normalize_skill(skill, has_vector_search=True) == canonical for skill in skills):
            return label
    return "ML, Data \\& GenAI:"


def _citation_supports_skill(
    citation: EditCitation,
    canonical: str,
    bundle: CandidateBundle,
    memory: CandidateMemory,
) -> bool:
    """Return whether this exact candidate citation supports a canonical skill."""
    if citation.source_type == "portfolio_project":
        project = next(
            (item for item in bundle.all_projects() if item.project_id == citation.source_id),
            None,
        )
        terms = [*project.technology_stack, *project.skills_demonstrated] if project else []
    elif citation.source_type == "master_skill":
        key = citation.source_id.rsplit(".", 1)[-1]
        key = "cloud_and_mlops" if key == "cloud_and_mLOps" else key
        terms = list(getattr(bundle.profile.master_skills, key, []))
    elif citation.source_type == "evidence_registry":
        record = bundle.get_evidence(citation.source_id)
        terms = list(record.supported_skills) if record else []
    elif citation.source_type == "experience_bullet":
        record = bundle.get_evidence(citation.evidence_id or "")
        terms = list(record.supported_skills) if record else []
    elif citation.source_type == "memory_fact":
        fact = next((item for item in memory.facts if item.fact_id == citation.source_id), None)
        if fact is None or fact.fact_type != "skill":
            return False
        terms = list(fact.skill_tags)
        if isinstance(fact.normalized_value, str):
            terms.append(fact.normalized_value)
        elif isinstance(fact.normalized_value, list):
            terms.extend(fact.normalized_value)
    else:
        return False
    return any(
        normalize_skill(term, has_vector_search=True) == canonical
        for term in terms
    )


def _parse_skills_block(block: str) -> list[tuple[str, list[str]]]:
    parsed: list[tuple[str, list[str]]] = []
    for match in _SKILL_LINE_PATTERN.finditer(block):
        skills = [skill.strip() for skill in match.group("skills").split(",") if skill.strip()]
        parsed.append((match.group("label"), skills))
    if len(parsed) != 3:
        raise ResumeTemplateError("Skills target must contain exactly three readable category lines")
    return parsed


def _apply_skill_edits(
    text: str,
    edits: list[SkillSectionEdit],
    *,
    job: Job,
    fit_analysis: FitAnalysisResult,
    bundle: CandidateBundle,
    memory: CandidateMemory,
    supported_skills: dict[str, set[str]],
) -> tuple[str, list[ResumeChange]]:
    skills_start = _anchor_content_start(text, _SKILLS_ANCHOR)
    skills_end = text.index(r"\end{itemize}", skills_start) + len(r"\end{itemize}")
    before_block = text[skills_start:skills_end]
    categories = _parse_skills_block(before_block)
    changes: list[ResumeChange] = []

    for edit in edits:
        canonical = normalize_skill(edit.skill, has_vector_search=True)
        display_canonical = normalize_skill(edit.display_skill, has_vector_search=True)
        if not canonical or canonical != display_canonical:
            raise ResumeEditPlanError(
                f"Skill display form {edit.display_skill!r} is not canonically equivalent to {edit.skill!r}"
            )
        if canonical in {
            normalize_skill(gap, has_vector_search=True)
            for gap in fit_analysis.core_skills.genuine_gaps
        }:
            raise ResumeEvidenceError(f"Cannot surface or add genuine-gap skill {edit.skill!r}")

        locations = [
            (category_index, skill_index, skill)
            for category_index, (_, skills) in enumerate(categories)
            for skill_index, skill in enumerate(skills)
            if normalize_skill(skill, has_vector_search=True) == canonical
        ]
        if edit.operation == SkillEditOperation.SURFACE_ALIGN:
            if not locations:
                raise ResumeEditPlanError(
                    f"surface_align requires an existing displayed skill: {edit.skill!r}"
                )
            if canonical not in supported_skills:
                raise ResumeEvidenceError(f"No candidate evidence supports skill {edit.skill!r}")
            category_index, skill_index, old_display = locations[0]
            categories[category_index][1][skill_index] = edit.display_skill
            before = old_display
        else:
            evidenced_elsewhere = {
                normalize_skill(skill, has_vector_search=True)
                for skill in fit_analysis.core_skills.evidenced_elsewhere_skills
            }
            if canonical not in evidenced_elsewhere:
                raise ResumeEvidenceError(
                    f"add_evidenced_skill requires a Fit Analysis evidenced-elsewhere skill: "
                    f"{edit.skill!r}"
                )
            if canonical not in supported_skills:
                raise ResumeEvidenceError(f"No candidate evidence supports skill {edit.skill!r}")
            if locations:
                raise ResumeEditPlanError(
                    f"Skill {edit.skill!r} is already displayed and cannot be added again"
                )
            label = edit.category or _master_skill_category(bundle, canonical)
            category_index = next(
                (index for index, (existing_label, _) in enumerate(categories) if existing_label == label),
                None,
            )
            if category_index is None:
                raise ResumeEditPlanError(f"Unknown existing skills category {label!r}")
            categories[category_index][1].append(edit.display_skill)
            before = ""

        candidate_citations = {
            citation.source_type
            for citation in edit.citations
        } & {
            "portfolio_project",
            "master_skill",
            "evidence_registry",
            "memory_fact",
            "experience_bullet",
        }
        if "job_posting" not in {c.source_type for c in edit.citations} or not candidate_citations:
            raise ResumeEvidenceError(
                f"Skill edit {edit.skill!r} requires job-posting and candidate-evidence citations"
            )
        if not any(
            _citation_supports_skill(citation, canonical, bundle, memory)
            for citation in edit.citations
        ):
            raise ResumeEvidenceError(
                f"Skill edit {edit.skill!r} lacks an exact candidate citation supporting that skill"
            )
        for citation in edit.citations:
            if citation.source_type == "memory_fact":
                fact = next(item for item in memory.facts if item.fact_id == citation.source_id)
                if fact.fact_type != "skill":
                    raise ResumeEvidenceError(
                        f"Memory fact {fact.fact_id!r} is not a skill and cannot authorize a skill edit"
                    )

        changes.append(
            ResumeChange(
                category=ResumeEditCategory.SKILLS,
                target_id=edit.skill,
                before=before,
                after=edit.display_skill,
                reason=edit.reason,
                citations=edit.citations,
            )
        )

    seen: set[str] = set()
    for _, skills in categories:
        for skill in skills:
            canonical = normalize_skill(skill, has_vector_search=True)
            if canonical in seen:
                raise ResumeEditPlanError(f"Skills edits produce duplicate canonical skill {skill!r}")
            seen.add(canonical)

    lines = [r"\begin{itemize}[leftmargin=0.2in, itemsep=1pt]"]
    for label, skills in categories:
        lines.append(f"  \\small\\item{{\\textbf{{{label}}} {', '.join(skills)}}}")
    lines.append(r"\end{itemize}")
    return _replace_span(text, (skills_start, skills_end), "\n".join(lines)), changes


def _render_project(project: PortfolioProject) -> str:
    """Render only controlled portfolio fields in the existing project style."""
    technologies = ", ".join(latex_escape(item) for item in project.technology_stack)
    return (
        f"  \\resumeEntry{{{latex_escape(project.name)}}}{{{project.year}}}\n"
        f"    {{{technologies}}}{{{latex_escape(project.domain)}}}\n"
        "  \\resumeItemListStart\n"
        f"    \\resumeItem{{{latex_escape(project.measurable_result)}}}\n"
        "  \\resumeItemListEnd\n\n"
    )


def _apply_project_swap(
    text: str,
    edit: ProjectSwapEdit,
    *,
    job: Job,
    fit_analysis: FitAnalysisResult,
    bundle: CandidateBundle,
) -> tuple[str, ResumeChange]:
    suggestion = fit_analysis.projects.swap_suggestion
    if suggestion is None:
        raise ResumeEditPlanError("Project swap supplied when Fit Analysis recommends no swap")
    if (
        edit.remove_project_id != suggestion.remove_project_id
        or edit.add_project_id != suggestion.add_project_id
    ):
        raise ResumeEditPlanError(
            "Project swap IDs must exactly match the supplied Fit Analysis recommendation"
        )

    project_by_id = {project.project_id: project for project in bundle.all_projects()}
    remove_project = project_by_id.get(edit.remove_project_id)
    add_project = project_by_id.get(edit.add_project_id)
    if remove_project is None or add_project is None:
        raise ResumeEditPlanError("Project swap references an unknown portfolio project")
    if not remove_project.on_base_resume or add_project.on_base_resume:
        raise ResumeEditPlanError(
            "Project swap must remove a base-resume project and add an off-resume project"
        )
    _require_source_types(
        edit.citations,
        {"job_posting", "portfolio_project", "fit_analysis"},
        context="project swap",
    )
    project_citations = {
        citation.source_id: citation
        for citation in edit.citations
        if citation.source_type == "portfolio_project"
    }
    for project in (remove_project, add_project):
        citation = project_citations.get(project.project_id)
        if citation is None or citation.evidence_id not in project.evidence_ids:
            raise ResumeEvidenceError(
                f"Project swap requires actual evidence citation for {project.project_id!r}"
            )

    project_spans = {
        anchor: (start, end)
        for start, end, label in _target_spans(text)
        for anchor in _PROJECT_ANCHORS
        if label == f"PROJECT:{_PROJECT_ANCHORS.index(anchor) + 1}"
    }
    target_anchor = next(
        (
            anchor
            for anchor, span in project_spans.items()
            if remove_project.name in text[span[0] : span[1]]
        ),
        None,
    )
    if target_anchor is None:
        raise ResumeTemplateError(
            f"Base resume does not contain project {remove_project.name!r} in a project target"
        )
    span = project_spans[target_anchor]
    before = text[span[0] : span[1]].rstrip()
    rendered = _render_project(add_project)
    text = _replace_span(text, span, rendered)
    return text, ResumeChange(
        category=ResumeEditCategory.PROJECT_SWAP,
        target_id=f"{remove_project.project_id}->{add_project.project_id}",
        before=before,
        after=rendered.rstrip(),
        reason=edit.reason,
        citations=edit.citations,
    )


def compile_resume_pdf(tex_path: Path, *, timeout_seconds: int = 60) -> CompilationResult:
    """Compile one generated LaTeX draft and require exactly one PDF page."""
    tex_path = tex_path.resolve()
    output_dir = tex_path.parent
    pdf_path = output_dir / f"{tex_path.stem}.pdf"
    command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    try:
        completed = subprocess.run(
            command,
            cwd=output_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except FileNotFoundError as exc:
        raise ResumeCompilationError("pdflatex executable was not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise ResumeCompilationError(
            f"pdflatex timed out after {timeout_seconds} seconds"
        ) from exc

    stdout_tail = completed.stdout[-4000:]
    stderr_tail = completed.stderr[-4000:]
    if completed.returncode != 0:
        raise ResumeCompilationError(
            f"pdflatex exited with status {completed.returncode}: "
            f"{(stderr_tail or stdout_tail)[-1000:]}"
        )
    if not pdf_path.is_file():
        raise ResumeCompilationError(f"pdflatex did not create expected PDF: {pdf_path}")
    try:
        page_count = len(PdfReader(str(pdf_path)).pages)
    except Exception as exc:  # pypdf exposes several parse-error subclasses
        raise ResumeCompilationError(f"Generated PDF could not be read: {pdf_path}") from exc
    if page_count != 1:
        raise OnePageConstraintError(
            f"Tailored resume must be exactly one page; generated {page_count} pages"
        )

    for suffix in (".aux", ".log", ".out"):
        artifact = output_dir / f"{tex_path.stem}{suffix}"
        if artifact.exists():
            artifact.unlink()
    return CompilationResult(
        command=command,
        return_code=completed.returncode,
        pdf_path=pdf_path,
        page_count=page_count,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def _validate_output_dir(output_dir: Path, revision_round: int) -> tuple[Path, Path, Path, Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / f"resume_draft_r{revision_round}.tex"
    pdf_path = output_dir / f"resume_draft_r{revision_round}.pdf"
    log_path = output_dir / f"change_log_r{revision_round}.json"
    before_pdf_path = output_dir / "resume_before.pdf"
    collisions = [path for path in (tex_path, pdf_path, log_path) if path.exists()]
    if collisions:
        raise ResumeOutputError(
            "Refusing to overwrite existing revision output: "
            + ", ".join(str(path) for path in collisions)
        )
    return tex_path, pdf_path, log_path, before_pdf_path


def resume_tailoring_tool(
    job: Job,
    job_score: JobScore,
    fit_analysis: FitAnalysisResult,
    bundle: CandidateBundle,
    memory: CandidateMemory,
    base_resume_tex_path: Path,
    base_resume_pdf_path: Path,
    output_dir: Path,
    edit_plan: ResumeEditPlan,
    revision_round: int = 0,
    review_feedback: str | None = None,
) -> ResumeTailoringResult:
    """Validate and apply one evidence-grounded, bounded resume edit plan."""
    if job.job_id != job_score.job_id:
        raise ResumeInputMismatchError("job.job_id does not match job_score.job_id")
    if job.job_id != fit_analysis.job_id:
        raise ResumeInputMismatchError("job.job_id does not match fit_analysis.job_id")
    if edit_plan.job_id != job.job_id:
        raise ResumeInputMismatchError("edit_plan.job_id does not match job.job_id")
    if revision_round not in {0, 1, 2}:
        raise ResumeEditPlanError("revision_round must be 0, 1, or 2")
    if revision_round in {1, 2} and not (review_feedback and review_feedback.strip()):
        raise ResumeEditPlanError(
            f"revision_round {revision_round} requires nonempty review_feedback"
        )

    base_resume_tex_path = base_resume_tex_path.resolve()
    base_resume_pdf_path = base_resume_pdf_path.resolve()
    if not base_resume_tex_path.is_file():
        raise ResumeTemplateError(f"Base resume LaTeX not found: {base_resume_tex_path}")
    if not base_resume_pdf_path.is_file():
        raise ResumeOutputError(f"Base resume PDF not found: {base_resume_pdf_path}")

    base_tex_bytes = base_resume_tex_path.read_bytes()
    base_pdf_bytes = base_resume_pdf_path.read_bytes()
    base_text = base_tex_bytes.decode("utf-8")
    _require_template_anchors(base_text)
    protected_before = _sha256_text(_protected_representation(base_text))

    supported_skills = _candidate_supported_skills(bundle, memory)
    all_edits = [
        edit_plan.professional_summary,
        *edit_plan.experience_bullet_edits,
        *edit_plan.skill_section_edits,
        *([edit_plan.project_swap] if edit_plan.project_swap else []),
    ]
    for edit in all_edits:
        _validate_edit_citations(
            edit.citations,
            job=job,
            fit_analysis=fit_analysis,
            bundle=bundle,
            memory=memory,
            base_resume_tex_path=base_resume_tex_path,
        )

    _validate_summary(
        edit_plan.professional_summary,
        job=job,
        fit_analysis=fit_analysis,
        bundle=bundle,
        memory=memory,
        supported_skills=supported_skills,
    )
    if fit_analysis.project_swap_recommended and edit_plan.project_swap is None:
        raise ResumeEditPlanError(
            "Fit Analysis recommends a project swap, so edit_plan.project_swap is required"
        )
    if not fit_analysis.project_swap_recommended and edit_plan.project_swap is not None:
        raise ResumeEditPlanError(
            "Fit Analysis recommends no project swap, so edit_plan.project_swap must be None"
        )

    bullets = _all_bullets(bundle)
    primary = next((entry for entry in bundle.profile.experience if entry.is_primary_role), None)
    if primary is None:
        raise ResumeEditPlanError("Candidate profile has no primary professional experience")
    editable_ids = {
        bullet.bullet_id
        for bullet in primary.bullets
        if bullet.editable_for_job_tailoring and not bullet.immutable_for_job_tailoring
    }
    plan_bullet_ids = {edit.bullet_id for edit in edit_plan.experience_bullet_edits}
    if plan_bullet_ids != editable_ids or len(editable_ids) != 2:
        raise ResumeEditPlanError(
            "Plan must edit exactly the two editable primary experience bullet IDs"
        )
    for edit in edit_plan.experience_bullet_edits:
        _, bullet = bullets[edit.bullet_id]
        _validate_bullet_edit(
            edit,
            bullet,
            job=job,
            fit_analysis=fit_analysis,
            bundle=bundle,
            supported_skills=supported_skills,
        )

    tailored_text = base_text
    changes: list[ResumeChange] = []

    summary_start = _anchor_content_start(tailored_text, _SUMMARY_ANCHOR)
    summary_end = tailored_text.index("%----------EDUCATION----------", summary_start)
    summary_before = tailored_text[summary_start:summary_end].strip()
    summary_after = latex_escape(edit_plan.professional_summary.new_text)
    tailored_text = _replace_span(
        tailored_text,
        (summary_start, summary_end),
        summary_after + "\n\n",
    )
    changes.append(
        ResumeChange(
            category=ResumeEditCategory.PROFESSIONAL_SUMMARY,
            target_id="summary",
            before=summary_before,
            after=edit_plan.professional_summary.new_text,
            reason=edit_plan.professional_summary.reason,
            citations=edit_plan.professional_summary.citations,
        )
    )

    for edit in edit_plan.experience_bullet_edits:
        anchor = _BULLET_ANCHORS[edit.bullet_id]
        span = _find_balanced_command(tailored_text, anchor, r"\resumeItem{")
        before = _extract_command_argument(tailored_text, span)
        rendered = f"\\resumeItem{{{latex_escape(edit.new_text)}}}"
        tailored_text = _replace_span(tailored_text, span, rendered)
        changes.append(
            ResumeChange(
                category=ResumeEditCategory.EXPERIENCE_BULLET,
                target_id=edit.bullet_id,
                before=before,
                after=edit.new_text,
                reason=edit.reason,
                citations=edit.citations,
            )
        )

    tailored_text, skill_changes = _apply_skill_edits(
        tailored_text,
        edit_plan.skill_section_edits,
        job=job,
        fit_analysis=fit_analysis,
        bundle=bundle,
        memory=memory,
        supported_skills=supported_skills,
    )
    changes.extend(skill_changes)

    if edit_plan.project_swap is not None:
        tailored_text, project_change = _apply_project_swap(
            tailored_text,
            edit_plan.project_swap,
            job=job,
            fit_analysis=fit_analysis,
            bundle=bundle,
        )
        changes.append(project_change)

    _require_template_anchors(tailored_text)
    protected_after = _sha256_text(_protected_representation(tailored_text))
    if protected_before != protected_after:
        raise ProtectedRegionError("A protected LaTeX region changed during tailoring")

    tex_path, expected_pdf_path, change_log_path, before_pdf_path = _validate_output_dir(
        output_dir,
        revision_round,
    )
    if before_pdf_path.exists():
        if before_pdf_path.read_bytes() != base_pdf_bytes:
            raise ResumeOutputError("Existing resume_before.pdf does not match supplied base resume")
    else:
        before_pdf_path.write_bytes(base_pdf_bytes)

    tex_path.write_text(tailored_text, encoding="utf-8", newline="\n")
    compilation = compile_resume_pdf(tex_path)
    if compilation.pdf_path.resolve() != expected_pdf_path.resolve():
        raise ResumeCompilationError("Compiler returned an unexpected PDF path")

    category_order = list(dict.fromkeys(change.category for change in changes))
    result = ResumeTailoringResult(
        job_id=job.job_id,
        title=job.title,
        company=job.company,
        revision_round=revision_round,
        review_feedback=review_feedback,
        edit_categories=category_order,
        changes=changes,
        change_count=len(changes),
        summary_change_count=sum(
            change.category == ResumeEditCategory.PROFESSIONAL_SUMMARY for change in changes
        ),
        experience_bullet_change_count=sum(
            change.category == ResumeEditCategory.EXPERIENCE_BULLET for change in changes
        ),
        skill_change_count=sum(
            change.category == ResumeEditCategory.SKILLS for change in changes
        ),
        project_swap_change_count=sum(
            change.category == ResumeEditCategory.PROJECT_SWAP for change in changes
        ),
        base_resume_pdf_path=before_pdf_path,
        draft_tex_path=tex_path,
        draft_pdf_path=compilation.pdf_path,
        change_log_path=change_log_path,
        compilation=compilation,
        page_count=compilation.page_count,
        evidence_citation_count=sum(len(change.citations) for change in changes),
        base_tex_sha256=_sha256_bytes(base_tex_bytes),
        tailored_tex_sha256=_sha256_text(tailored_text),
        protected_regions_sha256_before=protected_before,
        protected_regions_sha256_after=protected_after,
        protected_regions_unchanged=True,
        deterministic_plan_digest=_sha256_text(
            json.dumps(edit_plan.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        ),
    )
    change_log_path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


__all__ = [
    "CompilationResult",
    "EditCitation",
    "ExperienceBulletEdit",
    "OnePageConstraintError",
    "ProjectSwapEdit",
    "ProtectedRegionError",
    "ResumeChange",
    "ResumeCompilationError",
    "ResumeEditCategory",
    "ResumeEditPlan",
    "ResumeEditPlanError",
    "ResumeEvidenceError",
    "ResumeInputMismatchError",
    "ResumeOutputError",
    "ResumeTailoringError",
    "ResumeTailoringResult",
    "ResumeTemplateError",
    "SkillEditOperation",
    "SkillSectionEdit",
    "SummaryEdit",
    "compile_resume_pdf",
    "latex_escape",
    "resume_tailoring_tool",
]
