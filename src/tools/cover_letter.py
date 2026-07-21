"""Evidence-enforcing deterministic Cover Letter Tool (assignment tool #5)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pypdf import PdfReader

from src.models.bundle import CandidateBundle
from src.models.job import Job
from src.models.memory import CandidateMemory, MemoryFact
from src.services.resume_finalizer import FinalizedResumeResult
from src.tools.fit_analysis import FitAnalysisResult
from src.tools.scoring import JobScore, normalize_skill


class CoverLetterError(Exception):
    """Base class for deterministic cover-letter failures."""


class CoverLetterInputMismatchError(CoverLetterError):
    """Raised when job-scoped inputs do not identify the same job."""


class CoverLetterPlanError(CoverLetterError):
    """Raised when the structured plan violates document rules."""


class CoverLetterEvidenceError(CoverLetterError):
    """Raised when a citation or candidate claim is unsupported."""


class CoverLetterFinalizedResumeError(CoverLetterError):
    """Raised when the supplied finalized resume is not an approved artifact."""


class CoverLetterOutputError(CoverLetterError):
    """Raised when output paths are unsafe or would overwrite files."""


class CoverLetterCompilationError(CoverLetterError):
    """Raised when pdflatex cannot produce a readable cover-letter PDF."""


class CoverLetterOnePageConstraintError(CoverLetterError):
    """Raised when a generated or finalized PDF is not exactly one page."""


class CoverLetterCitationSourceType(StrEnum):
    """Supported, resolvable citation sources."""

    JOB_POSTING = "job_posting"
    COMPANY_DETAILS = "company_details"
    CANDIDATE_PROFILE = "candidate_profile"
    EDUCATION = "education"
    EXPERIENCE = "experience"
    EXPERIENCE_BULLET = "experience_bullet"
    PORTFOLIO_PROJECT = "portfolio_project"
    MASTER_SKILL = "master_skill"
    EVIDENCE_REGISTRY = "evidence_registry"
    MEMORY_FACT = "memory_fact"
    FIT_ANALYSIS = "fit_analysis"
    FINALIZED_RESUME = "finalized_resume"


class CoverLetterCitation(BaseModel):
    """A citation that must resolve to one supplied object and real field."""

    source_type: CoverLetterCitationSourceType
    source_id: str
    source_field: str
    evidence_id: str | None = None
    supported_claim: str = ""

    @field_validator("source_id", "source_field")
    @classmethod
    def require_nonempty_locator(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("citation source_id and source_field must be nonempty")
        return value


class CoverLetterParagraph(BaseModel):
    """One evidence-grounded body paragraph supplied in the plan."""

    text: str
    reason: str
    citations: list[CoverLetterCitation]

    @field_validator("text", "reason")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("paragraph text and reason must be nonempty")
        return value

    @field_validator("citations")
    @classmethod
    def require_citations(cls, value: list[CoverLetterCitation]) -> list[CoverLetterCitation]:
        if not value:
            raise ValueError("every cover-letter paragraph requires citations")
        return value


class CoverLetterSkillItem(BaseModel):
    """One relevant skill and its exact candidate evidence."""

    skill: str
    citations: list[CoverLetterCitation]

    @field_validator("skill")
    @classmethod
    def require_nonempty_skill(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("cover-letter skill must be nonempty")
        return value

    @field_validator("citations")
    @classmethod
    def require_citations(cls, value: list[CoverLetterCitation]) -> list[CoverLetterCitation]:
        if not value:
            raise ValueError("every cover-letter skill requires citations")
        return value


class CoverLetterPlan(BaseModel):
    """Structured content plan supplied by the future single runtime LLM agent."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    company_hook_phrase: str
    company_hook_source_field: str
    body_paragraphs: list[CoverLetterParagraph]
    skills: list[CoverLetterSkillItem]
    closing_sentence: str
    plan_rationale: str
    letter_date: date | None = None

    @field_validator(
        "job_id",
        "company_hook_phrase",
        "company_hook_source_field",
        "closing_sentence",
        "plan_rationale",
    )
    @classmethod
    def require_nonempty_plan_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("cover-letter plan text fields must be nonempty")
        return value

    @model_validator(mode="after")
    def validate_plan_shape(self) -> "CoverLetterPlan":
        if len(self.body_paragraphs) not in {1, 2}:
            raise ValueError("CoverLetterPlan requires exactly 1 or 2 body paragraphs")
        if not 3 <= len(self.skills) <= 8:
            raise ValueError("CoverLetterPlan requires between 3 and 8 skills")
        return self


class CoverLetterCompilationResult(BaseModel):
    """Bounded pdflatex execution result."""

    command: list[str]
    return_code: int
    pdf_path: Path
    page_count: int
    stdout_tail: str
    stderr_tail: str


class CoverLetterResult(BaseModel):
    """Validated output paths and evidence metadata for one cover letter."""

    job_id: str
    title: str
    company: str
    approved_resume_revision: int
    tex_path: Path
    pdf_path: Path
    evidence_log_path: Path
    compilation: CoverLetterCompilationResult
    page_count: int
    company_hook_phrase: str
    paragraph_count: int
    skill_count: int
    candidate_source_citation_count: int
    job_source_citation_count: int
    plan_digest: str
    tex_sha256: str
    pdf_sha256: str
    no_fabrication_validated: bool

    @model_validator(mode="after")
    def validate_success_invariants(self) -> "CoverLetterResult":
        if self.page_count != 1:
            raise ValueError("successful cover letter must be exactly one page")
        if self.paragraph_count not in {1, 2}:
            raise ValueError("successful cover letter must have 1 or 2 body paragraphs")
        if not 3 <= self.skill_count <= 8:
            raise ValueError("successful cover letter must have 3 to 8 skills")
        if self.candidate_source_citation_count <= 0:
            raise ValueError("successful cover letter requires candidate-source citations")
        if self.job_source_citation_count <= 0:
            raise ValueError("successful cover letter requires job-source citations")
        if not self.no_fabrication_validated:
            raise ValueError("successful cover letter must pass no-fabrication validation")
        return self


_CANDIDATE_SOURCE_TYPES = {
    CoverLetterCitationSourceType.CANDIDATE_PROFILE,
    CoverLetterCitationSourceType.EDUCATION,
    CoverLetterCitationSourceType.EXPERIENCE,
    CoverLetterCitationSourceType.EXPERIENCE_BULLET,
    CoverLetterCitationSourceType.PORTFOLIO_PROJECT,
    CoverLetterCitationSourceType.MASTER_SKILL,
    CoverLetterCitationSourceType.EVIDENCE_REGISTRY,
    CoverLetterCitationSourceType.MEMORY_FACT,
    CoverLetterCitationSourceType.FINALIZED_RESUME,
}
_ADDITIONAL_CANDIDATE_TYPES = {
    CoverLetterCitationSourceType.PORTFOLIO_PROJECT,
    CoverLetterCitationSourceType.EDUCATION,
    CoverLetterCitationSourceType.MASTER_SKILL,
    CoverLetterCitationSourceType.EVIDENCE_REGISTRY,
    CoverLetterCitationSourceType.MEMORY_FACT,
}
_SKILL_SOURCE_TYPES = {
    CoverLetterCitationSourceType.EXPERIENCE,
    CoverLetterCitationSourceType.EXPERIENCE_BULLET,
    CoverLetterCitationSourceType.PORTFOLIO_PROJECT,
    CoverLetterCitationSourceType.MASTER_SKILL,
    CoverLetterCitationSourceType.EVIDENCE_REGISTRY,
    CoverLetterCitationSourceType.MEMORY_FACT,
}
_JOB_SOURCE_TYPES = {
    CoverLetterCitationSourceType.JOB_POSTING,
    CoverLetterCitationSourceType.COMPANY_DETAILS,
}
_NUMBER_PATTERN = re.compile(r"(?<!\w)\d+(?:\.\d+)?%?(?!\w)")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_MEANINGLESS_HOOK_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
_DEGREE_MARKERS = (
    "phd",
    "ph.d",
    "doctorate",
    "doctoral",
    "master's",
    "masters",
    "master degree",
    "bachelor's",
    "bachelors",
    "bachelor degree",
    "m.s.",
    "b.s.",
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
    """Escape untrusted plain text so it cannot execute LaTeX commands."""
    return "".join(_LATEX_ESCAPE_MAP.get(character, character) for character in text)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _plan_digest(plan: CoverLetterPlan) -> str:
    return _sha256_bytes(_canonical_json(plan.model_dump(mode="json")).encode("utf-8"))


def _word_count(text: str) -> int:
    return len(_WORD_PATTERN.findall(text))


def _normalize_phrase(text: str) -> str:
    text = text.casefold().replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" \t\r\n.,;:!?\"()[]{}")


def _stringify(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _resolve_field(value: Any, field_path: str, *, label: str) -> Any:
    """Resolve a dotted Pydantic/dict/list field path without dynamic evaluation."""
    current = value
    for part in field_path.split("."):
        if isinstance(current, BaseModel):
            if part not in type(current).model_fields:
                raise CoverLetterEvidenceError(f"Unknown {label} citation field {field_path!r}")
            current = getattr(current, part)
        elif isinstance(current, dict):
            if part not in current:
                raise CoverLetterEvidenceError(f"Unknown {label} citation field {field_path!r}")
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit():
            index = int(part)
            if index >= len(current):
                raise CoverLetterEvidenceError(f"Unknown {label} citation field {field_path!r}")
            current = current[index]
        else:
            raise CoverLetterEvidenceError(f"Unknown {label} citation field {field_path!r}")
    return current


def _evidence_for_citation(
    citation: CoverLetterCitation,
    bundle: CandidateBundle,
) -> str:
    if citation.evidence_id is None:
        return ""
    record = bundle.get_evidence(citation.evidence_id)
    if record is None:
        raise CoverLetterEvidenceError(f"Unknown evidence ID {citation.evidence_id!r}")
    if "cover_letter" not in record.allowed_uses:
        raise CoverLetterEvidenceError(
            f"Evidence {record.evidence_id!r} is not authorized for cover letters"
        )
    return record.claim


def _find_experience_bullet(bundle: CandidateBundle, bullet_id: str):
    for entry in bundle.profile.experience:
        for bullet in entry.bullets:
            if bullet.bullet_id == bullet_id:
                return entry, bullet
    return None


def _master_skill_category(bundle: CandidateBundle, source_id: str) -> list[str] | None:
    key = source_id.rsplit(".", 1)[-1]
    if key == "cloud_and_mLOps":
        key = "cloud_and_mlops"
    if key not in type(bundle.profile.master_skills).model_fields:
        return None
    value = getattr(bundle.profile.master_skills, key)
    return list(value)


def _validate_and_resolve_citation(
    citation: CoverLetterCitation,
    *,
    job: Job,
    fit_analysis: FitAnalysisResult,
    bundle: CandidateBundle,
    memory: CandidateMemory,
    finalized_resume: FinalizedResumeResult,
) -> str:
    """Resolve one citation and return its candidate/job evidence text."""
    evidence_claim = _evidence_for_citation(citation, bundle)
    source_type = citation.source_type
    source_id = citation.source_id
    field = citation.source_field
    value: Any

    if source_type == CoverLetterCitationSourceType.JOB_POSTING:
        if citation.evidence_id:
            raise CoverLetterEvidenceError("Job citations cannot attach candidate evidence IDs")
        if source_id not in {job.job_id, "job_posting"}:
            raise CoverLetterEvidenceError(
                f"Job citation source ID {source_id!r} does not identify supplied job"
            )
        value = _resolve_field(job, field, label="job")
    elif source_type == CoverLetterCitationSourceType.COMPANY_DETAILS:
        if citation.evidence_id:
            raise CoverLetterEvidenceError(
                "Company Details citations cannot attach candidate evidence IDs"
            )
        if source_id not in {job.job_id, "company_details"} or field != "company_details":
            raise CoverLetterEvidenceError("Company Details citation must identify job.company_details")
        value = job.company_details
    elif source_type == CoverLetterCitationSourceType.CANDIDATE_PROFILE:
        if citation.evidence_id:
            raise CoverLetterEvidenceError(
                "Candidate-profile citations must use their resolved profile field"
            )
        if source_id != bundle.profile.candidate_id:
            raise CoverLetterEvidenceError(f"Unknown candidate profile source ID {source_id!r}")
        value = _resolve_field(bundle.profile, field, label="candidate-profile")
    elif source_type == CoverLetterCitationSourceType.EDUCATION:
        entry = next(
            (item for item in bundle.profile.education if item.education_id == source_id),
            None,
        )
        if entry is None:
            raise CoverLetterEvidenceError(f"Unknown education ID {source_id!r}")
        value = _resolve_field(entry, field, label="education")
        if citation.evidence_id and citation.evidence_id not in entry.evidence_ids:
            raise CoverLetterEvidenceError(f"Evidence does not belong to education {source_id!r}")
    elif source_type == CoverLetterCitationSourceType.EXPERIENCE:
        entry = next(
            (item for item in bundle.profile.experience if item.experience_id == source_id),
            None,
        )
        if entry is None:
            raise CoverLetterEvidenceError(f"Unknown experience ID {source_id!r}")
        value = _resolve_field(entry, field, label="experience")
        if citation.evidence_id and citation.evidence_id not in entry.evidence_ids:
            raise CoverLetterEvidenceError(f"Evidence does not belong to experience {source_id!r}")
    elif source_type == CoverLetterCitationSourceType.EXPERIENCE_BULLET:
        located = _find_experience_bullet(bundle, source_id)
        if located is None:
            raise CoverLetterEvidenceError(f"Unknown experience bullet ID {source_id!r}")
        _, bullet = located
        value = _resolve_field(bullet, field, label="experience-bullet")
        if citation.evidence_id and citation.evidence_id not in bullet.evidence_ids:
            raise CoverLetterEvidenceError(
                f"Evidence does not belong to experience bullet {source_id!r}"
            )
    elif source_type == CoverLetterCitationSourceType.PORTFOLIO_PROJECT:
        project = next(
            (item for item in bundle.all_projects() if item.project_id == source_id),
            None,
        )
        if project is None:
            raise CoverLetterEvidenceError(f"Unknown portfolio project ID {source_id!r}")
        value = _resolve_field(project, field, label="portfolio-project")
        if citation.evidence_id and citation.evidence_id not in project.evidence_ids:
            raise CoverLetterEvidenceError(f"Evidence does not belong to project {source_id!r}")
    elif source_type == CoverLetterCitationSourceType.MASTER_SKILL:
        skills = _master_skill_category(bundle, source_id)
        accepted_fields = {source_id, source_id.replace("cloud_and_mLOps", "cloud_and_mlops")}
        if skills is None or field not in accepted_fields | {"master_skills"}:
            raise CoverLetterEvidenceError(
                f"Unknown master-skill citation {source_id!r}.{field}"
            )
        if citation.evidence_id:
            record = bundle.get_evidence(citation.evidence_id)
            source_aliases = {
                source_id,
                source_id.replace("cloud_and_mlops", "cloud_and_mLOps"),
            }
            if record is None or record.source_record_id not in source_aliases:
                raise CoverLetterEvidenceError(
                    f"Evidence does not belong to master-skill category {source_id!r}"
                )
        value = skills
    elif source_type == CoverLetterCitationSourceType.EVIDENCE_REGISTRY:
        record = bundle.get_evidence(source_id)
        if record is None:
            raise CoverLetterEvidenceError(f"Unknown evidence-registry ID {source_id!r}")
        if "cover_letter" not in record.allowed_uses:
            raise CoverLetterEvidenceError(
                f"Evidence {source_id!r} is not authorized for cover letters"
            )
        value = _resolve_field(record, field, label="evidence-registry")
        if citation.evidence_id and citation.evidence_id != source_id:
            raise CoverLetterEvidenceError(
                "Evidence registry citation evidence_id must equal source_id"
            )
        evidence_claim = record.claim
    elif source_type == CoverLetterCitationSourceType.MEMORY_FACT:
        fact = next((item for item in memory.facts if item.fact_id == source_id), None)
        if fact is None:
            raise CoverLetterEvidenceError(f"Unknown memory fact ID {source_id!r}")
        if citation.evidence_id and citation.evidence_id not in fact.evidence_refs:
            raise CoverLetterEvidenceError(
                f"Evidence does not belong to memory fact {source_id!r}"
            )
        value = _resolve_field(fact, field, label="memory-fact")
    elif source_type == CoverLetterCitationSourceType.FIT_ANALYSIS:
        if citation.evidence_id:
            raise CoverLetterEvidenceError(
                "Fit Analysis citations cannot attach candidate evidence IDs"
            )
        if source_id not in {fit_analysis.job_id, "fit_analysis"}:
            raise CoverLetterEvidenceError(
                f"Fit Analysis citation {source_id!r} does not identify supplied result"
            )
        value = _resolve_field(fit_analysis, field, label="fit-analysis")
    elif source_type == CoverLetterCitationSourceType.FINALIZED_RESUME:
        if citation.evidence_id:
            raise CoverLetterEvidenceError(
                "Finalized-resume citations cannot attach candidate evidence IDs"
            )
        if source_id not in {finalized_resume.job_id, "finalized_resume"}:
            raise CoverLetterEvidenceError(
                f"Finalized resume citation {source_id!r} does not identify supplied result"
            )
        value = _resolve_field(finalized_resume, field, label="finalized-resume")
    else:  # pragma: no cover - enum validation prevents this
        raise CoverLetterEvidenceError(f"Unsupported citation source type {source_type!r}")

    return " ".join(part for part in (_stringify(value), evidence_claim) if part)


def _validate_company_hook(plan: CoverLetterPlan, job: Job) -> CoverLetterCitation:
    if plan.company_hook_source_field != "company_details":
        raise CoverLetterPlanError(
            "company_hook_source_field must be exactly 'company_details'"
        )
    hook_words = _WORD_PATTERN.findall(plan.company_hook_phrase)
    meaningful = [
        word for word in hook_words if word.casefold() not in _MEANINGLESS_HOOK_WORDS
    ]
    if len(meaningful) < 4:
        raise CoverLetterPlanError("Company Details hook requires at least 4 meaningful words")
    if len(hook_words) > 30:
        raise CoverLetterPlanError("Company Details hook must contain no more than 30 words")
    normalized_hook = _normalize_phrase(plan.company_hook_phrase)
    normalized_details = _normalize_phrase(job.company_details)
    if normalized_hook not in normalized_details:
        raise CoverLetterEvidenceError(
            "company_hook_phrase must be an exact normalized substring of Company Details"
        )
    return CoverLetterCitation(
        source_type=CoverLetterCitationSourceType.COMPANY_DETAILS,
        source_id=job.job_id,
        source_field="company_details",
        supported_claim=plan.company_hook_phrase,
    )


def _citation_supports_skill(
    citation: CoverLetterCitation,
    canonical_skill: str,
    *,
    bundle: CandidateBundle,
    memory: CandidateMemory,
) -> bool:
    terms: list[str] = []
    if citation.source_type == CoverLetterCitationSourceType.MASTER_SKILL:
        terms = _master_skill_category(bundle, citation.source_id) or []
    elif citation.source_type == CoverLetterCitationSourceType.PORTFOLIO_PROJECT:
        project = next(
            (item for item in bundle.all_projects() if item.project_id == citation.source_id),
            None,
        )
        if project:
            terms = [*project.technology_stack, *project.skills_demonstrated]
    elif citation.source_type == CoverLetterCitationSourceType.EVIDENCE_REGISTRY:
        record = bundle.get_evidence(citation.source_id)
        terms = list(record.supported_skills) if record else []
    elif citation.source_type in {
        CoverLetterCitationSourceType.EXPERIENCE,
        CoverLetterCitationSourceType.EXPERIENCE_BULLET,
    }:
        record = bundle.get_evidence(citation.evidence_id or "")
        terms = list(record.supported_skills) if record else []
    elif citation.source_type == CoverLetterCitationSourceType.MEMORY_FACT:
        fact = next(
            (item for item in memory.facts if item.fact_id == citation.source_id),
            None,
        )
        if fact is None or fact.fact_type != "skill":
            return False
        terms = list(fact.skill_tags)
        if isinstance(fact.normalized_value, str):
            terms.append(fact.normalized_value)
        elif isinstance(fact.normalized_value, list):
            terms.extend(str(item) for item in fact.normalized_value)
    for term in terms:
        canonical_term = normalize_skill(term, has_vector_search=True)
        if canonical_term == canonical_skill:
            return True
        if len(canonical_term) >= 4 and len(canonical_skill) >= 4 and (
            canonical_term in canonical_skill or canonical_skill in canonical_term
        ):
            return True
    return False


def _skill_is_job_relevant(skill: str, job: Job) -> bool:
    canonical = normalize_skill(skill, has_vector_search=True)
    for required in job.required_skills:
        required_canonical = normalize_skill(required, has_vector_search=True)
        if (
            canonical == required_canonical
            or (len(canonical) >= 4 and canonical in required_canonical)
            or (len(required_canonical) >= 4 and required_canonical in canonical)
        ):
            return True
    return False


def _validate_skills(
    skills: list[CoverLetterSkillItem],
    *,
    job: Job,
    fit_analysis: FitAnalysisResult,
    bundle: CandidateBundle,
    memory: CandidateMemory,
    resolve,
) -> None:
    if not 3 <= len(skills) <= 8:
        raise CoverLetterPlanError("Cover letter requires between 3 and 8 skills")
    gaps = {
        normalize_skill(gap, has_vector_search=True)
        for gap in fit_analysis.core_skills.genuine_gaps
    }
    seen: set[str] = set()
    for item in skills:
        canonical = normalize_skill(item.skill, has_vector_search=True)
        if not canonical:
            raise CoverLetterPlanError("Cover-letter skills must normalize to nonempty values")
        if canonical in seen:
            raise CoverLetterPlanError(
                f"Duplicate canonical cover-letter skill {item.skill!r}"
            )
        seen.add(canonical)
        if canonical in gaps:
            raise CoverLetterEvidenceError(
                f"Cannot present genuine-gap skill {item.skill!r} as a candidate strength"
            )
        if not _skill_is_job_relevant(item.skill, job):
            raise CoverLetterEvidenceError(
                f"Cover-letter skill {item.skill!r} is not relevant to supplied job requirements"
            )
        for citation in item.citations:
            resolve(citation)
        if not any(c.source_type in _SKILL_SOURCE_TYPES for c in item.citations):
            raise CoverLetterEvidenceError(
                f"Skill {item.skill!r} lacks a permitted candidate-evidence citation"
            )
        if not any(
            _citation_supports_skill(
                citation,
                canonical,
                bundle=bundle,
                memory=memory,
            )
            for citation in item.citations
        ):
            raise CoverLetterEvidenceError(
                f"No cited candidate evidence supports skill {item.skill!r}"
            )
        for citation in item.citations:
            if citation.source_type == CoverLetterCitationSourceType.MEMORY_FACT:
                fact = next(
                    item for item in memory.facts if item.fact_id == citation.source_id
                )
                if fact.fact_type != "skill":
                    raise CoverLetterEvidenceError(
                        f"candidate_fact {fact.fact_id!r} cannot authorize a skill"
                    )


def _validate_target_company_employment(text: str, job: Job) -> None:
    company = re.escape(job.company)
    patterns = (
        rf"\b(?:work|worked|working|employed)\s+(?:at|for|by)\s+{company}\b",
        rf"\b(?:my|the)\s+(?:role|experience|employment)\s+at\s+{company}\b",
        rf"\bwhile\s+at\s+{company}\b",
    )
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
        raise CoverLetterEvidenceError(
            f"Body text claims employment at target company {job.company!r}"
        )


def _validate_employer_claims(text: str, bundle: CandidateBundle) -> None:
    known = {
        _normalize_phrase(entry.employer)
        for entry in bundle.profile.experience
    }
    patterns = (
        re.compile(
            r"\b(?:worked|working|employed)\s+(?:at|for|by)\s+"
            r"([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,4})"
        ),
        re.compile(
            r"\bAt\s+([A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,4}),\s+I\b"
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            claimed = _normalize_phrase(match.group(1))
            if not any(
                claimed.startswith(employer) or employer.startswith(claimed)
                for employer in known
            ):
                raise CoverLetterEvidenceError(
                    f"Body text references unsupported employer {match.group(1)!r}"
                )


def _validate_named_project_claims(text: str, bundle: CandidateBundle) -> None:
    known = {_normalize_phrase(project.name) for project in bundle.all_projects()}
    pattern = re.compile(
        r"\b(?:my|the|documented)\s+"
        r"([A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){0,4})\s+project\b"
    )
    for match in pattern.finditer(text):
        claimed = _normalize_phrase(match.group(1))
        if not any(
            claimed.startswith(project) or project.startswith(claimed)
            for project in known
        ):
            raise CoverLetterEvidenceError(
                f"Body text references unsupported project {match.group(1)!r}"
            )


def _validate_degree_claims(
    text: str,
    citations: list[CoverLetterCitation],
    bundle: CandidateBundle,
) -> None:
    lower = text.casefold()
    if not any(marker in lower for marker in _DEGREE_MARKERS):
        return
    cited_ids = {
        citation.source_id
        for citation in citations
        if citation.source_type == CoverLetterCitationSourceType.EDUCATION
    }
    if not cited_ids:
        raise CoverLetterEvidenceError("Degree claims require an education citation")
    degrees = [
        education.degree
        for education in bundle.profile.education
        if education.education_id in cited_ids
    ]
    normalized_text = _normalize_phrase(text)
    if "phd" in normalized_text or "doctorate" in normalized_text or "doctoral" in normalized_text:
        if not any(
            marker in _normalize_phrase(degree)
            for degree in degrees
            for marker in ("phd", "doctorate", "doctoral")
        ):
            raise CoverLetterEvidenceError("Body text contains an unsupported doctoral degree claim")
    if ("master" in normalized_text or "m.s." in normalized_text) and not any(
        "m.s." in _normalize_phrase(degree) or "master" in _normalize_phrase(degree)
        for degree in degrees
    ):
        raise CoverLetterEvidenceError("Body text contains an unsupported master's degree claim")
    if ("bachelor" in normalized_text or "b.s." in normalized_text) and not any(
        "b.s." in _normalize_phrase(degree) or "bachelor" in _normalize_phrase(degree)
        for degree in degrees
    ):
        raise CoverLetterEvidenceError("Body text contains an unsupported bachelor's degree claim")


def _validate_required_skill_claims(
    text: str,
    citations: list[CoverLetterCitation],
    *,
    job: Job,
    fit_analysis: FitAnalysisResult,
    bundle: CandidateBundle,
    memory: CandidateMemory,
) -> None:
    normalized_text = normalize_skill(text, has_vector_search=True)
    gaps = {
        normalize_skill(gap, has_vector_search=True): gap
        for gap in fit_analysis.core_skills.genuine_gaps
    }
    for canonical, surface in gaps.items():
        if canonical and re.search(rf"(?<!\w){re.escape(canonical)}(?!\w)", normalized_text):
            raise CoverLetterEvidenceError(
                f"Body text presents genuine-gap capability {surface!r}"
            )
    for required in job.required_skills:
        canonical = normalize_skill(required, has_vector_search=True)
        if not canonical or not re.search(
            rf"(?<!\w){re.escape(canonical)}(?!\w)",
            normalized_text,
        ):
            continue
        if not any(
            _citation_supports_skill(
                citation,
                canonical,
                bundle=bundle,
                memory=memory,
            )
            for citation in citations
        ):
            raise CoverLetterEvidenceError(
                f"Body text claims required skill {required!r} without candidate evidence"
            )


def _validate_paragraphs(
    paragraphs: list[CoverLetterParagraph],
    *,
    job: Job,
    fit_analysis: FitAnalysisResult,
    bundle: CandidateBundle,
    memory: CandidateMemory,
    resolve,
) -> None:
    if len(paragraphs) not in {1, 2}:
        raise CoverLetterPlanError("Cover letter requires exactly 1 or 2 body paragraphs")
    if sum(_word_count(paragraph.text) for paragraph in paragraphs) > 220:
        raise CoverLetterPlanError("Total body text must contain no more than 220 words")

    all_types: set[CoverLetterCitationSourceType] = set()
    for paragraph in paragraphs:
        words = _word_count(paragraph.text)
        if not 35 <= words <= 120:
            raise CoverLetterPlanError(
                f"Each body paragraph must contain 35 to 120 words; found {words}"
            )
        if not paragraph.citations:
            raise CoverLetterEvidenceError("Every body paragraph requires citations")
        support_by_citation = {
            id(citation): resolve(citation) for citation in paragraph.citations
        }
        types = {citation.source_type for citation in paragraph.citations}
        all_types.update(types)
        if CoverLetterCitationSourceType.JOB_POSTING not in types:
            raise CoverLetterEvidenceError(
                "Each body paragraph requires a job-posting citation"
            )
        if not types.intersection(_CANDIDATE_SOURCE_TYPES):
            raise CoverLetterEvidenceError(
                "Each body paragraph requires candidate evidence"
            )

        candidate_support = " ".join(
            support_by_citation[id(citation)]
            for citation in paragraph.citations
            if citation.source_type in _CANDIDATE_SOURCE_TYPES
        )
        unsupported_numbers = set(_NUMBER_PATTERN.findall(paragraph.text)) - set(
            _NUMBER_PATTERN.findall(candidate_support)
        )
        if unsupported_numbers:
            raise CoverLetterEvidenceError(
                "Body paragraph contains unsupported numeric candidate claims: "
                f"{sorted(unsupported_numbers)}"
            )
        _validate_target_company_employment(paragraph.text, job)
        _validate_employer_claims(paragraph.text, bundle)
        _validate_named_project_claims(paragraph.text, bundle)
        _validate_degree_claims(paragraph.text, paragraph.citations, bundle)
        _validate_required_skill_claims(
            paragraph.text,
            paragraph.citations,
            job=job,
            fit_analysis=fit_analysis,
            bundle=bundle,
            memory=memory,
        )
        if (
            fit_analysis.seniority.primary_finding.status == "improvement_needed"
            and re.search(r"\bI am (?:a |an )?(?:senior|staff|principal)\b", paragraph.text, re.I)
        ):
            raise CoverLetterEvidenceError(
                "Body text contradicts the supplied seniority finding"
            )

    if CoverLetterCitationSourceType.JOB_POSTING not in all_types:
        raise CoverLetterEvidenceError("Body requires at least one job-posting citation")
    if not all_types.intersection(
        {
            CoverLetterCitationSourceType.EXPERIENCE,
            CoverLetterCitationSourceType.EXPERIENCE_BULLET,
        }
    ):
        raise CoverLetterEvidenceError(
            "Body requires at least one experience or experience-bullet citation"
        )
    if not all_types.intersection(_ADDITIONAL_CANDIDATE_TYPES):
        raise CoverLetterEvidenceError(
            "Body requires an additional portfolio, education, skill, evidence, or memory source"
        )


def _validate_finalized_resume(
    finalized: FinalizedResumeResult,
    *,
    job: Job,
) -> None:
    if finalized.job_id != job.job_id:
        raise CoverLetterInputMismatchError(
            "job.job_id does not match finalized_resume.job_id"
        )
    if finalized.title != job.title or finalized.company != job.company:
        raise CoverLetterFinalizedResumeError(
            "Finalized resume title/company do not match the supplied job"
        )
    if finalized.approved_revision_round not in {0, 1, 2}:
        raise CoverLetterFinalizedResumeError(
            "Finalized resume lacks valid approved revision information"
        )
    if finalized.page_count != 1:
        raise CoverLetterFinalizedResumeError(
            "Finalized resume result is not marked as one page"
        )
    required = {
        "resume_before.pdf": finalized.resume_before_path,
        "resume_after.tex": finalized.resume_after_tex_path,
        "resume_after.pdf": finalized.resume_after_pdf_path,
        "resume_change_log.json": finalized.resume_change_log_path,
    }
    for name, path in required.items():
        if path.is_symlink() or not path.is_file():
            raise CoverLetterFinalizedResumeError(
                f"Approved finalized resume artifact is missing or unsafe: {name}"
            )
        expected_hash = finalized.copied_file_sha256.get(name)
        if not expected_hash or _sha256_path(path) != expected_hash:
            raise CoverLetterFinalizedResumeError(
                f"Finalized resume copied-file hash is invalid for {name}"
            )
    try:
        change_log = json.loads(
            finalized.resume_change_log_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverLetterFinalizedResumeError(
            "Finalized resume change log is unreadable"
        ) from exc
    if change_log.get("job_id") != job.job_id:
        raise CoverLetterFinalizedResumeError(
            "Finalized resume change log job_id does not match"
        )
    if change_log.get("revision_round") != finalized.approved_revision_round:
        raise CoverLetterFinalizedResumeError(
            "Finalized resume change log does not confirm the approved revision"
        )
    try:
        pages = len(PdfReader(str(finalized.resume_after_pdf_path)).pages)
    except Exception as exc:
        raise CoverLetterFinalizedResumeError(
            "Finalized resume_after.pdf is unreadable"
        ) from exc
    if pages != 1:
        raise CoverLetterOnePageConstraintError(
            f"Finalized resume must remain exactly one page; found {pages}"
        )


def _document_date(plan: CoverLetterPlan) -> date:
    """Return the injected document date, or read the current date in one place."""
    return plan.letter_date or date.today()


def _render_opening_sentence(job, plan) -> str:
    """Render a grammatical opening while preserving the grounded hook phrase."""
    hook = plan.company_hook_phrase.strip().rstrip(".!?")
    return (
        f"I am excited to apply for the {latex_escape(job.title)} position at "
        f"{latex_escape(job.company)}. I am especially interested in this "
        "opportunity because the company description highlights the following "
        f"focus: {latex_escape(hook)}."
    )


def _render_contact_header(bundle: CandidateBundle) -> list[str]:
    persona = bundle.profile.persona
    contact_parts = [
        f"{persona.city}, {persona.state}".strip(", "),
        persona.phone,
        persona.email,
        persona.github,
    ]
    contact_parts = [part for part in contact_parts if part and part.strip()]
    lines = [
        rf"\begin{{center}}",
        rf"{{\Large\bfseries {latex_escape(persona.full_name)}}}\\",
    ]
    if contact_parts:
        lines.append(rf"{latex_escape(' | '.join(contact_parts))}")
    lines.append(r"\end{center}")
    return lines


def _render_latex(
    *,
    job: Job,
    bundle: CandidateBundle,
    plan: CoverLetterPlan,
) -> str:
    lines = [
        r"\documentclass[11pt]{article}",
        r"\setlength{\oddsidemargin}{-0.25in}",
        r"\setlength{\evensidemargin}{-0.25in}",
        r"\setlength{\textwidth}{7in}",
        r"\setlength{\topmargin}{-0.5in}",
        r"\setlength{\textheight}{10in}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{8pt}",
        r"\pagestyle{empty}",
        r"\begin{document}",
        *_render_contact_header(bundle),
        "",
        latex_escape(_document_date(plan).strftime("%B %d, %Y")),
        "",
        "Dear Hiring Manager,",
        "",
        _render_opening_sentence(job, plan),
        "",
    ]
    for paragraph in plan.body_paragraphs:
        lines.extend([latex_escape(paragraph.text), ""])
    skills = ", ".join(item.skill for item in plan.skills)
    lines.extend(
        [
            rf"\textbf{{Relevant skills:}} {latex_escape(skills)}",
            "",
            latex_escape(plan.closing_sentence),
            "",
            "Sincerely,",
            "",
            latex_escape(bundle.profile.persona.full_name),
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def compile_cover_letter_pdf(
    tex_path: Path,
    *,
    timeout_seconds: int = 60,
) -> CoverLetterCompilationResult:
    """Compile a cover letter with pdflatex and enforce exactly one page."""
    tex_path = tex_path.resolve()
    output_dir = tex_path.parent
    pdf_path = output_dir / f"{tex_path.stem}.pdf"
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
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
        raise CoverLetterCompilationError("pdflatex executable was not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise CoverLetterCompilationError(
            f"pdflatex timed out after {timeout_seconds} seconds"
        ) from exc

    stdout_tail = completed.stdout[-4000:]
    stderr_tail = completed.stderr[-4000:]
    if completed.returncode != 0:
        raise CoverLetterCompilationError(
            f"pdflatex exited with status {completed.returncode}: "
            f"{(stderr_tail or stdout_tail)[-1000:]}"
        )
    if not pdf_path.is_file():
        raise CoverLetterCompilationError(
            f"pdflatex did not create expected PDF: {pdf_path}"
        )
    try:
        page_count = len(PdfReader(str(pdf_path)).pages)
    except Exception as exc:
        raise CoverLetterCompilationError(
            f"Generated cover-letter PDF could not be read: {pdf_path}"
        ) from exc
    if page_count != 1:
        raise CoverLetterOnePageConstraintError(
            f"Cover letter must be exactly one page; generated {page_count} pages"
        )
    for suffix in (".aux", ".log", ".out"):
        artifact = output_dir / f"{tex_path.stem}{suffix}"
        if artifact.exists():
            artifact.unlink()
    return CoverLetterCompilationResult(
        command=command,
        return_code=completed.returncode,
        pdf_path=pdf_path,
        page_count=page_count,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


def _prepare_output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "cover_letter.tex"
    pdf_path = output_dir / "cover_letter.pdf"
    evidence_path = output_dir / "cover_letter_evidence.json"
    collisions = [path for path in (tex_path, pdf_path, evidence_path) if path.exists()]
    if collisions:
        raise CoverLetterOutputError(
            "Refusing to overwrite existing cover-letter output: "
            + ", ".join(str(path) for path in collisions)
        )
    return tex_path, pdf_path, evidence_path


def validate_cover_letter_plan(
    plan: CoverLetterPlan,
    *,
    job: Job,
    job_score: JobScore,
    fit_analysis: FitAnalysisResult,
    bundle: CandidateBundle,
    memory: CandidateMemory,
    finalized_resume: FinalizedResumeResult,
) -> CoverLetterCitation:
    """Pure validation for a cover letter plan; no rendering or file writes."""
    if job.job_id != job_score.job_id:
        raise CoverLetterInputMismatchError(
            "job.job_id does not match job_score.job_id"
        )
    if job.job_id != fit_analysis.job_id:
        raise CoverLetterInputMismatchError(
            "job.job_id does not match fit_analysis.job_id"
        )
    if plan.job_id != job.job_id:
        raise CoverLetterInputMismatchError("plan.job_id does not match job.job_id")
    if memory.candidate_id != bundle.profile.candidate_id:
        raise CoverLetterInputMismatchError(
            "memory.candidate_id does not match candidate bundle"
        )
    _validate_finalized_resume(finalized_resume, job=job)
    hook_citation = _validate_company_hook(plan, job)

    resolved_cache: dict[int, str] = {}

    def resolve(citation: CoverLetterCitation) -> str:
        key = id(citation)
        if key not in resolved_cache:
            resolved_cache[key] = _validate_and_resolve_citation(
                citation,
                job=job,
                fit_analysis=fit_analysis,
                bundle=bundle,
                memory=memory,
                finalized_resume=finalized_resume,
            )
        return resolved_cache[key]

    resolve(hook_citation)
    _validate_paragraphs(
        plan.body_paragraphs,
        job=job,
        fit_analysis=fit_analysis,
        bundle=bundle,
        memory=memory,
        resolve=resolve,
    )
    _validate_skills(
        plan.skills,
        job=job,
        fit_analysis=fit_analysis,
        bundle=bundle,
        memory=memory,
        resolve=resolve,
    )
    closing_words = _word_count(plan.closing_sentence)
    if not 3 <= closing_words <= 35:
        raise CoverLetterPlanError(
            "closing_sentence must be concise (3 to 35 words)"
        )

    all_citations = [
        hook_citation,
        *(
            citation
            for paragraph in plan.body_paragraphs
            for citation in paragraph.citations
        ),
        *(citation for skill in plan.skills for citation in skill.citations),
    ]
    for citation in all_citations:
        resolve(citation)
    candidate_count = sum(
        citation.source_type in _CANDIDATE_SOURCE_TYPES
        for citation in all_citations
    )
    job_count = sum(
        citation.source_type in _JOB_SOURCE_TYPES
        for citation in all_citations
    )
    if candidate_count <= 0:
        raise CoverLetterEvidenceError(
            "Cover letter requires candidate-source citations"
        )
    if job_count <= 0:
        raise CoverLetterEvidenceError("Cover letter requires job-source citations")
    return hook_citation


def cover_letter_tool(
    job: Job,
    job_score: JobScore,
    fit_analysis: FitAnalysisResult,
    bundle: CandidateBundle,
    memory: CandidateMemory,
    finalized_resume: FinalizedResumeResult,
    output_dir: Path,
    plan: CoverLetterPlan,
) -> CoverLetterResult:
    """Validate, render, compile, and log one evidence-grounded cover letter."""
    if job.job_id != job_score.job_id:
        raise CoverLetterInputMismatchError(
            "job.job_id does not match job_score.job_id"
        )
    if job.job_id != fit_analysis.job_id:
        raise CoverLetterInputMismatchError(
            "job.job_id does not match fit_analysis.job_id"
        )
    if plan.job_id != job.job_id:
        raise CoverLetterInputMismatchError("plan.job_id does not match job.job_id")
    if memory.candidate_id != bundle.profile.candidate_id:
        raise CoverLetterInputMismatchError(
            "memory.candidate_id does not match candidate bundle"
        )
    _validate_finalized_resume(finalized_resume, job=job)
    hook_citation = validate_cover_letter_plan(
        plan,
        job=job,
        job_score=job_score,
        fit_analysis=fit_analysis,
        bundle=bundle,
        memory=memory,
        finalized_resume=finalized_resume,
    )

    all_citations = [
        hook_citation,
        *(
            citation
            for paragraph in plan.body_paragraphs
            for citation in paragraph.citations
        ),
        *(citation for skill in plan.skills for citation in skill.citations),
    ]
    candidate_count = sum(
        citation.source_type in _CANDIDATE_SOURCE_TYPES
        for citation in all_citations
    )
    job_count = sum(
        citation.source_type in _JOB_SOURCE_TYPES
        for citation in all_citations
    )

    tex_path, expected_pdf_path, evidence_path = _prepare_output_paths(output_dir)
    latex = _render_latex(job=job, bundle=bundle, plan=plan)
    tex_path.write_text(latex, encoding="utf-8", newline="\n")
    compilation = compile_cover_letter_pdf(tex_path)
    if compilation.pdf_path.resolve() != expected_pdf_path.resolve():
        raise CoverLetterCompilationError("Compiler returned an unexpected PDF path")

    digest = _plan_digest(plan)
    tex_hash = _sha256_path(tex_path)
    pdf_hash = _sha256_path(compilation.pdf_path)
    result = CoverLetterResult(
        job_id=job.job_id,
        title=job.title,
        company=job.company,
        approved_resume_revision=finalized_resume.approved_revision_round,
        tex_path=tex_path,
        pdf_path=compilation.pdf_path,
        evidence_log_path=evidence_path,
        compilation=compilation,
        page_count=compilation.page_count,
        company_hook_phrase=plan.company_hook_phrase,
        paragraph_count=len(plan.body_paragraphs),
        skill_count=len(plan.skills),
        candidate_source_citation_count=candidate_count,
        job_source_citation_count=job_count,
        plan_digest=digest,
        tex_sha256=tex_hash,
        pdf_sha256=pdf_hash,
        no_fabrication_validated=True,
    )
    evidence_payload = {
        "job_id": job.job_id,
        "title": job.title,
        "company": job.company,
        "finalized_resume_revision": finalized_resume.approved_revision_round,
        "company_hook": {
            "phrase": plan.company_hook_phrase,
            "citation": hook_citation.model_dump(mode="json"),
        },
        "body_paragraphs": [
            {
                "text": paragraph.text,
                "reason": paragraph.reason,
                "citations": [
                    citation.model_dump(mode="json")
                    for citation in paragraph.citations
                ],
            }
            for paragraph in plan.body_paragraphs
        ],
        "skills": [
            {
                "skill": item.skill,
                "citations": [
                    citation.model_dump(mode="json") for citation in item.citations
                ],
            }
            for item in plan.skills
        ],
        "closing_sentence": plan.closing_sentence,
        "output_paths": {
            "tex": str(tex_path),
            "pdf": str(compilation.pdf_path),
            "evidence_log": str(evidence_path),
        },
        "compilation": compilation.model_dump(mode="json"),
        "page_count": compilation.page_count,
        "candidate_source_citation_count": candidate_count,
        "job_source_citation_count": job_count,
        "deterministic_plan_digest": digest,
        "latex_sha256": tex_hash,
        "pdf_sha256": pdf_hash,
        "no_fabrication_validated": True,
    }
    evidence_path.write_text(
        json.dumps(evidence_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


__all__ = [
    "CoverLetterCitation",
    "CoverLetterCitationSourceType",
    "CoverLetterCompilationError",
    "CoverLetterCompilationResult",
    "CoverLetterError",
    "CoverLetterEvidenceError",
    "CoverLetterFinalizedResumeError",
    "CoverLetterInputMismatchError",
    "CoverLetterOnePageConstraintError",
    "CoverLetterOutputError",
    "CoverLetterParagraph",
    "CoverLetterPlan",
    "CoverLetterPlanError",
    "CoverLetterResult",
    "CoverLetterSkillItem",
    "compile_cover_letter_pdf",
    "cover_letter_tool",
    "latex_escape",
]
