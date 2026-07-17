"""Deterministic evidence-grounded fit analysis tool (assignment callable tool #3)."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.models.bundle import CandidateBundle
from src.models.candidate import Education, ExperienceBullet, ExperienceEntry, PortfolioProject
from src.models.job import Job
from src.models.memory import CandidateMemory
from src.tools.scoring import (
    JobScore,
    build_candidate_domain_universe,
    build_candidate_skill_universe,
    normalize_domain_term,
    normalize_skill,
)

FindingStatus = Literal["aligned", "improvement_needed", "genuine_gap", "informational"]
CitationSourceType = Literal[
    "job_posting",
    "candidate_profile",
    "education",
    "experience",
    "experience_bullet",
    "portfolio_project",
    "master_skill",
    "evidence_registry",
    "memory_fact",
    "resume_tex",
]
TailoringActionType = Literal[
    "rewrite_summary_to_emphasize_fit",
    "revise_editable_experience_bullet",
    "add_evidenced_skill",
    "preserve_genuine_gap",
    "swap_project",
    "no_project_swap",
]

_EDUCATION_KEYWORDS = (
    "bachelor's",
    "bachelors",
    "bachelor",
    "master's",
    "masters",
    "master",
    "degree",
    "computer science",
    "data science",
    "engineering",
    "statistics",
    "mathematics",
    "related technical field",
)
_STAFF_TITLE_KEYWORDS = ("staff", "principal", "lead", "manager", "director", "head")
_NONWORD_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_SWAP_SCORE_MARGIN = 10.0


class FitAnalysisError(Exception):
    """Raised when fit analysis inputs are invalid."""


class EvidenceCitation(BaseModel):
    """Structured evidence citation for a fit-analysis finding."""

    source_type: CitationSourceType
    source_id: str
    source_field: str
    supported_claim: str
    evidence_id: str | None = None


class AnalysisFinding(BaseModel):
    """Single evidence-grounded finding within a fit dimension."""

    status: FindingStatus
    summary: str
    citations: list[EvidenceCitation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_citations(self) -> "AnalysisFinding":
        """Require citations for non-informational findings."""
        if self.status != "informational" and not self.citations:
            raise ValueError(f"Finding with status {self.status!r} requires citations")
        seen: set[tuple[str, str, str, str | None]] = set()
        for citation in self.citations:
            key = (
                citation.source_type,
                citation.source_id,
                citation.source_field,
                citation.evidence_id,
            )
            if key in seen:
                raise ValueError("Duplicate citation within a finding")
            seen.add(key)
        return self


class RelevantExperienceAnalysis(BaseModel):
    """Experience alignment analysis."""

    findings: list[AnalysisFinding]


class SeniorityAnalysis(BaseModel):
    """Seniority alignment analysis."""

    primary_finding: AnalysisFinding


class EducationAnalysis(BaseModel):
    """Education alignment analysis."""

    findings: list[AnalysisFinding]


class CoreSkillsAnalysis(BaseModel):
    """Core skills classification analysis."""

    aligned_skills: list[str]
    evidenced_elsewhere_skills: list[str]
    genuine_gaps: list[str]
    findings: list[AnalysisFinding]


class ProjectComparison(BaseModel):
    """Internal project-fit comparison for one portfolio project."""

    project_id: str
    project_name: str
    on_base_resume: bool
    internal_score: float
    required_skill_overlap_count: int
    domain_aligned: bool
    industry_aligned: bool


class ProjectSwapSuggestion(BaseModel):
    """Optional project swap recommendation."""

    remove_project_id: str
    remove_project_name: str
    add_project_id: str
    add_project_name: str
    current_project_score: float
    replacement_project_score: float
    score_improvement: float
    matched_technologies: list[str]
    matched_skills: list[str]
    domain_alignment: str
    industry_alignment: str
    reason: str
    citations: list[EvidenceCitation]


class ProjectsAnalysis(BaseModel):
    """Project alignment and swap analysis."""

    current_project_comparisons: list[ProjectComparison]
    replacement_comparisons: list[ProjectComparison]
    weakest_current_project_id: str
    strongest_current_project_id: str
    swap_suggestion: ProjectSwapSuggestion | None
    findings: list[AnalysisFinding]


class TailoringAction(BaseModel):
    """Deterministic tailoring to-do derived from fit findings."""

    action_type: TailoringActionType
    summary: str
    target_id: str | None = None
    citations: list[EvidenceCitation] = Field(default_factory=list)


class FitAnalysisResult(BaseModel):
    """Evidence-grounded fit analysis for one Top 3 job."""

    job_id: str
    title: str
    company: str
    score_rank: int
    final_score: float
    relevant_experience: RelevantExperienceAnalysis
    seniority: SeniorityAnalysis
    education: EducationAnalysis
    core_skills: CoreSkillsAnalysis
    projects: ProjectsAnalysis
    tailoring_actions: list[TailoringAction]
    formatted_text: str
    evidence_citation_count: int
    genuine_gap_count: int
    evidenced_missing_skill_count: int
    project_swap_recommended: bool


class ResumeVisibility(BaseModel):
    """Deterministic view of what the base resume currently exposes."""

    skills_text: str
    experience_bullet_texts: list[str]
    project_entries: list[str]
    combined_text: str


def _round_score(value: float) -> float:
    return round(value, 2)


def _job_citation(field: str, claim: str) -> EvidenceCitation:
    return EvidenceCitation(
        source_type="job_posting",
        source_id="job_posting",
        source_field=field,
        supported_claim=claim,
    )


def _job_skill_citation(skill: str) -> EvidenceCitation:
    return _job_citation(f"required_skills.{skill}", skill)


def _dedupe_citations(citations: list[EvidenceCitation]) -> list[EvidenceCitation]:
    seen: set[tuple[str, str, str, str | None]] = set()
    deduped: list[EvidenceCitation] = []
    for citation in citations:
        key = (
            citation.source_type,
            citation.source_id,
            citation.source_field,
            citation.evidence_id,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def parse_resume_visibility(resume_tex_path: Path) -> ResumeVisibility:
    """Read the resume LaTeX and extract visible skills, bullets, and projects."""
    if not resume_tex_path.is_file():
        raise FitAnalysisError(f"Resume LaTeX not found: {resume_tex_path}")

    content = resume_tex_path.read_text(encoding="utf-8")
    skills_match = re.search(
        r"\\section\{Skills\}(.*?)(?=\\end\{document\})",
        content,
        flags=re.DOTALL,
    )
    skills_text = skills_match.group(1) if skills_match else ""
    experience_bullet_texts = [
        re.sub(r"\\[a-zA-Z]+", " ", match).strip()
        for match in re.findall(r"\\resumeItem\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", content, flags=re.DOTALL)
    ]
    project_entries = [
        re.sub(r"\\[a-zA-Z]+", " ", " ".join(part for part in groups if part)).strip()
        for groups in re.findall(
            r"\\resumeEntry\{([^}]*)\}\{([^}]*)\}\s*\{([^}]*)\}\{([^}]*)\}",
            content,
        )
    ]
    combined_parts = [skills_text, *experience_bullet_texts, *project_entries]
    combined_text = " ".join(part for part in combined_parts if part)
    return ResumeVisibility(
        skills_text=skills_text,
        experience_bullet_texts=experience_bullet_texts,
        project_entries=project_entries,
        combined_text=combined_text,
    )


def _profile_has_vector_search(bundle: CandidateBundle) -> bool:
    terms = list(bundle.all_master_skills())
    for project in bundle.all_projects():
        terms.extend(project.technology_stack)
        terms.extend(project.skills_demonstrated)
    return any("vector search" in term.casefold() for term in terms)


def _skill_visible_on_resume(
    skill: str,
    visibility: ResumeVisibility,
    *,
    has_vector_search: bool,
) -> bool:
    """Return True when a skill phrase appears on the current resume."""
    canonical = normalize_skill(skill, has_vector_search=has_vector_search)
    if not canonical:
        return False
    normalized_resume = normalize_skill(visibility.combined_text, has_vector_search=has_vector_search)
    if canonical in normalized_resume:
        return True
    tokens = canonical.split()
    if len(tokens) == 1:
        return re.search(rf"\b{re.escape(tokens[0])}\b", normalized_resume) is not None
    return canonical in normalized_resume


def _classify_core_skills(
    job_score: JobScore,
    visibility: ResumeVisibility,
    *,
    has_vector_search: bool,
) -> tuple[list[str], list[str], list[str]]:
    """Split required skills into aligned, evidenced elsewhere, and genuine gaps."""
    aligned: list[str] = []
    evidenced_elsewhere: list[str] = []
    genuine_gaps: list[str] = []

    for evidence in job_score.matched_skill_evidence:
        if evidence.matched:
            if _skill_visible_on_resume(evidence.job_skill, visibility, has_vector_search=has_vector_search):
                aligned.append(evidence.job_skill)
            else:
                evidenced_elsewhere.append(evidence.job_skill)
        else:
            genuine_gaps.append(evidence.job_skill)

    return aligned, evidenced_elsewhere, genuine_gaps


def _experience_overlap_terms(job: Job, bullet: ExperienceBullet) -> set[str]:
    """Return normalized overlap tokens between a bullet and the posting."""
    bullet_text = normalize_skill(bullet.text, has_vector_search=True)
    overlap: set[str] = set()
    for skill in job.required_skills:
        canonical = normalize_skill(skill, has_vector_search=True)
        if canonical and canonical in bullet_text:
            overlap.add(skill)
    job_blob = normalize_skill(
        f"{job.title} {job.industry_domain} {job.job_description}",
        has_vector_search=True,
    )
    for token in ("machine learning", "ai", "data", "model", "python", "rag", "llm"):
        if token in bullet_text and token in job_blob:
            overlap.add(token)
    return overlap


def _analyze_relevant_experience(
    job: Job,
    bundle: CandidateBundle,
    visibility: ResumeVisibility,
) -> RelevantExperienceAnalysis:
    """Analyze professional experience against the posting."""
    findings: list[AnalysisFinding] = []
    primary = next(entry for entry in bundle.profile.experience if entry.is_primary_role)
    best_bullet: ExperienceBullet | None = None
    best_overlap: set[str] = set()

    for bullet in primary.bullets:
        overlap = _experience_overlap_terms(job, bullet)
        if len(overlap) >= len(best_overlap):
            best_overlap = overlap
            best_bullet = bullet

    if best_bullet and best_overlap:
        citations = [
            _job_citation("required_skills", f"Posting requires skills including {', '.join(sorted(best_overlap))}."),
            EvidenceCitation(
                source_type="experience_bullet",
                source_id=best_bullet.bullet_id,
                source_field="text",
                supported_claim=best_bullet.text,
                evidence_id=best_bullet.evidence_ids[0] if best_bullet.evidence_ids else None,
            ),
            EvidenceCitation(
                source_type="experience",
                source_id=primary.experience_id,
                source_field="job_title",
                supported_claim=f"{primary.job_title} at {primary.employer}",
                evidence_id=primary.evidence_ids[0] if primary.evidence_ids else None,
            ),
        ]
        findings.append(
            AnalysisFinding(
                status="aligned",
                summary=(
                    f"Primary experience bullet {best_bullet.bullet_id} aligns with the posting through "
                    f"{', '.join(sorted(best_overlap))}."
                ),
                citations=_dedupe_citations(citations),
            )
        )
    else:
        findings.append(
            AnalysisFinding(
                status="improvement_needed",
                summary=(
                    "Primary professional experience does not show strong direct overlap with the posting's "
                    "required skills or domain language."
                ),
                citations=[
                    _job_citation("required_skills", job.required_skills_raw),
                    EvidenceCitation(
                        source_type="experience",
                        source_id=primary.experience_id,
                        source_field="bullets",
                        supported_claim="Primary role bullets exist but overlap is limited.",
                        evidence_id=primary.evidence_ids[0] if primary.evidence_ids else None,
                    ),
                ],
            )
        )

    internship = next(entry for entry in bundle.profile.experience if not entry.is_primary_role)
    internship_overlap = set()
    for bullet in internship.bullets:
        internship_overlap.update(_experience_overlap_terms(job, bullet))
    if internship_overlap:
        findings.append(
            AnalysisFinding(
                status="aligned",
                summary=(
                    f"Internship experience at {internship.employer} supports additional overlap with "
                    f"{', '.join(sorted(internship_overlap))}."
                ),
                citations=[
                    _job_citation("job_description", "Posting references related ML and data responsibilities."),
                    EvidenceCitation(
                        source_type="experience",
                        source_id=internship.experience_id,
                        source_field="employer",
                        supported_claim=f"{internship.job_title} at {internship.employer}",
                        evidence_id=internship.evidence_ids[0] if internship.evidence_ids else None,
                    ),
                ],
            )
        )

    return RelevantExperienceAnalysis(findings=findings)


def _analyze_seniority(job: Job, job_score: JobScore, bundle: CandidateBundle) -> SeniorityAnalysis:
    """Produce exactly one primary seniority finding."""
    candidate_years = bundle.profile.preferences.years_of_experience
    title_lower = normalize_skill(job.title, has_vector_search=False)

    if job_score.required_minimum_years is not None and job.experience_parse_status in {"exact", "approximate"}:
        minimum = job_score.required_minimum_years
        if candidate_years >= minimum:
            status: FindingStatus = "aligned"
            summary = (
                f"Candidate has {candidate_years} years of experience, meeting the posting minimum of "
                f"{minimum} years ({job.experience_parse_status})."
            )
        else:
            status = "improvement_needed"
            summary = (
                f"Candidate has {candidate_years} years of experience, below the posting minimum of "
                f"{minimum} years."
            )
        citations = [
            EvidenceCitation(
                source_type="candidate_profile",
                source_id=bundle.profile.candidate_id,
                source_field="preferences.years_of_experience",
                supported_claim=str(candidate_years),
            ),
            _job_citation("experience_requirement_raw", job.experience_requirement_raw),
        ]
    elif job.experience_parse_status in {"ambiguous", "unspecified"}:
        if any(keyword in title_lower for keyword in _STAFF_TITLE_KEYWORDS):
            status = "improvement_needed"
            summary = (
                "Posting seniority wording suggests a staff-level role, while the requirement remains "
                f"ambiguous and the candidate has {candidate_years} years."
            )
        elif "senior" in title_lower:
            status = "improvement_needed" if candidate_years < 4 else "aligned"
            summary = (
                f"Posting title includes senior wording with no reliable numeric minimum; candidate has "
                f"{candidate_years} years and scoring experience component is "
                f"{job_score.breakdown.experience_score}."
            )
        else:
            status = "aligned"
            summary = (
                f"No reliable explicit minimum is stated; candidate experience of {candidate_years} years "
                f"aligns with the deterministic scoring experience component "
                f"({job_score.breakdown.experience_score})."
            )
        citations = [
            EvidenceCitation(
                source_type="candidate_profile",
                source_id=bundle.profile.candidate_id,
                source_field="preferences.years_of_experience",
                supported_claim=str(candidate_years),
            ),
            _job_citation("experience_requirement_raw", job.experience_requirement_raw),
            _job_citation("title", job.title),
        ]
    else:
        status = "aligned"
        summary = f"Candidate experience of {candidate_years} years aligns with the posting requirement."
        citations = [
            EvidenceCitation(
                source_type="candidate_profile",
                source_id=bundle.profile.candidate_id,
                source_field="preferences.years_of_experience",
                supported_claim=str(candidate_years),
            ),
            _job_citation("experience_requirement_raw", job.experience_requirement_raw),
        ]

    return SeniorityAnalysis(
        primary_finding=AnalysisFinding(status=status, summary=summary, citations=citations)
    )


def _posting_education_requirements(job: Job) -> list[str]:
    """Detect explicit education requirements in posting text."""
    blob = f"{job.job_description} {job.required_skills_raw} {job.title}".casefold()
    blob = blob.replace("'", "")
    return [keyword.replace("'", "") for keyword in _EDUCATION_KEYWORDS if keyword.replace("'", "") in blob]


def _education_matches_candidate(education: Education, requirements: list[str]) -> bool:
    """Return True when candidate education satisfies explicit requirement keywords."""
    degree_blob = education.degree.casefold()
    normalized_requirements = {requirement.replace("'", "") for requirement in requirements}

    if normalized_requirements.intersection({"master", "masters"}):
        if "m.s." in degree_blob or "master" in degree_blob:
            return True

    if "data science" in normalized_requirements and "data science" in degree_blob:
        return True

    mapping = {
        "bachelor": ("b.s.", "bachelor", "bs"),
        "bachelors": ("b.s.", "bachelor", "bs"),
        "computer science": ("computer science",),
        "engineering": ("engineering", "computer science", "data science"),
        "statistics": ("statistics", "data science"),
        "mathematics": ("mathematics", "statistics"),
    }
    for requirement in normalized_requirements:
        tokens = mapping.get(requirement, (requirement,))
        if any(token in degree_blob for token in tokens):
            return True
    return False


def _analyze_education(job: Job, bundle: CandidateBundle) -> EducationAnalysis:
    """Analyze education requirements against candidate records."""
    requirements = _posting_education_requirements(job)
    findings: list[AnalysisFinding] = []

    if not requirements:
        ms = next((edu for edu in bundle.profile.education if "M.S." in edu.degree), bundle.profile.education[0])
        findings.append(
            AnalysisFinding(
                status="informational",
                summary=(
                    "The posting does not state an explicit degree requirement; candidate education "
                    f"includes {ms.degree} from {ms.institution}."
                ),
                citations=[
                    _job_citation("job_description", "No explicit degree requirement detected."),
                    EvidenceCitation(
                        source_type="education",
                        source_id=ms.education_id,
                        source_field="degree",
                        supported_claim=ms.degree,
                        evidence_id=ms.evidence_ids[0] if ms.evidence_ids else None,
                    ),
                ],
            )
        )
        return EducationAnalysis(findings=findings)

    matched_records = [
        edu for edu in bundle.profile.education if _education_matches_candidate(edu, requirements)
    ]
    if matched_records:
        edu = matched_records[0]
        findings.append(
            AnalysisFinding(
                status="aligned",
                summary=f"Candidate education {edu.degree} satisfies explicit posting education keywords.",
                citations=[
                    _job_citation("job_description", ", ".join(requirements)),
                    EvidenceCitation(
                        source_type="education",
                        source_id=edu.education_id,
                        source_field="degree",
                        supported_claim=edu.degree,
                        evidence_id=edu.evidence_ids[0] if edu.evidence_ids else None,
                    ),
                ],
            )
        )
    else:
        findings.append(
            AnalysisFinding(
                status="improvement_needed",
                summary=(
                    "Posting mentions education keywords that are not directly matched by the recorded "
                    "candidate degrees."
                ),
                citations=[
                    _job_citation("job_description", ", ".join(requirements)),
                    EvidenceCitation(
                        source_type="education",
                        source_id=bundle.profile.education[0].education_id,
                        source_field="degree",
                        supported_claim=bundle.profile.education[0].degree,
                        evidence_id=bundle.profile.education[0].evidence_ids[0]
                        if bundle.profile.education[0].evidence_ids
                        else None,
                    ),
                ],
            )
        )
    return EducationAnalysis(findings=findings)


def _domain_terms_match(left: str, right: str) -> bool:
    left_norm = normalize_domain_term(left)
    right_norm = normalize_domain_term(right)
    return left_norm in right_norm or right_norm in left_norm or left_norm == right_norm


def _project_internal_score(project: PortfolioProject, job: Job) -> tuple[float, int, bool, bool]:
    """Compute internal project comparison score used only within fit analysis."""
    required = [normalize_skill(skill, has_vector_search=True) for skill in job.required_skills]
    project_skills = {
        normalize_skill(skill, has_vector_search=True)
        for skill in [*project.technology_stack, *project.skills_demonstrated]
    }
    overlap = [skill for skill in required if skill in project_skills]
    overlap_score = 0.0 if not required else (len(overlap) / len(required)) * 60.0

    domain_aligned = _domain_terms_match(project.domain, job.industry_domain)
    industry_aligned = _domain_terms_match(project.industry, job.industry_domain)
    domain_score = 25.0 if domain_aligned or industry_aligned else 0.0

    job_blob = normalize_skill(
        f"{job.title} {job.job_description}",
        has_vector_search=True,
    )
    project_blob = normalize_skill(
        f"{project.name} {project.short_description} {' '.join(project.technology_stack)}",
        has_vector_search=True,
    )
    shared_tokens = sum(
        1
        for token in ("ai", "machine learning", "model", "data", "rag", "llm", "forecast", "nlp")
        if token in job_blob and token in project_blob
    )
    keyword_score = min(15.0, shared_tokens * 3.0)

    return _round_score(overlap_score + domain_score + keyword_score), len(overlap), domain_aligned, industry_aligned


def _evidence_sources_for_skill(
    skill: str,
    bundle: CandidateBundle,
    memory: CandidateMemory,
    skill_universe,
) -> list[EvidenceCitation]:
    """Build citations for evidenced-elsewhere skills."""
    citations: list[EvidenceCitation] = []
    canonical = normalize_skill(skill, has_vector_search=True)
    sources = skill_universe.canonical_to_sources.get(canonical, [])
    for source in sources:
        if source.source_type == "master_skill":
            citations.append(
                EvidenceCitation(
                    source_type="master_skill",
                    source_id=source.source_id,
                    source_field="master_skills",
                    supported_claim=source.display_skill,
                )
            )
        elif source.source_type == "project":
            project = next(
                (item for item in bundle.all_projects() if item.project_id == source.source_id),
                None,
            )
            citations.append(
                EvidenceCitation(
                    source_type="portfolio_project",
                    source_id=source.source_id,
                    source_field="skills_demonstrated",
                    supported_claim=source.display_skill,
                    evidence_id=project.evidence_ids[0] if project and project.evidence_ids else None,
                )
            )
        elif source.source_type == "evidence":
            citations.append(
                EvidenceCitation(
                    source_type="evidence_registry",
                    source_id=source.source_id,
                    source_field="supported_skills",
                    supported_claim=source.display_skill,
                    evidence_id=source.source_id,
                )
            )
        elif source.source_type == "memory":
            citations.append(
                EvidenceCitation(
                    source_type="memory_fact",
                    source_id=source.source_id,
                    source_field="skill_tags",
                    supported_claim=source.display_skill,
                )
            )
    return _dedupe_citations(citations)


def _analyze_core_skills(
    job: Job,
    job_score: JobScore,
    bundle: CandidateBundle,
    memory: CandidateMemory,
    visibility: ResumeVisibility,
    skill_universe,
    *,
    has_vector_search: bool,
) -> CoreSkillsAnalysis:
    """Classify required skills and produce findings."""
    aligned, evidenced_elsewhere, genuine_gaps = _classify_core_skills(
        job_score,
        visibility,
        has_vector_search=has_vector_search,
    )
    findings: list[AnalysisFinding] = []

    if aligned:
        findings.append(
            AnalysisFinding(
                status="aligned",
                summary=f"Aligned skills currently visible on the resume: {', '.join(aligned)}.",
                citations=_dedupe_citations(
                    [_job_skill_citation(skill) for skill in aligned[:5]]
                    + [
                        EvidenceCitation(
                            source_type="resume_tex",
                            source_id="sample_resume.tex",
                            source_field="Skills",
                            supported_claim="Skill appears in the current resume source.",
                        )
                    ]
                ),
            )
        )

    for skill in evidenced_elsewhere:
        findings.append(
            AnalysisFinding(
                status="improvement_needed",
                summary=f"{skill} is supported in the profile but not clearly visible on the current resume.",
                citations=_dedupe_citations(
                    [
                        _job_skill_citation(skill),
                        *_evidence_sources_for_skill(skill, bundle, memory, skill_universe),
                    ]
                ),
            )
        )

    for skill in genuine_gaps:
        findings.append(
            AnalysisFinding(
                status="genuine_gap",
                summary=f"{skill} is required by the posting and has no supported candidate evidence.",
                citations=[_job_skill_citation(skill)],
            )
        )

    return CoreSkillsAnalysis(
        aligned_skills=aligned,
        evidenced_elsewhere_skills=evidenced_elsewhere,
        genuine_gaps=genuine_gaps,
        findings=findings,
    )


def _analyze_projects(job: Job, bundle: CandidateBundle) -> ProjectsAnalysis:
    """Compare base-resume and swap-available projects."""
    current_projects = bundle.base_resume_projects()
    replacement_projects = bundle.swap_available_projects()
    current_comparisons: list[ProjectComparison] = []
    replacement_comparisons: list[ProjectComparison] = []

    for project in current_projects:
        score, overlap_count, domain_aligned, industry_aligned = _project_internal_score(project, job)
        current_comparisons.append(
            ProjectComparison(
                project_id=project.project_id,
                project_name=project.name,
                on_base_resume=True,
                internal_score=score,
                required_skill_overlap_count=overlap_count,
                domain_aligned=domain_aligned,
                industry_aligned=industry_aligned,
            )
        )

    for project in replacement_projects:
        score, overlap_count, domain_aligned, industry_aligned = _project_internal_score(project, job)
        replacement_comparisons.append(
            ProjectComparison(
                project_id=project.project_id,
                project_name=project.name,
                on_base_resume=False,
                internal_score=score,
                required_skill_overlap_count=overlap_count,
                domain_aligned=domain_aligned,
                industry_aligned=industry_aligned,
            )
        )

    weakest = min(current_comparisons, key=lambda item: (item.internal_score, item.project_id))
    strongest = max(current_comparisons, key=lambda item: (item.internal_score, item.project_id))

    swap_suggestion: ProjectSwapSuggestion | None = None
    best_replacement: ProjectComparison | None = None
    for candidate in replacement_comparisons:
        if candidate.internal_score < weakest.internal_score + _SWAP_SCORE_MARGIN:
            continue
        if candidate.required_skill_overlap_count == 0 and not (
            candidate.domain_aligned or candidate.industry_aligned
        ):
            continue
        if best_replacement is None or candidate.internal_score > best_replacement.internal_score:
            best_replacement = candidate

    findings: list[AnalysisFinding] = []
    weakest_project = next(p for p in current_projects if p.project_id == weakest.project_id)
    strongest_project = next(p for p in current_projects if p.project_id == strongest.project_id)

    findings.append(
        AnalysisFinding(
            status="improvement_needed",
            summary=(
                f"Current resume project {weakest.project_name} has the lowest internal fit score "
                f"({weakest.internal_score}) for this posting."
            ),
            citations=[
                _job_citation("required_skills", job.required_skills_raw),
                EvidenceCitation(
                    source_type="portfolio_project",
                    source_id=weakest.project_id,
                    source_field="skills_demonstrated",
                    supported_claim=weakest_project.short_description,
                    evidence_id=weakest_project.evidence_ids[0],
                ),
            ],
        )
    )
    findings.append(
        AnalysisFinding(
            status="aligned",
            summary=(
                f"Current resume project {strongest.project_name} is the strongest project match "
                f"({strongest.internal_score}) for this posting."
            ),
            citations=[
                _job_citation("industry_domain", job.industry_domain),
                EvidenceCitation(
                    source_type="portfolio_project",
                    source_id=strongest.project_id,
                    source_field="domain",
                    supported_claim=f"{strongest_project.domain} / {strongest_project.industry}",
                    evidence_id=strongest_project.evidence_ids[0],
                ),
            ],
        )
    )

    if best_replacement is not None:
        add_project = next(p for p in replacement_projects if p.project_id == best_replacement.project_id)
        matched_skills = [
            skill
            for skill in job.required_skills
            if normalize_skill(skill, has_vector_search=True)
            in {
                normalize_skill(item, has_vector_search=True)
                for item in [*add_project.technology_stack, *add_project.skills_demonstrated]
            }
        ]
        matched_technologies = [
            tech
            for tech in add_project.technology_stack
            if normalize_skill(tech, has_vector_search=True)
            in {normalize_skill(skill, has_vector_search=True) for skill in job.required_skills}
        ]
        swap_suggestion = ProjectSwapSuggestion(
            remove_project_id=weakest.project_id,
            remove_project_name=weakest.project_name,
            add_project_id=best_replacement.project_id,
            add_project_name=best_replacement.project_name,
            current_project_score=weakest.internal_score,
            replacement_project_score=best_replacement.internal_score,
            score_improvement=_round_score(best_replacement.internal_score - weakest.internal_score),
            matched_technologies=matched_technologies,
            matched_skills=matched_skills,
            domain_alignment=add_project.domain,
            industry_alignment=add_project.industry,
            reason=(
                f"Replace {weakest.project_name} with {best_replacement.project_name} to improve required-skill "
                f"and domain alignment for this posting."
            ),
            citations=[
                _job_citation("required_skills", job.required_skills_raw),
                EvidenceCitation(
                    source_type="portfolio_project",
                    source_id=weakest.project_id,
                    source_field="project_id",
                    supported_claim=weakest.project_name,
                    evidence_id=weakest_project.evidence_ids[0],
                ),
                EvidenceCitation(
                    source_type="portfolio_project",
                    source_id=add_project.project_id,
                    source_field="project_id",
                    supported_claim=add_project.name,
                    evidence_id=add_project.evidence_ids[0],
                ),
            ],
        )
        findings.append(
            AnalysisFinding(
                status="improvement_needed",
                summary=(
                    f"Swap suggestion: replace {weakest.project_name} with {best_replacement.project_name} "
                    f"(+{swap_suggestion.score_improvement} internal fit points)."
                ),
                citations=swap_suggestion.citations,
            )
        )
    else:
        findings.append(
            AnalysisFinding(
                status="informational",
                summary=(
                    "No valid project swap meets the threshold; the current base-resume projects are the "
                    "best available evidence set for this role."
                ),
                citations=[
                    _job_citation("required_skills", job.required_skills_raw),
                    EvidenceCitation(
                        source_type="portfolio_project",
                        source_id=strongest.project_id,
                        source_field="project_id",
                        supported_claim=strongest.project_name,
                        evidence_id=strongest_project.evidence_ids[0],
                    ),
                ],
            )
        )

    return ProjectsAnalysis(
        current_project_comparisons=current_comparisons,
        replacement_comparisons=replacement_comparisons,
        weakest_current_project_id=weakest.project_id,
        strongest_current_project_id=strongest.project_id,
        swap_suggestion=swap_suggestion,
        findings=findings,
    )


def _build_tailoring_actions(
    result_parts: FitAnalysisResult,
    bundle: CandidateBundle,
    job: Job,
) -> list[TailoringAction]:
    """Create deterministic tailoring to-do actions from fit findings."""
    actions: list[TailoringAction] = []
    primary = next(entry for entry in bundle.profile.experience if entry.is_primary_role)
    editable_bullets = [bullet for bullet in primary.bullets if bullet.editable_for_job_tailoring]

    if result_parts.core_skills.evidenced_elsewhere_skills or result_parts.projects.swap_suggestion:
        actions.append(
            TailoringAction(
                action_type="rewrite_summary_to_emphasize_fit",
                summary="Rewrite the professional summary to emphasize the strongest evidenced fit areas.",
                citations=[_job_citation("title", job.title)],
            )
        )

    bullet_actions = 0
    for finding in result_parts.relevant_experience.findings:
        if finding.status != "improvement_needed" or bullet_actions >= 2:
            continue
        if not editable_bullets:
            break
        bullet = editable_bullets[bullet_actions]
        actions.append(
            TailoringAction(
                action_type="revise_editable_experience_bullet",
                summary=f"Revise {bullet.bullet_id} to strengthen posting alignment.",
                target_id=bullet.bullet_id,
                citations=finding.citations,
            )
        )
        bullet_actions += 1

    for skill in result_parts.core_skills.evidenced_elsewhere_skills:
        actions.append(
            TailoringAction(
                action_type="add_evidenced_skill",
                summary=f"Add evidenced skill {skill} to the resume skills section if space allows.",
                target_id=skill,
                citations=[
                    citation
                    for finding in result_parts.core_skills.findings
                    for citation in finding.citations
                    if any(token in citation.supported_claim for token in (skill,))
                ][:3]
                or [_job_skill_citation(skill)],
            )
        )

    for skill in result_parts.core_skills.genuine_gaps:
        actions.append(
            TailoringAction(
                action_type="preserve_genuine_gap",
                summary=f"Do not add {skill}; it remains a genuine gap with no candidate evidence.",
                target_id=skill,
                citations=[_job_skill_citation(skill)],
            )
        )

    if result_parts.projects.swap_suggestion is not None:
        swap = result_parts.projects.swap_suggestion
        actions.append(
            TailoringAction(
                action_type="swap_project",
                summary=(
                    f"Swap {swap.remove_project_name} ({swap.remove_project_id}) for "
                    f"{swap.add_project_name} ({swap.add_project_id})."
                ),
                target_id=swap.add_project_id,
                citations=swap.citations,
            )
        )
    else:
        actions.append(
            TailoringAction(
                action_type="no_project_swap",
                summary="Keep the current base-resume projects; no swap meets the evidence threshold.",
                citations=[
                    _job_citation("required_skills", job.required_skills_raw),
                ],
            )
        )

    return actions


def _format_finding_line(finding: AnalysisFinding) -> str:
    marker = {"aligned": "✅", "improvement_needed": "❌", "genuine_gap": "❌", "informational": "ℹ️"}[
        finding.status
    ]
    citation_refs = []
    for citation in finding.citations:
        ref = citation.evidence_id or citation.source_id
        citation_refs.append(f"[{ref}]")
    suffix = f" {' '.join(citation_refs)}" if citation_refs else ""
    return f"{marker} {finding.summary}{suffix}"


def _build_formatted_text(result: FitAnalysisResult) -> str:
    """Build user-facing formatted fit analysis text."""
    lines = [
        "Tell me why this job is a good fit for me.",
        "",
        "Relevant Experience",
    ]
    lines.extend(_format_finding_line(finding) for finding in result.relevant_experience.findings)
    lines.extend(["", "Seniority", _format_finding_line(result.seniority.primary_finding), "", "Education"])
    lines.extend(_format_finding_line(finding) for finding in result.education.findings)
    lines.extend(
        [
            "",
            "Core Skills",
            f"✅ Aligned: {', '.join(result.core_skills.aligned_skills) if result.core_skills.aligned_skills else 'None'}",
            (
                "❌ Missing but evidenced in your profile: "
                f"{', '.join(result.core_skills.evidenced_elsewhere_skills) if result.core_skills.evidenced_elsewhere_skills else 'None'}"
            ),
            (
                "❌ Genuine gaps: "
                f"{', '.join(result.core_skills.genuine_gaps) if result.core_skills.genuine_gaps else 'None'}"
            ),
            "",
            "Projects",
        ]
    )
    lines.extend(_format_finding_line(finding) for finding in result.projects.findings)
    return "\n".join(lines)


def _count_citations(result: FitAnalysisResult) -> int:
    citations: list[EvidenceCitation] = []
    groups = [
        result.relevant_experience.findings,
        [result.seniority.primary_finding],
        result.education.findings,
        result.core_skills.findings,
        result.projects.findings,
    ]
    for group in groups:
        for finding in group:
            citations.extend(finding.citations)
    return len(_dedupe_citations(citations))


def fit_analysis_tool(
    job: Job,
    job_score: JobScore,
    bundle: CandidateBundle,
    memory: CandidateMemory,
    resume_tex_path: Path,
) -> FitAnalysisResult:
    """Produce evidence-grounded fit analysis for one Top 3 job."""
    if job.job_id != job_score.job_id:
        raise FitAnalysisError(
            f"job.job_id {job.job_id!r} does not match job_score.job_id {job_score.job_id!r}"
        )

    visibility = parse_resume_visibility(resume_tex_path)
    has_vector_search = _profile_has_vector_search(bundle)
    skill_universe = build_candidate_skill_universe(bundle, memory)

    relevant_experience = _analyze_relevant_experience(job, bundle, visibility)
    seniority = _analyze_seniority(job, job_score, bundle)
    education = _analyze_education(job, bundle)
    core_skills = _analyze_core_skills(
        job,
        job_score,
        bundle,
        memory,
        visibility,
        skill_universe,
        has_vector_search=has_vector_search,
    )
    projects = _analyze_projects(job, bundle)

    preliminary = FitAnalysisResult(
        job_id=job.job_id,
        title=job.title,
        company=job.company,
        score_rank=job_score.rank,
        final_score=job_score.final_score,
        relevant_experience=relevant_experience,
        seniority=seniority,
        education=education,
        core_skills=core_skills,
        projects=projects,
        tailoring_actions=[],
        formatted_text="",
        evidence_citation_count=0,
        genuine_gap_count=len(core_skills.genuine_gaps),
        evidenced_missing_skill_count=len(core_skills.evidenced_elsewhere_skills),
        project_swap_recommended=projects.swap_suggestion is not None,
    )
    tailoring_actions = _build_tailoring_actions(preliminary, bundle, job)
    preliminary = preliminary.model_copy(update={"tailoring_actions": tailoring_actions})
    formatted_text = _build_formatted_text(preliminary)
    citation_count = _count_citations(preliminary)

    return preliminary.model_copy(
        update={
            "formatted_text": formatted_text,
            "evidence_citation_count": citation_count,
        }
    )


__all__ = [
    "AnalysisFinding",
    "CitationSourceType",
    "CoreSkillsAnalysis",
    "EducationAnalysis",
    "EvidenceCitation",
    "FindingStatus",
    "FitAnalysisError",
    "FitAnalysisResult",
    "ProjectComparison",
    "ProjectSwapSuggestion",
    "ProjectsAnalysis",
    "RelevantExperienceAnalysis",
    "ResumeVisibility",
    "SeniorityAnalysis",
    "TailoringAction",
    "TailoringActionType",
    "fit_analysis_tool",
    "parse_resume_visibility",
]
