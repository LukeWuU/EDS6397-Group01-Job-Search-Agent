"""Deterministic job scoring tool (assignment callable tool #2)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from src.models.bundle import CandidateBundle
from src.models.job import Job, WorkMode
from src.models.memory import CandidateMemory, MemoryFact
from src.tools.filtering import normalize_location_text, normalize_title

SKILLS_WEIGHT = 0.50
EXPERIENCE_WEIGHT = 0.25
INDUSTRY_DOMAIN_WEIGHT = 0.15
LOCATION_WEIGHT = 0.10
NEUTRAL_SKILLS_SCORE = 50.0
APPROXIMATE_EXPERIENCE_DEDUCTION = 5.0
OVERQUALIFICATION_YEARS_BUFFER = 8
OVERQUALIFICATION_SCORE = 85.0

FORMULA_DESCRIPTION = (
    "final_score = skills_score * 0.50 + experience_score * 0.25 + "
    "industry_domain_score * 0.15 + location_score * 0.10. "
    "All numerical scores are computed only by deterministic Python using the "
    "entire portfolio, master skills, evidence registry, and memory. "
    "No LLM generates or adjusts scores."
)

_NONWORD_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_STAFF_TITLE_KEYWORDS = ("staff", "principal", "lead", "manager", "director", "head")

_SKILL_EQUIVALENCE: dict[str, tuple[str, ...]] = {
    "machine learning": (
        "machine learning",
        "scikit learn",
        "pytorch",
        "tensorflow",
        "xgboost",
        "pandas",
        "numpy",
        "model evaluation",
    ),
    "generative ai": (
        "generative ai",
        "retrieval augmented generation",
        "embeddings",
        "prompt engineering",
        "large language models",
        "transformers",
    ),
    "retrieval augmented generation": (
        "retrieval augmented generation",
        "embeddings",
        "vector search",
        "prompt engineering",
    ),
    "natural language processing": ("natural language processing", "nlp", "transformers"),
    "mlops": ("mlops", "mlflow", "docker", "ci cd", "model monitoring"),
    "rest api": ("rest api", "fastapi"),
    "postgresql": ("postgresql", "postgres", "sql"),
}

_DOMAIN_CANONICAL_GROUPS: dict[str, tuple[str, ...]] = {
    "healthcare": ("healthcare", "health care", "clinical", "clinical operations"),
    "financial services": (
        "financial services",
        "finance",
        "fintech",
        "financial technology",
        "accounting",
        "accounting technology",
        "financial crime",
    ),
    "retail and supply chain": (
        "retail",
        "retail and supply chain",
        "ecommerce",
        "e commerce",
        "digital commerce",
        "specialty retail",
    ),
    "industrial systems": (
        "industrial",
        "industrial systems",
        "industrial quality",
        "manufacturing",
    ),
    "enterprise knowledge systems": (
        "enterprise knowledge systems",
        "enterprise technology",
        "business software",
        "knowledge systems",
    ),
    "energy": ("energy", "utility", "utility analytics", "commodities"),
    "media": ("media", "digital media", "entertainment", "news"),
    "cloud computing": ("cloud", "cloud computing", "saas"),
    "government": ("government", "public sector", "defense", "federal"),
    "real estate": ("real estate", "property technology", "proptech", "reit"),
    "cybersecurity": ("cybersecurity", "security", "public safety"),
    "developer productivity": ("developer productivity", "machine learning operations"),
}

_DOMAIN_RELATED_GROUPS: dict[str, tuple[str, ...]] = {
    "healthcare": ("enterprise knowledge systems", "clinical operations"),
    "financial services": ("enterprise knowledge systems", "accounting technology"),
    "retail and supply chain": ("enterprise knowledge systems", "digital commerce"),
    "industrial systems": ("manufacturing", "energy", "developer productivity"),
    "enterprise knowledge systems": ("cloud computing", "saas", "business software"),
    "energy": ("industrial systems", "financial services"),
    "government": ("cybersecurity", "enterprise knowledge systems"),
    "real estate": ("enterprise knowledge systems", "cloud computing"),
}


class ScoreWeights(BaseModel):
    """Fixed component weights for the scoring formula."""

    skills: float = SKILLS_WEIGHT
    experience: float = EXPERIENCE_WEIGHT
    industry_domain: float = INDUSTRY_DOMAIN_WEIGHT
    location: float = LOCATION_WEIGHT

    @model_validator(mode="after")
    def validate_sum(self) -> "ScoreWeights":
        """Ensure component weights sum to exactly 1.0."""
        total = self.skills + self.experience + self.industry_domain + self.location
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Score weights must sum to 1.0, got {total}")
        return self


class SkillSourceReference(BaseModel):
    """Evidence source supporting a candidate skill match."""

    source_type: Literal["master_skill", "project", "evidence", "memory"]
    source_id: str
    display_skill: str


class SkillMatchEvidence(BaseModel):
    """Skill match details for one required job skill."""

    job_skill: str
    matched: bool
    canonical_job_skill: str
    canonical_candidate_skill: str | None = None
    evidence_sources: list[SkillSourceReference] = Field(default_factory=list)


class ScoreBreakdown(BaseModel):
    """Component and weighted scores for one job."""

    skills_score: float
    skills_weighted: float
    experience_score: float
    experience_weighted: float
    industry_domain_score: float
    industry_domain_weighted: float
    location_score: float
    location_weighted: float


class JobScore(BaseModel):
    """Deterministic score for one job."""

    rank: int
    job_id: str
    title: str
    company: str
    final_score: float
    breakdown: ScoreBreakdown
    matched_required_skills: list[str]
    unmatched_required_skills: list[str]
    matched_skill_evidence: list[SkillMatchEvidence]
    candidate_years: int
    required_minimum_years: int | None
    experience_parse_status: str
    domain_matches: list[str]
    location_explanation: str


class ScoringResult(BaseModel):
    """Complete deterministic scoring output."""

    total_scored: int
    ranked_jobs: list[JobScore]
    top_3: list[JobScore]
    weights: ScoreWeights
    formula_description: str
    candidate_skill_count: int
    memory_fact_count: int
    warning: str | None = None


class CandidateSkillUniverse(BaseModel):
    """Canonical candidate skills with deduplicated source references."""

    canonical_to_sources: dict[str, list[SkillSourceReference]]
    display_by_canonical: dict[str, str]

    @property
    def canonical_skills(self) -> set[str]:
        """Return the set of canonical candidate skills."""
        return set(self.canonical_to_sources)


def _round_score(value: float) -> float:
    """Round a score to two decimal places."""
    return round(value, 2)


def _apply_alias_patterns(text: str, has_vector_search: bool) -> str:
    """Apply deterministic skill alias normalization."""
    replacements = [
        (r"\bgen\s+ai\b", "generative ai"),
        (r"\bgenai\b", "generative ai"),
        (r"\bmachine\s*learning\b", "machine learning"),
        (r"\bmachine\s*-\s*learning\b", "machine learning"),
        (r"\bml\b", "machine learning"),
        (r"\brag\b", "retrieval augmented generation"),
        (r"\bllms\b", "large language models"),
        (r"\bllm\b", "large language models"),
        (r"\bnlp\b", "natural language processing"),
        (r"\bcv\b", "computer vision"),
        (r"\bscikit\s*-\s*learn\b", "scikit learn"),
        (r"\bsklearn\b", "scikit learn"),
        (r"\bml\s*ops\b", "mlops"),
        (r"\bmlops\b", "mlops"),
        (r"\brest\s+apis\b", "rest api"),
        (r"\brest\s+api\b", "rest api"),
        (r"\bci\s*/\s*cd\b", "ci cd"),
        (r"\bpostgres\b", "postgresql"),
        (r"\bmodel\s+monitoring\b", "model monitoring"),
    ]
    normalized = text
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    if has_vector_search:
        normalized = re.sub(r"\bvector\s+databases\b", "vector search", normalized)
        normalized = re.sub(r"\bvector\s+db\b", "vector search", normalized)
    return normalized


def normalize_skill(skill: str, *, has_vector_search: bool = False) -> str:
    """Normalize a skill phrase to a canonical deterministic form."""
    text = skill.casefold().strip()
    text = _NONWORD_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    text = _apply_alias_patterns(text, has_vector_search)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _add_skill_source(
    universe: dict[str, list[SkillSourceReference]],
    display_by_canonical: dict[str, str],
    skill: str,
    source: SkillSourceReference,
    *,
    has_vector_search: bool,
) -> None:
    """Insert a skill and source reference into the universe."""
    canonical = normalize_skill(skill, has_vector_search=has_vector_search)
    if not canonical:
        return
    display_by_canonical.setdefault(canonical, skill.strip())
    existing = universe.setdefault(canonical, [])
    source_key = (source.source_type, source.source_id, source.display_skill.casefold())
    if not any(
        (item.source_type, item.source_id, item.display_skill.casefold()) == source_key
        for item in existing
    ):
        existing.append(source)


def _profile_has_vector_search(bundle: CandidateBundle) -> bool:
    """Return True when profile evidence includes vector search."""
    terms: list[str] = list(bundle.all_master_skills())
    for project in bundle.all_projects():
        terms.extend(project.technology_stack)
        terms.extend(project.skills_demonstrated)
    for record in bundle.evidence.evidence_records:
        terms.extend(record.supported_skills)
    return any("vector search" in term.casefold() for term in terms)


def build_candidate_skill_universe(
    bundle: CandidateBundle,
    memory: CandidateMemory,
) -> CandidateSkillUniverse:
    """Build the whole-profile candidate skill universe with source references."""
    universe: dict[str, list[SkillSourceReference]] = {}
    display_by_canonical: dict[str, str] = {}
    has_vector_search = _profile_has_vector_search(bundle)

    for category_name, skills in (
        ("languages", bundle.profile.master_skills.languages),
        ("ml_and_data", bundle.profile.master_skills.ml_and_data),
        ("generative_ai", bundle.profile.master_skills.generative_ai),
        ("cloud_and_mlops", bundle.profile.master_skills.cloud_and_mlops),
        ("systems_and_tools", bundle.profile.master_skills.systems_and_tools),
    ):
        for skill in skills:
            _add_skill_source(
                universe,
                display_by_canonical,
                skill,
                SkillSourceReference(
                    source_type="master_skill",
                    source_id=f"master_skills.{category_name}",
                    display_skill=skill,
                ),
                has_vector_search=has_vector_search,
            )

    for project in bundle.all_projects():
        for skill in [*project.technology_stack, *project.skills_demonstrated]:
            _add_skill_source(
                universe,
                display_by_canonical,
                skill,
                SkillSourceReference(
                    source_type="project",
                    source_id=project.project_id,
                    display_skill=skill,
                ),
                has_vector_search=has_vector_search,
            )

    for record in bundle.evidence.evidence_records:
        for skill in record.supported_skills:
            _add_skill_source(
                universe,
                display_by_canonical,
                skill,
                SkillSourceReference(
                    source_type="evidence",
                    source_id=record.evidence_id,
                    display_skill=skill,
                ),
                has_vector_search=has_vector_search,
            )

    for fact in memory.facts:
        if fact.fact_type != "skill":
            continue
        skill_values = list(fact.skill_tags)
        if isinstance(fact.normalized_value, str):
            skill_values.append(fact.normalized_value)
        elif isinstance(fact.normalized_value, list):
            skill_values.extend(str(item) for item in fact.normalized_value)
        for skill in skill_values:
            _add_skill_source(
                universe,
                display_by_canonical,
                skill,
                SkillSourceReference(
                    source_type="memory",
                    source_id=fact.fact_id,
                    display_skill=skill,
                ),
                has_vector_search=has_vector_search,
            )

    return CandidateSkillUniverse(
        canonical_to_sources=universe,
        display_by_canonical=display_by_canonical,
    )


def _equivalent_canonical_skills(canonical_job: str) -> tuple[str, ...]:
    """Return deterministic equivalent canonical skill terms for matching."""
    equivalents = {canonical_job}
    equivalents.update(_SKILL_EQUIVALENCE.get(canonical_job, ()))
    return tuple(sorted(equivalents))


def _skill_match(
    job_skill: str,
    skill_universe: CandidateSkillUniverse,
    *,
    has_vector_search: bool,
) -> tuple[bool, str | None, list[SkillSourceReference]]:
    """Match one required job skill against the candidate skill universe."""
    canonical_job = normalize_skill(job_skill, has_vector_search=has_vector_search)
    if not canonical_job:
        return False, None, []

    search_terms = _equivalent_canonical_skills(canonical_job)
    matched_sources: list[SkillSourceReference] = []
    matched_canonical: str | None = None

    for term in search_terms:
        if term in skill_universe.canonical_to_sources:
            matched_canonical = matched_canonical or term
            matched_sources.extend(skill_universe.canonical_to_sources[term])

    if matched_sources:
        deduped: list[SkillSourceReference] = []
        seen: set[tuple[str, str, str]] = set()
        for source in matched_sources:
            key = (source.source_type, source.source_id, source.display_skill.casefold())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
        return True, matched_canonical, deduped

    for term in search_terms:
        for candidate_canonical in sorted(skill_universe.canonical_to_sources):
            if term == candidate_canonical:
                return True, candidate_canonical, list(
                    skill_universe.canonical_to_sources[candidate_canonical]
                )
            if len(term) >= 4 and len(candidate_canonical) >= 4:
                if term in candidate_canonical or candidate_canonical in term:
                    return True, candidate_canonical, list(
                        skill_universe.canonical_to_sources[candidate_canonical]
                    )
    return False, None, []


def _score_skills(
    job: Job,
    skill_universe: CandidateSkillUniverse,
    *,
    has_vector_search: bool,
) -> tuple[float, list[str], list[str], list[SkillMatchEvidence]]:
    """Compute the skills component score."""
    if not job.required_skills:
        return NEUTRAL_SKILLS_SCORE, [], [], []

    matched: list[str] = []
    unmatched: list[str] = []
    evidence: list[SkillMatchEvidence] = []

    for job_skill in job.required_skills:
        is_match, candidate_canonical, sources = _skill_match(
            job_skill,
            skill_universe,
            has_vector_search=has_vector_search,
        )
        evidence.append(
            SkillMatchEvidence(
                job_skill=job_skill,
                matched=is_match,
                canonical_job_skill=normalize_skill(
                    job_skill,
                    has_vector_search=has_vector_search,
                ),
                canonical_candidate_skill=candidate_canonical,
                evidence_sources=sources,
            )
        )
        if is_match:
            matched.append(job_skill)
        else:
            unmatched.append(job_skill)

    score = _round_score(100.0 * len(matched) / len(job.required_skills))
    return score, matched, unmatched, evidence


def _title_experience_band(title: str) -> tuple[str, int, int]:
    """Return a deterministic seniority band for title-based experience scoring."""
    normalized = normalize_title(title)
    if re.search(r"\b(junior|entry|level 1|level i)\b", normalized):
        return "junior", 0, 2
    for keyword in _STAFF_TITLE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", normalized):
            return "staff", 6, 99
    if re.search(r"\bsenior\b", normalized):
        return "senior", 4, 99
    return "mid", 2, 4


def _score_title_based_experience(candidate_years: int, title: str) -> float:
    """Score experience when no reliable explicit minimum exists."""
    band, low, high = _title_experience_band(title)
    if band == "junior":
        if candidate_years <= high:
            return 100.0
        if candidate_years <= high + 2:
            return _round_score(max(70.0, 100.0 - 10.0 * (candidate_years - high)))
        return _round_score(max(40.0, 100.0 - 15.0 * (candidate_years - high)))
    if band == "mid":
        if low <= candidate_years <= high:
            return 100.0
        if candidate_years < low:
            return _round_score(max(0.0, 100.0 * candidate_years / low))
        if candidate_years <= high + 4:
            return 85.0
        return 70.0
    if band == "senior":
        if candidate_years >= 4:
            return 100.0
        return _round_score(max(0.0, 100.0 * candidate_years / 4.0))
    if candidate_years >= 6:
        return 100.0
    return _round_score(max(0.0, 100.0 * candidate_years / 6.0))


def _score_experience(job: Job, candidate_years: int) -> float:
    """Compute the experience component score."""
    minimum = job.minimum_years
    if minimum is not None and job.experience_parse_status in {"exact", "approximate"}:
        if candidate_years < minimum:
            score = max(0.0, 100.0 * candidate_years / minimum)
        elif candidate_years == minimum:
            score = 100.0
        elif candidate_years > minimum + OVERQUALIFICATION_YEARS_BUFFER:
            score = OVERQUALIFICATION_SCORE
        else:
            score = 100.0
        if job.experience_parse_status == "approximate":
            score = max(0.0, score - APPROXIMATE_EXPERIENCE_DEDUCTION)
        return _round_score(score)

    return _round_score(_score_title_based_experience(candidate_years, job.title))


def normalize_domain_term(term: str) -> str:
    """Normalize a domain or industry term."""
    text = term.casefold().strip()
    text = _NONWORD_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _canonicalize_domain(term: str) -> str | None:
    """Map a domain term to a canonical group label when possible."""
    normalized = normalize_domain_term(term)
    if not normalized:
        return None
    for canonical, aliases in _DOMAIN_CANONICAL_GROUPS.items():
        if normalized == canonical or normalized in aliases:
            return canonical
        if any(alias in normalized for alias in aliases):
            return canonical
    return normalized


def build_candidate_domain_universe(bundle: CandidateBundle) -> set[str]:
    """Build the candidate domain universe from preferences and portfolio."""
    domains: set[str] = set()
    for domain in bundle.profile.preferences.target_domains:
        canonical = _canonicalize_domain(domain)
        if canonical:
            domains.add(canonical)
    for project in bundle.all_projects():
        for term in (project.domain, project.industry):
            canonical = _canonicalize_domain(term)
            if canonical:
                domains.add(canonical)
    return domains


def _domain_relationship_score(job_domain: str, candidate_domains: set[str]) -> tuple[float, list[str]]:
    """Score industry/domain alignment and return matched domains."""
    job_canonical = _canonicalize_domain(job_domain)
    if job_canonical is None:
        return 40.0, []

    matched = sorted(domain for domain in candidate_domains if domain == job_canonical)
    if matched:
        return 100.0, matched

    related_matches: set[str] = set()
    for candidate_domain in candidate_domains:
        related = _DOMAIN_RELATED_GROUPS.get(candidate_domain, ())
        if job_canonical in related or candidate_domain in _DOMAIN_RELATED_GROUPS.get(job_canonical, ()):
            related_matches.add(candidate_domain)
        job_aliases = _DOMAIN_CANONICAL_GROUPS.get(job_canonical, ())
        candidate_aliases = _DOMAIN_CANONICAL_GROUPS.get(candidate_domain, ())
        if any(alias in normalize_domain_term(job_domain) for alias in candidate_aliases):
            related_matches.add(candidate_domain)
        if any(alias in candidate_aliases for alias in job_aliases):
            related_matches.add(candidate_domain)

    if related_matches:
        return 80.0, sorted(related_matches)

    general_ai_terms = (
        "ai",
        "machine learning",
        "software",
        "technology",
        "saas",
        "cloud",
        "data",
    )
    if any(term in normalize_domain_term(job_domain) for term in general_ai_terms):
        return 60.0, sorted(candidate_domains)[:3]

    return 40.0, []


def _extract_physical_locations(preferred_locations: Sequence[str]) -> list[str]:
    """Extract preferred physical city tokens."""
    cities: list[str] = []
    for location in preferred_locations:
        if "remote" in location.casefold():
            continue
        normalized = normalize_location_text(location)
        city = normalized.split(",")[0].strip() if "," in normalized else normalized.split()[0]
        if city:
            cities.append(city)
    return cities


def _allows_remote_work(remote_preference: str) -> bool:
    """Return True when remote work is allowed."""
    lower = remote_preference.casefold()
    if any(phrase in lower for phrase in ("remote only", "remote-only", "only remote")):
        return True
    if any(phrase in lower for phrase in ("no remote", "not remote", "onsite only", "on-site only")):
        return False
    return "remote" in lower or "hybrid" in lower


def _location_has_preferred_city(location_raw: str, preferred_cities: Sequence[str]) -> bool:
    """Return True when a preferred city appears in the job location."""
    normalized = normalize_location_text(location_raw)
    return any(city in normalized for city in preferred_cities)


def _location_has_explicit_remote_option(location_raw: str) -> bool:
    """Return True when the posting explicitly mentions remote work."""
    return "remote" in location_raw.casefold()


def _score_location(job: Job, bundle: CandidateBundle) -> tuple[float, str]:
    """Compute the location component score."""
    preferences = bundle.profile.preferences
    preferred_cities = _extract_physical_locations(preferences.preferred_locations)
    allows_remote = _allows_remote_work(preferences.remote_preference)
    has_preferred_city = _location_has_preferred_city(job.location_raw, preferred_cities)
    has_remote_option = _location_has_explicit_remote_option(job.location_raw)

    if job.work_mode == WorkMode.REMOTE:
        if allows_remote:
            return 100.0, "Remote job aligns with remote-eligible preferences."
        return 0.0, "Remote job conflicts with onsite-only preferences."

    if job.work_mode == WorkMode.ONSITE:
        if has_preferred_city:
            return 100.0, "Onsite job matches a preferred physical location."
        return 0.0, "Onsite job is outside preferred physical locations."

    if job.work_mode == WorkMode.HYBRID:
        if has_preferred_city:
            return 95.0, "Hybrid job matches a preferred physical location."
        return 0.0, "Hybrid job lacks a preferred physical location."

    if job.work_mode == WorkMode.MIXED:
        if has_remote_option:
            return 95.0, "Mixed posting includes an explicit remote option."
        if has_preferred_city:
            return 90.0, "Mixed posting includes a preferred physical location."
        return 0.0, "Mixed posting lacks remote option and preferred location."

    if has_preferred_city or has_remote_option:
        return 65.0, "Unknown work mode with acceptable location wording."
    return 65.0, "Unknown work mode; location alignment is uncertain."


def _build_breakdown(
    skills_score: float,
    experience_score: float,
    industry_domain_score: float,
    location_score: float,
    weights: ScoreWeights,
) -> tuple[ScoreBreakdown, float]:
    """Build weighted breakdown and final score."""
    skills_weighted = _round_score(skills_score * weights.skills)
    experience_weighted = _round_score(experience_score * weights.experience)
    industry_weighted = _round_score(industry_domain_score * weights.industry_domain)
    location_weighted = _round_score(location_score * weights.location)
    final_score = _round_score(
        skills_weighted + experience_weighted + industry_weighted + location_weighted
    )
    breakdown = ScoreBreakdown(
        skills_score=_round_score(skills_score),
        skills_weighted=skills_weighted,
        experience_score=_round_score(experience_score),
        experience_weighted=experience_weighted,
        industry_domain_score=_round_score(industry_domain_score),
        industry_domain_weighted=industry_weighted,
        location_score=_round_score(location_score),
        location_weighted=location_weighted,
    )
    return breakdown, final_score


def _rank_jobs(scored_jobs: list[JobScore], input_index_by_job_id: dict[str, int]) -> list[JobScore]:
    """Rank jobs using deterministic tie-breaking rules."""
    ordered = sorted(
        scored_jobs,
        key=lambda item: (
            -item.final_score,
            -item.breakdown.skills_score,
            -item.breakdown.experience_score,
            -item.breakdown.industry_domain_score,
            input_index_by_job_id[item.job_id],
            item.job_id,
        ),
    )
    ranked: list[JobScore] = []
    for index, job_score in enumerate(ordered, start=1):
        ranked.append(job_score.model_copy(update={"rank": index}))
    return ranked


def _score_single_job(
    job: Job,
    bundle: CandidateBundle,
    skill_universe: CandidateSkillUniverse,
    candidate_domains: set[str],
    *,
    has_vector_search: bool,
) -> JobScore:
    """Score one job deterministically."""
    candidate_years = bundle.profile.preferences.years_of_experience
    skills_score, matched, unmatched, skill_evidence = _score_skills(
        job,
        skill_universe,
        has_vector_search=has_vector_search,
    )
    experience_score = _score_experience(job, candidate_years)
    industry_score, domain_matches = _domain_relationship_score(
        job.industry_domain,
        candidate_domains,
    )
    location_score, location_explanation = _score_location(job, bundle)
    weights = ScoreWeights()
    breakdown, final_score = _build_breakdown(
        skills_score,
        experience_score,
        industry_score,
        location_score,
        weights,
    )
    return JobScore(
        rank=0,
        job_id=job.job_id,
        title=job.title,
        company=job.company,
        final_score=final_score,
        breakdown=breakdown,
        matched_required_skills=matched,
        unmatched_required_skills=unmatched,
        matched_skill_evidence=skill_evidence,
        candidate_years=candidate_years,
        required_minimum_years=job.minimum_years,
        experience_parse_status=job.experience_parse_status,
        domain_matches=domain_matches,
        location_explanation=location_explanation,
    )


def scoring_tool(
    jobs: Sequence[Job],
    bundle: CandidateBundle,
    memory: CandidateMemory,
) -> ScoringResult:
    """Score jobs using deterministic Python logic over the whole candidate profile."""
    if not jobs:
        raise ValueError("scoring_tool requires at least one job")

    skill_universe = build_candidate_skill_universe(bundle, memory)
    has_vector_search = "vector search" in skill_universe.canonical_skills
    candidate_domains = build_candidate_domain_universe(bundle)
    weights = ScoreWeights()

    input_index_by_job_id = {job.job_id: index for index, job in enumerate(jobs)}
    scored_jobs = [
        _score_single_job(
            job,
            bundle,
            skill_universe,
            candidate_domains,
            has_vector_search=has_vector_search,
        )
        for job in jobs
    ]
    ranked_jobs = _rank_jobs(scored_jobs, input_index_by_job_id)
    top_3 = ranked_jobs[:3]
    warning = None
    if len(ranked_jobs) < 3:
        warning = (
            f"Only {len(ranked_jobs)} job(s) were provided; top_3 contains all scored jobs."
        )

    return ScoringResult(
        total_scored=len(ranked_jobs),
        ranked_jobs=ranked_jobs,
        top_3=top_3,
        weights=weights,
        formula_description=FORMULA_DESCRIPTION,
        candidate_skill_count=len(skill_universe.canonical_skills),
        memory_fact_count=len(memory.facts),
        warning=warning,
    )


__all__ = [
    "CandidateSkillUniverse",
    "FORMULA_DESCRIPTION",
    "JobScore",
    "ScoreBreakdown",
    "ScoreWeights",
    "ScoringResult",
    "SkillMatchEvidence",
    "SkillSourceReference",
    "build_candidate_domain_universe",
    "build_candidate_skill_universe",
    "normalize_domain_term",
    "normalize_skill",
    "scoring_tool",
]
