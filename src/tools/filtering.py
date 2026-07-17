"""Deterministic job filtering tool (assignment callable tool #1)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.models.candidate import CandidateProfile, Preferences
from src.models.job import Job, WorkMode


class FilterReasonCode(StrEnum):
    """Deterministic rejection reason codes."""

    EXCLUDED_COMPANY = "excluded_company"
    TITLE_MISMATCH = "title_mismatch"
    EXPERIENCE_MISMATCH = "experience_mismatch"
    SENIORITY_MISMATCH = "seniority_mismatch"
    LOCATION_MISMATCH = "location_mismatch"
    REMOTE_ONLY_MISMATCH = "remote_only_mismatch"


class FilterWarningCode(StrEnum):
    """Non-rejecting filter warnings."""

    AMBIGUOUS_EXPERIENCE = "ambiguous_experience"
    UNSPECIFIED_EXPERIENCE = "unspecified_experience"
    UNKNOWN_WORK_MODE = "unknown_work_mode"
    UNCLEAR_LOCATION = "unclear_location"


class FilterReason(BaseModel):
    """Structured rejection reason for a filtered job."""

    code: FilterReasonCode
    message: str
    field: str
    observed_value: str
    expected_value: str


class FilterWarning(BaseModel):
    """Structured warning for an accepted or rejected job."""

    code: FilterWarningCode
    message: str
    field: str
    observed_value: str


class JobFilterDecision(BaseModel):
    """Filtering decision for a single job."""

    job_id: str
    title: str
    company: str
    accepted: bool
    rejection_reasons: list[FilterReason] = Field(default_factory=list)
    warnings: list[FilterWarning] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_reasons(self) -> "JobFilterDecision":
        """Enforce accepted/rejected reason invariants."""
        if self.accepted and self.rejection_reasons:
            raise ValueError("accepted jobs must not contain rejection reasons")
        if not self.accepted and not self.rejection_reasons:
            raise ValueError("rejected jobs must contain at least one rejection reason")
        return self


class FilteringResult(BaseModel):
    """Complete filtering output for a job batch."""

    total_jobs: int
    accepted_count: int
    rejected_count: int
    accepted_jobs: list[Job]
    rejected_jobs: list[Job]
    decisions: list[JobFilterDecision]
    policy_summary: dict[str, Any]


_NONWORD_PATTERN = re.compile(r"[^\w\s]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_SENIOR_TITLE_KEYWORDS = ("staff", "principal", "lead", "manager", "director", "head")
_SENIOR_FALLBACK_MIN_YEARS = 4
_STAFF_FALLBACK_MIN_YEARS = 6


def normalize_company(name: str) -> str:
    """Normalize company names for exact exclusion matching."""
    text = name.casefold().strip()
    text = _NONWORD_PATTERN.sub(" ", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_title(title: str) -> str:
    """Normalize job or target titles for deterministic compatibility checks."""
    text = title.casefold()
    text = _NONWORD_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    text = re.sub(r"\bgen\s+ai\b", "generative ai", text)
    text = re.sub(r"\bgenai\b", "generative ai", text)
    text = re.sub(r"\bml\b", "machine learning", text)
    text = re.sub(r"\bmlops\b", "mlops", text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_location_text(text: str) -> str:
    """Normalize location strings for deterministic city/state matching."""
    normalized = text.casefold()
    normalized = normalized.replace("texas", "tx")
    normalized = _NONWORD_PATTERN.sub(" ", normalized)
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def _is_remote_only_preference(remote_preference: str) -> bool:
    """Return True when the profile explicitly requires remote-only work."""
    lower = remote_preference.casefold()
    return any(
        phrase in lower
        for phrase in ("remote only", "remote-only", "only remote")
    )


def _allows_remote_work(preferences: Preferences) -> bool:
    """Return True when remote work is allowed by the profile."""
    lower = preferences.remote_preference.casefold()
    if _is_remote_only_preference(preferences.remote_preference):
        return True
    disallow_phrases = (
        "no remote",
        "not remote",
        "onsite only",
        "on-site only",
        "on site only",
    )
    if any(phrase in lower for phrase in disallow_phrases):
        return False
    return "remote" in lower or "hybrid" in lower


def _extract_physical_locations(preferred_locations: Sequence[str]) -> list[dict[str, str]]:
    """Extract preferred physical city/state tokens from profile locations."""
    physical: list[dict[str, str]] = []
    for location in preferred_locations:
        if "remote" in location.casefold():
            continue
        normalized = normalize_location_text(location)
        city = normalized.split(",")[0].strip() if "," in normalized else normalized.split()[0]
        physical.append({"city": city, "normalized": normalized})
    return physical


def _location_has_preferred_city(location_raw: str, physical_locations: Sequence[dict[str, str]]) -> bool:
    """Return True when a preferred physical city appears in the job location."""
    normalized_location = normalize_location_text(location_raw)
    for preferred in physical_locations:
        city = preferred["city"]
        if city and city in normalized_location:
            return True
    return False


def _location_has_explicit_remote_option(location_raw: str) -> bool:
    """Return True when the posting explicitly mentions remote work."""
    return "remote" in location_raw.casefold()


def _has_reliable_minimum_years(job: Job) -> bool:
    """Return True when the job has a reliable explicit minimum experience."""
    return (
        job.experience_parse_status in {"exact", "approximate"}
        and job.minimum_years is not None
    )


def _title_matches_targets(normalized_job_title: str, normalized_targets: Sequence[str]) -> bool:
    """Return True when a job title is compatible with any target title."""
    for target in normalized_targets:
        if target in normalized_job_title or normalized_job_title in target:
            return True
    return False


def _check_company_exclusion(
    job: Job,
    preferences: Preferences,
) -> FilterReason | None:
    """Reject jobs from excluded companies using normalized exact matching."""
    normalized_job_company = normalize_company(job.company)
    for excluded in preferences.excluded_companies:
        normalized_excluded = normalize_company(excluded)
        if normalized_job_company == normalized_excluded:
            return FilterReason(
                code=FilterReasonCode.EXCLUDED_COMPANY,
                message=(
                    f"Company {job.company!r} matches excluded preference {excluded!r}."
                ),
                field="company",
                observed_value=job.company,
                expected_value=f"not {excluded}",
            )
    return None


def _check_title_match(
    job: Job,
    normalized_targets: Sequence[str],
    raw_targets: Sequence[str],
) -> FilterReason | None:
    """Reject jobs whose titles do not match any target title."""
    normalized_job_title = normalize_title(job.title)
    if _title_matches_targets(normalized_job_title, normalized_targets):
        return None
    return FilterReason(
        code=FilterReasonCode.TITLE_MISMATCH,
        message=(
            f"Job title {job.title!r} does not match any target title after normalization."
        ),
        field="title",
        observed_value=normalized_job_title,
        expected_value=", ".join(raw_targets),
    )


def _check_experience_requirement(
    job: Job,
    candidate_years: int,
) -> tuple[FilterReason | None, list[FilterWarning]]:
    """Apply explicit experience requirements and experience warnings."""
    warnings: list[FilterWarning] = []

    if job.experience_parse_status == "ambiguous":
        warnings.append(
            FilterWarning(
                code=FilterWarningCode.AMBIGUOUS_EXPERIENCE,
                message=(
                    "Experience requirement contains ambiguous or multi-clause wording; "
                    "no numeric minimum was invented."
                ),
                field="experience_requirement_raw",
                observed_value=job.experience_requirement_raw,
            )
        )
    elif job.experience_parse_status == "unspecified":
        warnings.append(
            FilterWarning(
                code=FilterWarningCode.UNSPECIFIED_EXPERIENCE,
                message=(
                    "Experience requirement does not state a reliable explicit minimum."
                ),
                field="experience_requirement_raw",
                observed_value=job.experience_requirement_raw,
            )
        )
    elif job.experience_parse_status == "approximate" and job.minimum_years is not None:
        warnings.append(
            FilterWarning(
                code=FilterWarningCode.AMBIGUOUS_EXPERIENCE,
                message=(
                    "Experience requirement uses approximate wording; "
                    f"comparing against approximate minimum of {job.minimum_years} years."
                ),
                field="minimum_years",
                observed_value=str(job.minimum_years),
            )
        )

    if _has_reliable_minimum_years(job):
        assert job.minimum_years is not None
        if job.minimum_years > candidate_years:
            return (
                FilterReason(
                    code=FilterReasonCode.EXPERIENCE_MISMATCH,
                    message=(
                        f"Required minimum of {job.minimum_years} years exceeds "
                        f"candidate experience of {candidate_years} years."
                    ),
                    field="minimum_years",
                    observed_value=str(job.minimum_years),
                    expected_value=f"<={candidate_years}",
                ),
                warnings,
            )

    return None, warnings


def _check_seniority_fallback(
    job: Job,
    candidate_years: int,
) -> FilterReason | None:
    """Apply seniority fallback only when no reliable explicit minimum exists."""
    if _has_reliable_minimum_years(job):
        return None

    normalized_title = normalize_title(job.title)
    required_years: int | None = None
    trigger: str | None = None

    if re.search(r"\bsenior\b", normalized_title):
        required_years = _SENIOR_FALLBACK_MIN_YEARS
        trigger = "senior"
    else:
        for keyword in _SENIOR_TITLE_KEYWORDS:
            if re.search(rf"\b{re.escape(keyword)}\b", normalized_title):
                required_years = _STAFF_FALLBACK_MIN_YEARS
                trigger = keyword
                break

    if required_years is None or candidate_years >= required_years:
        return None

    return FilterReason(
        code=FilterReasonCode.SENIORITY_MISMATCH,
        message=(
            f"Title contains {trigger!r} seniority wording and requires at least "
            f"{required_years} years by fallback rule, but candidate has "
            f"{candidate_years} years."
        ),
        field="title",
        observed_value=job.title,
        expected_value=f">={required_years} years without explicit minimum",
    )


def _check_location_and_work_mode(
    job: Job,
    preferences: Preferences,
    physical_locations: Sequence[dict[str, str]],
) -> tuple[FilterReason | None, list[FilterWarning]]:
    """Apply location and work-mode rules."""
    warnings: list[FilterWarning] = []
    remote_only = _is_remote_only_preference(preferences.remote_preference)
    allows_remote = _allows_remote_work(preferences)
    has_preferred_city = _location_has_preferred_city(job.location_raw, physical_locations)
    has_remote_option = _location_has_explicit_remote_option(job.location_raw)

    if job.work_mode == WorkMode.UNKNOWN:
        warnings.append(
            FilterWarning(
                code=FilterWarningCode.UNKNOWN_WORK_MODE,
                message="Work mode could not be classified confidently from location text.",
                field="work_mode",
                observed_value=job.work_mode.value,
            )
        )
        if not has_preferred_city and not has_remote_option:
            warnings.append(
                FilterWarning(
                    code=FilterWarningCode.UNCLEAR_LOCATION,
                    message="Location text does not clearly match a preferred city or remote option.",
                    field="location_raw",
                    observed_value=job.location_raw,
                )
            )
        return None, warnings

    if remote_only:
        if job.work_mode == WorkMode.REMOTE:
            return None, warnings
        if job.work_mode == WorkMode.MIXED and has_remote_option:
            return None, warnings
        return (
            FilterReason(
                code=FilterReasonCode.REMOTE_ONLY_MISMATCH,
                message=(
                    "Profile requires remote-only work, but the posting is not a remote "
                    "or explicitly remote-eligible mixed arrangement."
                ),
                field="work_mode",
                observed_value=f"{job.work_mode.value}: {job.location_raw}",
                expected_value=preferences.remote_preference,
            ),
            warnings,
        )

    if job.work_mode == WorkMode.REMOTE:
        if allows_remote:
            return None, warnings
        return (
            FilterReason(
                code=FilterReasonCode.LOCATION_MISMATCH,
                message="Remote job rejected because profile preferences disallow remote work.",
                field="remote_preference",
                observed_value=preferences.remote_preference,
                expected_value="remote-eligible preference",
            ),
            warnings,
        )

    if job.work_mode == WorkMode.ONSITE:
        if has_preferred_city:
            return None, warnings
        return (
            FilterReason(
                code=FilterReasonCode.LOCATION_MISMATCH,
                message=(
                    "Onsite job location does not contain a preferred physical city/state."
                ),
                field="location_raw",
                observed_value=job.location_raw,
                expected_value=", ".join(preferences.preferred_locations),
            ),
            warnings,
        )

    if job.work_mode == WorkMode.HYBRID:
        if has_preferred_city or has_remote_option:
            return None, warnings
        return (
            FilterReason(
                code=FilterReasonCode.LOCATION_MISMATCH,
                message=(
                    "Hybrid job lacks both a preferred physical location and an explicit "
                    "remote option."
                ),
                field="location_raw",
                observed_value=job.location_raw,
                expected_value=", ".join(preferences.preferred_locations),
            ),
            warnings,
        )

    if job.work_mode == WorkMode.MIXED:
        if has_remote_option or has_preferred_city:
            return None, warnings
        return (
            FilterReason(
                code=FilterReasonCode.LOCATION_MISMATCH,
                message=(
                    "Mixed work-mode posting lacks a preferred physical location and "
                    "explicit remote option."
                ),
                field="location_raw",
                observed_value=job.location_raw,
                expected_value=", ".join(preferences.preferred_locations),
            ),
            warnings,
        )

    return None, warnings


def _build_policy_summary(profile: CandidateProfile) -> dict[str, Any]:
    """Describe the deterministic filtering policy applied."""
    preferences = profile.preferences
    return {
        "candidate_years": preferences.years_of_experience,
        "target_titles": list(preferences.target_job_titles),
        "preferred_locations": list(preferences.preferred_locations),
        "remote_preference": preferences.remote_preference,
        "onsite_policy": preferences.onsite_policy,
        "excluded_companies": list(preferences.excluded_companies),
        "experience_rule": (
            "Reject when a reliable exact or approximate minimum_years exceeds "
            "candidate years; ambiguous or unspecified requirements never invent "
            "a numeric minimum."
        ),
        "seniority_fallback": {
            "senior": _SENIOR_FALLBACK_MIN_YEARS,
            "staff_principal_lead_manager_director_head": _STAFF_FALLBACK_MIN_YEARS,
            "applies_only_without_reliable_explicit_minimum": True,
        },
        "location_rules": {
            "remote": "Accept when remote work is allowed by profile preferences.",
            "onsite": "Accept only when location contains a preferred physical city/state.",
            "hybrid": "Accept when preferred city/state is present or remote is explicit.",
            "mixed": "Accept when remote is explicit or preferred city/state is present.",
            "unknown": "Do not reject solely on unknown work mode; emit warnings.",
        },
        "remote_only_support": _is_remote_only_preference(preferences.remote_preference),
    }


def _evaluate_job(
    job: Job,
    profile: CandidateProfile,
    normalized_targets: Sequence[str],
    physical_locations: Sequence[dict[str, str]],
) -> JobFilterDecision:
    """Evaluate one job against all deterministic filtering rules."""
    preferences = profile.preferences
    rejection_reasons: list[FilterReason] = []
    warnings: list[FilterWarning] = []

    company_reason = _check_company_exclusion(job, preferences)
    if company_reason is not None:
        rejection_reasons.append(company_reason)

    title_reason = _check_title_match(job, normalized_targets, preferences.target_job_titles)
    if title_reason is not None:
        rejection_reasons.append(title_reason)

    experience_reason, experience_warnings = _check_experience_requirement(
        job,
        preferences.years_of_experience,
    )
    warnings.extend(experience_warnings)
    if experience_reason is not None:
        rejection_reasons.append(experience_reason)

    seniority_reason = _check_seniority_fallback(job, preferences.years_of_experience)
    if seniority_reason is not None:
        rejection_reasons.append(seniority_reason)

    location_reason, location_warnings = _check_location_and_work_mode(
        job,
        preferences,
        physical_locations,
    )
    warnings.extend(location_warnings)
    if location_reason is not None:
        rejection_reasons.append(location_reason)

    accepted = len(rejection_reasons) == 0
    return JobFilterDecision(
        job_id=job.job_id,
        title=job.title,
        company=job.company,
        accepted=accepted,
        rejection_reasons=rejection_reasons,
        warnings=warnings,
    )


def filtering_tool(
    jobs: Sequence[Job],
    profile: CandidateProfile,
) -> FilteringResult:
    """Filter jobs using explicit, deterministic candidate preference rules."""
    preferences = profile.preferences
    normalized_targets = [normalize_title(title) for title in preferences.target_job_titles]
    physical_locations = _extract_physical_locations(preferences.preferred_locations)

    decisions: list[JobFilterDecision] = []
    accepted_jobs: list[Job] = []
    rejected_jobs: list[Job] = []

    for job in jobs:
        decision = _evaluate_job(job, profile, normalized_targets, physical_locations)
        decisions.append(decision)
        if decision.accepted:
            accepted_jobs.append(job)
        else:
            rejected_jobs.append(job)

    return FilteringResult(
        total_jobs=len(jobs),
        accepted_count=len(accepted_jobs),
        rejected_count=len(rejected_jobs),
        accepted_jobs=accepted_jobs,
        rejected_jobs=rejected_jobs,
        decisions=decisions,
        policy_summary=_build_policy_summary(profile),
    )


__all__ = [
    "FilterReason",
    "FilterReasonCode",
    "FilterWarning",
    "FilterWarningCode",
    "FilteringResult",
    "JobFilterDecision",
    "filtering_tool",
    "normalize_company",
    "normalize_location_text",
    "normalize_title",
]
