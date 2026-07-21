"""Typed job posting model and deterministic parsing helpers."""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class WorkMode(StrEnum):
    """Deterministic work-mode classification derived from location text."""

    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    MIXED = "mixed"
    UNKNOWN = "unknown"


ExperienceParseStatus = Literal["exact", "approximate", "ambiguous", "unspecified"]

_YEAR_PATTERN = re.compile(r"(\d+)\+?\s*years?", re.IGNORECASE)
_APPROX_PATTERN = re.compile(r"approximately\s+(\d+)", re.IGNORECASE)
_UNSPECIFIED_PATTERNS = (
    re.compile(r"staff[- ]level", re.IGNORECASE),
    re.compile(r"exact minimum (?:years )?not (?:explicitly )?stated", re.IGNORECASE),
    re.compile(r"exact minimum years should be confirmed", re.IGNORECASE),
)
_MULTI_REQUIREMENT_PATTERN = re.compile(
    r"\d+\+?\s*years?[^.;]{0,80}\band\b[^.;]{0,80}\d+\+?\s*years?",
    re.IGNORECASE,
)
_ALTERNATIVE_PATTERN = re.compile(
    r"\d+\+?\s*years?[^.;]{0,120}\bor\s+\d+\+?\s*years?",
    re.IGNORECASE,
)


def derive_job_id(url: str) -> str:
    """Return a stable SHA-256 identifier for a canonical job URL."""
    canonical = url.strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_required_skills(raw: str) -> list[str]:
    """Parse semicolon-separated skills while preserving meaningful phrases."""
    seen: set[str] = set()
    parsed: list[str] = []
    for part in raw.split(";"):
        skill = part.strip()
        if not skill:
            continue
        key = skill.casefold()
        if key in seen:
            continue
        seen.add(key)
        parsed.append(skill)
    return parsed


def _extract_year_values(text: str) -> list[int]:
    """Extract explicit numeric year values in document order."""
    values: list[int] = []
    seen_positions: set[tuple[int, int]] = set()
    for match in _YEAR_PATTERN.finditer(text):
        span = match.span()
        if span in seen_positions:
            continue
        seen_positions.add(span)
        values.append(int(match.group(1)))
    for match in _APPROX_PATTERN.finditer(text):
        value = int(match.group(1))
        if value not in values:
            values.append(value)
    return values


def _is_unspecified_wording(text: str) -> bool:
    """Detect staff-level or explicit no-minimum wording."""
    return any(pattern.search(text) for pattern in _UNSPECIFIED_PATTERNS)


def parse_experience_requirement(
    raw: str,
) -> tuple[ExperienceParseStatus, int | None, list[int]]:
    """Parse experience requirements without inventing unsupported minimums."""
    text = raw.strip()
    lower = text.lower()
    year_values = _extract_year_values(text)

    if not text:
        return "unspecified", None, []

    if _is_unspecified_wording(text):
        return "unspecified", None, year_values

    if _APPROX_PATTERN.search(text):
        approx_values = [int(match.group(1)) for match in _APPROX_PATTERN.finditer(text)]
        if len(approx_values) == 1 and len(year_values) <= 1:
            return "approximate", approx_values[0], year_values or approx_values
        return "ambiguous", None, year_values

    if not year_values:
        return "unspecified", None, []

    if _ALTERNATIVE_PATTERN.search(text):
        return "ambiguous", None, year_values

    if len(year_values) > 1 and (
        ";" in text or "depending on background" in lower
    ):
        return "ambiguous", None, year_values

    if _MULTI_REQUIREMENT_PATTERN.search(text):
        return "exact", max(year_values), year_values

    if len(year_values) > 1:
        return "ambiguous", None, year_values

    return "exact", year_values[0], year_values


def parse_work_mode(location_raw: str) -> WorkMode:
    """Derive a conservative work mode from raw location text."""
    text = location_raw.strip()
    lower = text.lower()
    has_remote = "remote" in lower
    has_hybrid = "hybrid" in lower
    has_onsite = any(
        marker in lower
        for marker in ("in person", "in-office", "in office", "onsite", "on-site")
    )

    if has_remote and has_hybrid:
        return WorkMode.MIXED
    if has_remote and has_onsite:
        return WorkMode.MIXED
    if has_hybrid and "remote" in lower:
        return WorkMode.MIXED
    if has_remote:
        return WorkMode.REMOTE
    if has_hybrid:
        return WorkMode.HYBRID
    if has_onsite:
        return WorkMode.ONSITE
    if re.search(r",\s*[A-Za-z]{2}\b", text):
        return WorkMode.ONSITE
    return WorkMode.UNKNOWN


class Job(BaseModel):
    """Normalized job posting loaded from the assignment CSV dataset."""

    job_id: str
    title: str
    company: str
    industry_domain: str
    location_raw: str
    work_mode: WorkMode
    required_skills_raw: str
    required_skills: list[str]
    experience_requirement_raw: str
    minimum_years: int | None
    experience_parse_status: ExperienceParseStatus
    experience_year_values: list[int]
    job_description: str
    company_details: str
    url: str
    source_row: int = Field(ge=1, description="1-based CSV data row number")
