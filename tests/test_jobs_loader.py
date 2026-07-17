"""Tests for deterministic job CSV loading and parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.job import WorkMode, derive_job_id, parse_experience_requirement
from src.services.jobs_loader import load_jobs

JOBS_CSV = ROOT / "data" / "AI_ML_Jobs_Dataset_20.csv"


def test_loads_exactly_twenty_jobs_in_csv_order() -> None:
    """All 20 dataset jobs load in original order."""
    jobs = load_jobs(JOBS_CSV)

    assert len(jobs) == 20
    assert jobs[0].company == "Quantifind"
    assert jobs[-1].company == "AMETEK, Inc."
    assert [job.source_row for job in jobs] == list(range(2, 22))


def test_job_ids_and_urls_are_unique_and_stable() -> None:
    """Every job has a deterministic unique ID derived from its URL."""
    jobs = load_jobs(JOBS_CSV)
    job_ids = [job.job_id for job in jobs]
    urls = [job.url for job in jobs]

    assert len(set(job_ids)) == 20
    assert len(set(urls)) == 20
    assert jobs[0].job_id == derive_job_id(jobs[0].url)


def test_required_skills_are_nonempty_and_parsed() -> None:
    """Semicolon-separated skills parse to non-empty deduplicated lists."""
    jobs = load_jobs(JOBS_CSV)

    for job in jobs:
        assert job.required_skills
        assert job.required_skills_raw
        assert len(job.required_skills) == len(
            {skill.casefold() for skill in job.required_skills}
        )


def test_bom_safe_csv_header_handling() -> None:
    """UTF-8 BOM CSV headers map to model fields correctly."""
    jobs = load_jobs(JOBS_CSV)
    assert jobs[0].title == "Applied AI Engineer"
    assert jobs[0].industry_domain.startswith("Financial Crime")


@pytest.mark.parametrize(
    ("company", "expected_mode"),
    [
        ("Flash AI", WorkMode.REMOTE),
        ("Quantifind", WorkMode.MIXED),
        ("Booz Allen Hamilton", WorkMode.ONSITE),
        ("Intel", WorkMode.HYBRID),
        ("BlackLine", WorkMode.MIXED),
        ("Verus Research LLC", WorkMode.ONSITE),
        ("Camden Property Trust", WorkMode.REMOTE),
    ],
)
def test_work_mode_parsing(company: str, expected_mode: WorkMode) -> None:
    """Work modes cover remote, hybrid, onsite, and mixed cases in the dataset."""
    jobs = {job.company: job for job in load_jobs(JOBS_CSV)}
    assert jobs[company].work_mode == expected_mode


def test_clean_experience_requirements_parse_exactly() -> None:
    """Simple N+ years requirements parse to exact minimum years."""
    jobs = {job.company: job for job in load_jobs(JOBS_CSV)}

    assert jobs["Quantifind"].experience_parse_status == "exact"
    assert jobs["Quantifind"].minimum_years == 4
    assert jobs["CNN"].experience_parse_status == "exact"
    assert jobs["CNN"].minimum_years == 1


def test_approximate_requirements_are_marked_approximate() -> None:
    """Approximately-worded requirements are marked approximate."""
    jobs = {job.company: job for job in load_jobs(JOBS_CSV)}
    bear = jobs["BEAR Cloud"]

    assert bear.experience_parse_status == "approximate"
    assert bear.minimum_years == 3
    assert 3 in bear.experience_year_values


def test_ambiguous_and_unspecified_rows_do_not_invent_minimums() -> None:
    """Multi-clause or staff-level rows remain conservative."""
    jobs = {job.company: job for job in load_jobs(JOBS_CSV)}

    assert jobs["Snap Inc."].experience_parse_status == "ambiguous"
    assert jobs["Snap Inc."].minimum_years is None

    assert jobs["CVS Health"].experience_parse_status == "unspecified"
    assert jobs["CVS Health"].minimum_years is None

    assert jobs["Infinite Electronics International"].experience_parse_status == "unspecified"
    assert jobs["Infinite Electronics International"].minimum_years is None


def test_parse_experience_requirement_examples() -> None:
    """Direct parser checks for representative requirement strings."""
    status, minimum, values = parse_experience_requirement("3+ years of experience")
    assert status == "exact"
    assert minimum == 3
    assert values == [3]

    status, minimum, values = parse_experience_requirement(
        "3+ years after a bachelor's degree, or 2+ years after a master's degree"
    )
    assert status == "ambiguous"
    assert minimum is None
    assert values == [3, 2]
