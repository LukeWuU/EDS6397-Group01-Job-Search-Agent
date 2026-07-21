"""Tests for the deterministic filtering tool."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.candidate import CandidateProfile
from src.models.job import Job, WorkMode, derive_job_id
from src.services.candidate_loader import load_candidate_bundle
from src.services.jobs_loader import load_jobs
from src.tools.filtering import (
    FilterReasonCode,
    FilterWarningCode,
    filtering_tool,
    normalize_company,
    normalize_title,
)

JOBS_CSV = ROOT / "data" / "AI_ML_Jobs_Dataset_20.csv"
PROFILE = ROOT / "candidate" / "profile.json"
PORTFOLIO = ROOT / "candidate" / "portfolio.json"
EVIDENCE = ROOT / "candidate" / "evidence_registry.json"


@pytest.fixture
def repository_jobs() -> list[Job]:
    """Load the assignment job dataset."""
    return load_jobs(JOBS_CSV)


@pytest.fixture
def repository_profile() -> CandidateProfile:
    """Load the assignment candidate profile."""
    bundle = load_candidate_bundle(PROFILE, PORTFOLIO, EVIDENCE)
    return bundle.profile


def make_job(
    *,
    title: str,
    company: str = "Synthetic Co",
    location_raw: str = "Remote, United States",
    work_mode: WorkMode = WorkMode.REMOTE,
    experience_requirement_raw: str = "3+ years",
    minimum_years: int | None = 3,
    experience_parse_status: str = "exact",
    experience_year_values: list[int] | None = None,
    url: str | None = None,
) -> Job:
    """Build a minimal synthetic job for edge-case tests."""
    job_url = url or f"https://example.com/jobs/{derive_job_id(title + company)}"
    return Job(
        job_id=derive_job_id(job_url),
        title=title,
        company=company,
        industry_domain="Test",
        location_raw=location_raw,
        work_mode=work_mode,
        required_skills_raw="Python",
        required_skills=["Python"],
        experience_requirement_raw=experience_requirement_raw,
        minimum_years=minimum_years,
        experience_parse_status=experience_parse_status,  # type: ignore[arg-type]
        experience_year_values=experience_year_values or ([minimum_years] if minimum_years else []),
        job_description="Synthetic job description.",
        company_details="Synthetic company details.",
        url=job_url,
        source_row=1,
    )


def make_profile(**preference_overrides: object) -> CandidateProfile:
    """Clone the repository profile with preference overrides."""
    bundle = load_candidate_bundle(PROFILE, PORTFOLIO, EVIDENCE)
    profile = bundle.profile.model_copy(deep=True)
    preferences = profile.preferences.model_copy(
        update=preference_overrides,
        deep=True,
    )
    return profile.model_copy(update={"preferences": preferences}, deep=True)


def test_actual_dataset_filtering_invariants(
    repository_jobs: list[Job],
    repository_profile: CandidateProfile,
) -> None:
    """All repository jobs receive one auditable decision in original order."""
    result = filtering_tool(repository_jobs, repository_profile)

    assert result.total_jobs == 20
    assert result.accepted_count + result.rejected_count == 20
    assert len(result.decisions) == 20
    assert result.accepted_count >= 3
    assert len({job.job_id for job in result.accepted_jobs}) == result.accepted_count
    assert len({job.job_id for job in result.rejected_jobs}) == result.rejected_count

    assert [job.job_id for job in result.accepted_jobs] == [
        decision.job_id for decision in result.decisions if decision.accepted
    ]
    assert [job.job_id for job in result.rejected_jobs] == [
        decision.job_id for decision in result.decisions if not decision.accepted
    ]

    for decision in result.decisions:
        if decision.accepted:
            assert decision.rejection_reasons == []
        else:
            assert decision.rejection_reasons


def test_filtering_is_deterministic(
    repository_jobs: list[Job],
    repository_profile: CandidateProfile,
) -> None:
    """Identical inputs produce identical serialized outputs."""
    first = filtering_tool(repository_jobs, repository_profile)
    second = filtering_tool(repository_jobs, repository_profile)
    assert first.model_dump() == second.model_dump()


def test_company_exclusion_and_non_match(
    repository_profile: CandidateProfile,
) -> None:
    """Excluded companies reject exactly; similar names do not."""
    excluded = make_job(title="Machine Learning Engineer", company="Booz Allen Hamilton")
    similar = make_job(title="Machine Learning Engineer", company="Booz Allen")

    excluded_result = filtering_tool([excluded], repository_profile)
    similar_result = filtering_tool([similar], repository_profile)

    assert excluded_result.rejected_count == 1
    assert excluded_result.decisions[0].rejection_reasons[0].code == FilterReasonCode.EXCLUDED_COMPANY
    assert similar_result.accepted_count == 1
    assert normalize_company("Booz Allen Hamilton") != normalize_company("Booz Allen")


def test_experience_exact_above_candidate_is_rejected(
    repository_profile: CandidateProfile,
) -> None:
    """Exact minimum above candidate years is rejected."""
    job = make_job(
        title="Machine Learning Engineer",
        experience_requirement_raw="5+ years",
        minimum_years=5,
        experience_parse_status="exact",
        experience_year_values=[5],
    )
    result = filtering_tool([job], repository_profile)
    codes = [reason.code for reason in result.decisions[0].rejection_reasons]
    assert FilterReasonCode.EXPERIENCE_MISMATCH in codes


def test_experience_exact_equal_candidate_is_accepted(
    repository_profile: CandidateProfile,
) -> None:
    """Exact minimum equal to candidate years is accepted."""
    job = make_job(
        title="Machine Learning Engineer",
        experience_requirement_raw="3+ years",
        minimum_years=3,
        experience_parse_status="exact",
        experience_year_values=[3],
    )
    result = filtering_tool([job], repository_profile)
    assert result.accepted_count == 1


def test_approximate_minimum_compares_and_warns(
    repository_profile: CandidateProfile,
) -> None:
    """Approximate requirements compare numerically and emit a warning."""
    job = make_job(
        title="Machine Learning Engineer",
        experience_requirement_raw="approximately 3 years",
        minimum_years=3,
        experience_parse_status="approximate",
        experience_year_values=[3],
    )
    result = filtering_tool([job], repository_profile)
    assert result.accepted_count == 1
    warning_codes = [warning.code for warning in result.decisions[0].warnings]
    assert FilterWarningCode.AMBIGUOUS_EXPERIENCE in warning_codes


def test_ambiguous_requirement_does_not_invent_minimum(
    repository_profile: CandidateProfile,
) -> None:
    """Ambiguous requirements never invent a numeric minimum."""
    job = make_job(
        title="Machine Learning Engineer",
        experience_requirement_raw=(
            "3+ years after a bachelor's degree, or 2+ years after a master's degree"
        ),
        minimum_years=None,
        experience_parse_status="ambiguous",
        experience_year_values=[3, 2],
    )
    result = filtering_tool([job], repository_profile)
    assert result.accepted_count == 1
    assert any(
        warning.code == FilterWarningCode.AMBIGUOUS_EXPERIENCE
        for warning in result.decisions[0].warnings
    )
    assert all(
        reason.code != FilterReasonCode.EXPERIENCE_MISMATCH
        for reason in result.decisions[0].rejection_reasons
    )


def test_unspecified_requirement_does_not_invent_minimum(
    repository_profile: CandidateProfile,
) -> None:
    """Unspecified requirements never invent a numeric minimum."""
    job = make_job(
        title="Machine Learning Engineer",
        experience_requirement_raw="Exact minimum not explicitly stated",
        minimum_years=None,
        experience_parse_status="unspecified",
        experience_year_values=[],
    )
    result = filtering_tool([job], repository_profile)
    assert result.accepted_count == 1
    assert any(
        warning.code == FilterWarningCode.UNSPECIFIED_EXPERIENCE
        for warning in result.decisions[0].warnings
    )


def test_seniority_fallback_rejects_staff_and_senior_roles(
    repository_profile: CandidateProfile,
) -> None:
    """Staff and senior titles reject 3-year candidates without explicit minimums."""
    staff_job = make_job(
        title="Staff Machine Learning Engineer",
        experience_requirement_raw="Staff-level role; exact minimum years should be confirmed",
        minimum_years=None,
        experience_parse_status="unspecified",
        experience_year_values=[],
        location_raw="Remote, United States",
        work_mode=WorkMode.REMOTE,
    )
    senior_job = make_job(
        title="Senior Machine Learning Engineer",
        experience_requirement_raw="Exact minimum not explicitly stated",
        minimum_years=None,
        experience_parse_status="unspecified",
        experience_year_values=[],
        location_raw="Remote, United States",
        work_mode=WorkMode.REMOTE,
    )
    plain_job = make_job(
        title="Machine Learning Engineer",
        experience_requirement_raw=(
            "3+ years after a bachelor's degree, or 2+ years after a master's degree"
        ),
        minimum_years=None,
        experience_parse_status="ambiguous",
        experience_year_values=[3, 2],
        location_raw="Remote, United States",
        work_mode=WorkMode.REMOTE,
    )

    staff_result = filtering_tool([staff_job], repository_profile)
    senior_result = filtering_tool([senior_job], repository_profile)
    plain_result = filtering_tool([plain_job], repository_profile)

    assert any(
        reason.code == FilterReasonCode.SENIORITY_MISMATCH
        for reason in staff_result.decisions[0].rejection_reasons
    )
    assert any(
        reason.code == FilterReasonCode.SENIORITY_MISMATCH
        for reason in senior_result.decisions[0].rejection_reasons
    )
    assert plain_result.accepted_count == 1


@pytest.mark.parametrize(
    ("job_title", "accepted"),
    [
        ("ML Engineer", True),
        ("GenAI Engineer", True),
        ("Underwater Basket Weaver", False),
    ],
)
def test_title_normalization_matching(
    repository_profile: CandidateProfile,
    job_title: str,
    accepted: bool,
) -> None:
    """Title normalization supports ML and GenAI aliases."""
    job = make_job(title=job_title)
    result = filtering_tool([job], repository_profile)
    assert result.accepted_count == (1 if accepted else 0)
    if not accepted:
        assert any(
            reason.code == FilterReasonCode.TITLE_MISMATCH
            for reason in result.decisions[0].rejection_reasons
        )


def test_location_rules(
    repository_profile: CandidateProfile,
) -> None:
    """Location and work-mode rules accept and reject deterministically."""
    remote_job = make_job(
        title="Machine Learning Engineer",
        location_raw="Remote, United States",
        work_mode=WorkMode.REMOTE,
    )
    houston_job = make_job(
        title="Machine Learning Engineer",
        location_raw="Houston, TX 77056 (In-office)",
        work_mode=WorkMode.ONSITE,
    )
    austin_hybrid_job = make_job(
        title="Machine Learning Engineer",
        location_raw="Austin, Texas (Hybrid)",
        work_mode=WorkMode.HYBRID,
    )
    atlanta_job = make_job(
        title="Machine Learning Engineer",
        location_raw="Atlanta, GA",
        work_mode=WorkMode.ONSITE,
    )
    dallas_hybrid_job = make_job(
        title="Machine Learning Engineer",
        location_raw="Dallas, TX (Hybrid)",
        work_mode=WorkMode.HYBRID,
    )
    unknown_job = make_job(
        title="Machine Learning Engineer",
        location_raw="United States",
        work_mode=WorkMode.UNKNOWN,
    )

    remote_result = filtering_tool([remote_job], repository_profile)
    houston_result = filtering_tool([houston_job], repository_profile)
    austin_result = filtering_tool([austin_hybrid_job], repository_profile)
    atlanta_result = filtering_tool([atlanta_job], repository_profile)
    dallas_result = filtering_tool([dallas_hybrid_job], repository_profile)
    unknown_result = filtering_tool([unknown_job], repository_profile)

    assert remote_result.accepted_count == 1
    assert houston_result.accepted_count == 1
    assert austin_result.accepted_count == 1
    assert atlanta_result.rejected_count == 1
    assert dallas_result.rejected_count == 1
    assert unknown_result.accepted_count == 1
    assert any(
        warning.code == FilterWarningCode.UNKNOWN_WORK_MODE
        for warning in unknown_result.decisions[0].warnings
    )


def test_remote_only_profile_rules() -> None:
    """Remote-only profiles accept remote and explicit mixed remote options only."""
    profile = make_profile(remote_preference="Remote only")

    remote_job = make_job(title="Machine Learning Engineer", work_mode=WorkMode.REMOTE)
    onsite_job = make_job(
        title="Machine Learning Engineer",
        location_raw="Houston, TX",
        work_mode=WorkMode.ONSITE,
    )
    hybrid_only_job = make_job(
        title="Machine Learning Engineer",
        location_raw="Austin, Texas (Hybrid)",
        work_mode=WorkMode.HYBRID,
    )
    mixed_remote_job = make_job(
        title="Machine Learning Engineer",
        location_raw="Remote or hybrid; office optional",
        work_mode=WorkMode.MIXED,
    )

    remote_result = filtering_tool([remote_job], profile)
    onsite_result = filtering_tool([onsite_job], profile)
    hybrid_result = filtering_tool([hybrid_only_job], profile)
    mixed_result = filtering_tool([mixed_remote_job], profile)

    assert remote_result.accepted_count == 1
    assert onsite_result.rejected_count == 1
    assert hybrid_result.rejected_count == 1
    assert mixed_result.accepted_count == 1
    assert any(
        reason.code == FilterReasonCode.REMOTE_ONLY_MISMATCH
        for reason in hybrid_result.decisions[0].rejection_reasons
    )


def test_inputs_remain_immutable(
    repository_jobs: list[Job],
    repository_profile: CandidateProfile,
) -> None:
    """Filtering does not mutate job or profile inputs."""
    jobs_before = [job.model_dump() for job in repository_jobs]
    profile_before = repository_profile.model_dump()

    filtering_tool(repository_jobs, repository_profile)

    assert [job.model_dump() for job in repository_jobs] == jobs_before
    assert repository_profile.model_dump() == profile_before


def test_title_normalization_helpers() -> None:
    """Normalization helpers expand ML and GenAI consistently."""
    assert "machine learning" in normalize_title("ML Engineer")
    assert "generative ai" in normalize_title("GenAI Engineer")
    assert normalize_title("Gen AI Engineer") == normalize_title("Generative AI Engineer")


def test_actual_dataset_multi_clause_experience_requirements(
    repository_jobs: list[Job],
    repository_profile: CandidateProfile,
) -> None:
    """Multi-clause AND requirements use the highest explicit minimum."""
    jobs_by_company = {job.company: job for job in repository_jobs}
    result = filtering_tool(repository_jobs, repository_profile)
    decisions_by_company = {
        decision.company: decision for decision in result.decisions
    }

    for company in ("Intel", "Chickasaw Nation Industries"):
        job = jobs_by_company[company]
        decision = decisions_by_company[company]

        assert job.experience_parse_status == "exact"
        assert job.minimum_years == 5
        assert decision.accepted is False
        assert any(
            reason.code == FilterReasonCode.EXPERIENCE_MISMATCH
            for reason in decision.rejection_reasons
        )

    camden_job = jobs_by_company["Camden Property Trust"]
    camden_decision = decisions_by_company["Camden Property Trust"]

    assert camden_job.experience_parse_status == "exact"
    assert camden_job.minimum_years == 3
    assert camden_decision.accepted is True
    assert all(
        reason.code != FilterReasonCode.EXPERIENCE_MISMATCH
        for reason in camden_decision.rejection_reasons
    )
