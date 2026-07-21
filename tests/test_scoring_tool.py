"""Tests for the deterministic scoring tool."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.bundle import CandidateBundle
from src.models.job import Job, WorkMode, derive_job_id
from src.models.memory import CandidateMemory, MemoryFact, MemoryProvenance
from src.services.candidate_loader import load_candidate_bundle
from src.services.jobs_loader import load_jobs
from src.services.memory_loader import load_memory
from src.tools.filtering import filtering_tool
from src.tools.scoring import (
    build_candidate_skill_universe,
    normalize_skill,
    scoring_tool,
)

JOBS_CSV = ROOT / "data" / "AI_ML_Jobs_Dataset_20.csv"
PROFILE = ROOT / "candidate" / "profile.json"
PORTFOLIO = ROOT / "candidate" / "portfolio.json"
EVIDENCE = ROOT / "candidate" / "evidence_registry.json"
MEMORY = ROOT / "memory.json"
CANDIDATE_ID = "cand-mira-solenne-001"


@pytest.fixture
def repository_bundle() -> CandidateBundle:
    """Load the assignment candidate bundle."""
    return load_candidate_bundle(PROFILE, PORTFOLIO, EVIDENCE)


@pytest.fixture
def repository_memory() -> CandidateMemory:
    """Load the assignment memory document."""
    return load_memory(MEMORY, CANDIDATE_ID)


def make_job(
    *,
    title: str = "Machine Learning Engineer",
    company: str = "Synthetic Co",
    industry_domain: str = "Technology / AI",
    location_raw: str = "Remote, United States",
    work_mode: WorkMode = WorkMode.REMOTE,
    required_skills: list[str] | None = None,
    experience_requirement_raw: str = "3+ years",
    minimum_years: int | None = 3,
    experience_parse_status: str = "exact",
    experience_year_values: list[int] | None = None,
    url: str | None = None,
) -> Job:
    """Build a minimal synthetic job for scoring edge cases."""
    skills = required_skills or ["Python"]
    job_url = url or f"https://example.com/jobs/{derive_job_id(title + company)}"
    return Job(
        job_id=derive_job_id(job_url),
        title=title,
        company=company,
        industry_domain=industry_domain,
        location_raw=location_raw,
        work_mode=work_mode,
        required_skills_raw="; ".join(skills),
        required_skills=skills,
        experience_requirement_raw=experience_requirement_raw,
        minimum_years=minimum_years,
        experience_parse_status=experience_parse_status,  # type: ignore[arg-type]
        experience_year_values=experience_year_values or ([minimum_years] if minimum_years else []),
        job_description="Synthetic job description.",
        company_details="Synthetic company details.",
        url=job_url,
        source_row=1,
    )


def make_memory_fact(
    *,
    fact_id: str = "fact-skill-001",
    fact_type: str = "skill",
    skill_tags: list[str] | None = None,
    normalized_value: str | list[str] = "LangChain",
    statement: str = "Candidate has LangChain experience.",
) -> MemoryFact:
    """Build a synthetic memory fact."""
    return MemoryFact(
        fact_id=fact_id,
        fact_type=fact_type,  # type: ignore[arg-type]
        statement=statement,
        normalized_value=normalized_value,
        skill_tags=skill_tags or ["LangChain"],
        evidence_refs=["EV-PROJ-001"],
        provenance=MemoryProvenance(
            source="candidate_review",
            review_round=1,
            run_id="run-test-001",
            reviewer_role="reviewer",
        ),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        applied_in_run=True,
    )


def test_actual_filtered_dataset_scoring_invariants(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Score accepted repository jobs with deterministic invariants."""
    jobs = load_jobs(JOBS_CSV)
    jobs_before = [job.model_dump() for job in jobs]
    bundle_before = repository_bundle.model_dump()
    memory_before = repository_memory.model_dump()

    accepted = filtering_tool(jobs, repository_bundle.profile).accepted_jobs
    result = scoring_tool(accepted, repository_bundle, repository_memory)

    assert len(accepted) == 5
    assert result.total_scored == 5
    assert len(result.top_3) == 3
    assert [job.rank for job in result.ranked_jobs] == list(range(1, 6))
    assert result.ranked_jobs == sorted(
        result.ranked_jobs,
        key=lambda item: item.rank,
    )
    final_scores = [job.final_score for job in result.ranked_jobs]
    assert final_scores == sorted(final_scores, reverse=True)
    assert result.memory_fact_count == len(repository_memory.facts)
    assert (
        result.memory_skill_fact_count
        + result.memory_candidate_fact_count
        == result.memory_fact_count
    )
    assert sorted(
        result.memory_skill_fact_ids + result.memory_candidate_fact_ids
    ) == sorted(fact.fact_id for fact in repository_memory.facts)
    assert "candidate_fact entries are considered" in result.memory_scoring_policy

    for job_score in result.ranked_jobs:
        assert 0 <= job_score.final_score <= 100
        breakdown = job_score.breakdown
        for component in (
            breakdown.skills_score,
            breakdown.experience_score,
            breakdown.industry_domain_score,
            breakdown.location_score,
        ):
            assert 0 <= component <= 100
        weighted_sum = _round(
            breakdown.skills_weighted
            + breakdown.experience_weighted
            + breakdown.industry_domain_weighted
            + breakdown.location_weighted
        )
        assert abs(weighted_sum - job_score.final_score) <= 0.02

    second = scoring_tool(accepted, repository_bundle, repository_memory)
    assert result.model_dump() == second.model_dump()
    assert [job.model_dump() for job in jobs] == jobs_before
    assert repository_bundle.model_dump() == bundle_before
    assert repository_memory.model_dump() == memory_before


def _round(value: float) -> float:
    return round(value, 2)


def test_non_resume_portfolio_project_skill_matches(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Skills evidenced only in swap-available projects still match."""
    job = make_job(required_skills=["PyTorch"])
    result = scoring_tool([job], repository_bundle, repository_memory)
    evidence = result.ranked_jobs[0].matched_skill_evidence[0]
    assert evidence.matched is True
    assert any(source.source_type == "project" for source in evidence.evidence_sources)
    assert any(
        source.source_id == "proj-vision-inspect"
        for source in evidence.evidence_sources
    )


def test_experience_evidence_skill_matches(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Experience evidence records contribute to skill matching."""
    job = make_job(required_skills=["MLflow"])
    result = scoring_tool([job], repository_bundle, repository_memory)
    evidence = result.ranked_jobs[0].matched_skill_evidence[0]
    assert evidence.matched is True
    assert any(source.source_type == "evidence" for source in evidence.evidence_sources)


def test_missing_skill_remains_unmatched(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Skills absent from the profile remain unmatched."""
    job = make_job(required_skills=["UnderwaterBasketWeaving"])
    result = scoring_tool([job], repository_bundle, repository_memory)
    job_score = result.ranked_jobs[0]
    assert job_score.unmatched_required_skills == ["UnderwaterBasketWeaving"]
    assert job_score.matched_skill_evidence[0].evidence_sources == []


def test_memory_skill_fact_improves_skill_score(
    repository_bundle: CandidateBundle,
) -> None:
    """A skill present only in memory enters the skill universe."""
    memory = CandidateMemory(
        schema_version="1.0",
        candidate_id=CANDIDATE_ID,
        facts=[make_memory_fact(skill_tags=["LangChain"], normalized_value="LangChain")],
    )
    job = make_job(required_skills=["LangChain"])
    result = scoring_tool([job], repository_bundle, memory)
    evidence = result.ranked_jobs[0].matched_skill_evidence[0]
    assert evidence.matched is True
    assert any(source.source_type == "memory" for source in evidence.evidence_sources)
    assert result.memory_fact_count == 1
    assert result.memory_skill_fact_count == 1
    assert result.memory_candidate_fact_count == 0
    assert result.memory_skill_fact_ids == ["fact-skill-001"]
    assert result.memory_candidate_fact_ids == []


def test_candidate_fact_does_not_become_skill(
    repository_bundle: CandidateBundle,
) -> None:
    """Non-skill candidate facts do not become skill matches."""
    memory = CandidateMemory(
        schema_version="1.0",
        candidate_id=CANDIDATE_ID,
        facts=[
            make_memory_fact(
                fact_id="fact-candidate-001",
                fact_type="candidate_fact",
                skill_tags=[],
                normalized_value="Prefers healthcare roles",
                statement="Candidate prefers healthcare roles.",
            )
        ],
    )
    job = make_job(required_skills=["Prefers healthcare roles"])
    result = scoring_tool([job], repository_bundle, memory)
    assert result.ranked_jobs[0].unmatched_required_skills == ["Prefers healthcare roles"]
    assert result.memory_skill_fact_count == 0
    assert result.memory_candidate_fact_count == 1
    assert result.memory_skill_fact_ids == []
    assert result.memory_candidate_fact_ids == ["fact-candidate-001"]
    assert "do not directly change" in result.memory_scoring_policy


def test_empty_memory_scores_successfully(
    repository_bundle: CandidateBundle,
) -> None:
    """An isolated empty memory document scores successfully."""
    empty_memory = CandidateMemory(
        schema_version="1.0",
        candidate_id=CANDIDATE_ID,
        facts=[],
    )
    job = make_job(required_skills=["Python"])

    result = scoring_tool([job], repository_bundle, empty_memory)

    assert result.memory_fact_count == 0
    assert result.memory_skill_fact_count == 0
    assert result.memory_candidate_fact_count == 0
    assert result.memory_skill_fact_ids == []
    assert result.memory_candidate_fact_ids == []
    assert result.ranked_jobs[0].matched_required_skills == ["Python"]


def test_experience_equal_years_returns_100(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Exact minimum equal to candidate years scores 100 on experience."""
    job = make_job(minimum_years=3, experience_parse_status="exact")
    result = scoring_tool([job], repository_bundle, repository_memory)
    assert result.ranked_jobs[0].breakdown.experience_score == 100.0


def test_experience_below_minimum_is_proportional(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Candidate below minimum receives proportional experience score."""
    job = make_job(minimum_years=6, experience_parse_status="exact")
    result = scoring_tool([job], repository_bundle, repository_memory)
    assert result.ranked_jobs[0].breakdown.experience_score == 50.0


def test_approximate_requirement_receives_deduction(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Approximate requirements receive the uncertainty deduction."""
    job = make_job(
        minimum_years=3,
        experience_parse_status="approximate",
        experience_requirement_raw="approximately 3 years",
    )
    result = scoring_tool([job], repository_bundle, repository_memory)
    assert result.ranked_jobs[0].breakdown.experience_score == 95.0


def test_ambiguous_senior_and_staff_score_below_non_senior(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Ambiguous senior and staff titles score below a matching non-senior role."""
    plain = make_job(title="Machine Learning Engineer", minimum_years=None, experience_parse_status="ambiguous")
    senior = make_job(title="Senior Machine Learning Engineer", minimum_years=None, experience_parse_status="unspecified")
    staff = make_job(title="Staff Machine Learning Engineer", minimum_years=None, experience_parse_status="unspecified")

    plain_score = scoring_tool([plain], repository_bundle, repository_memory).ranked_jobs[0]
    senior_score = scoring_tool([senior], repository_bundle, repository_memory).ranked_jobs[0]
    staff_score = scoring_tool([staff], repository_bundle, repository_memory).ranked_jobs[0]

    assert plain_score.required_minimum_years is None
    assert senior_score.required_minimum_years is None
    assert staff_score.required_minimum_years is None
    assert plain_score.breakdown.experience_score == 100.0
    assert senior_score.breakdown.experience_score < plain_score.breakdown.experience_score
    assert staff_score.breakdown.experience_score < senior_score.breakdown.experience_score


def test_skill_normalization_aliases(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Normalization handles ML, GenAI, RAG, and scikit-learn aliases."""
    jobs = [
        make_job(required_skills=["ML"], url="https://example.com/a"),
        make_job(required_skills=["GenAI"], url="https://example.com/b"),
        make_job(required_skills=["RAG"], url="https://example.com/c"),
        make_job(required_skills=["sklearn"], url="https://example.com/d"),
    ]
    result = scoring_tool(jobs, repository_bundle, repository_memory)
    by_id = {job.job_id: job for job in result.ranked_jobs}
    assert by_id[derive_job_id("https://example.com/a")].matched_required_skills == ["ML"]
    assert by_id[derive_job_id("https://example.com/b")].matched_required_skills == ["GenAI"]
    assert by_id[derive_job_id("https://example.com/c")].matched_required_skills == ["RAG"]
    assert by_id[derive_job_id("https://example.com/d")].matched_required_skills == ["sklearn"]
    assert normalize_skill("Underwater Basket") != normalize_skill("Basket Weaving")


def test_domain_healthcare_and_manufacturing_scores(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Healthcare and manufacturing roles receive higher domain alignment."""
    healthcare = make_job(
        title="Machine Learning Engineer",
        industry_domain="Healthcare / Clinical Operations",
        url="https://example.com/health",
    )
    manufacturing = make_job(
        title="Machine Learning Engineer",
        industry_domain="Manufacturing / Industrial Quality",
        url="https://example.com/mfg",
    )
    unrelated = make_job(
        title="Machine Learning Engineer",
        industry_domain="Underwater Basket Weaving",
        url="https://example.com/other",
    )
    result = scoring_tool([healthcare, manufacturing, unrelated], repository_bundle, repository_memory)
    scores_by_id = {job.job_id: job for job in result.ranked_jobs}
    healthcare_score = scores_by_id[healthcare.job_id]
    manufacturing_score = scores_by_id[manufacturing.job_id]
    unrelated_score = scores_by_id[unrelated.job_id]
    assert healthcare_score.domain_matches == ["healthcare"]
    assert healthcare_score.breakdown.industry_domain_score >= unrelated_score.breakdown.industry_domain_score
    assert manufacturing_score.breakdown.industry_domain_score >= unrelated_score.breakdown.industry_domain_score


def test_location_scoring_rules(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Location scoring follows remote, preferred onsite, preferred hybrid, and reject rules."""
    remote = make_job(location_raw="Remote, United States", work_mode=WorkMode.REMOTE, url="https://example.com/r")
    houston = make_job(
        location_raw="Houston, TX",
        work_mode=WorkMode.ONSITE,
        url="https://example.com/h",
    )
    austin_hybrid = make_job(
        location_raw="Austin, Texas (Hybrid)",
        work_mode=WorkMode.HYBRID,
        url="https://example.com/a",
    )
    atlanta = make_job(
        location_raw="Atlanta, GA",
        work_mode=WorkMode.ONSITE,
        url="https://example.com/x",
    )
    result = scoring_tool([remote, houston, austin_hybrid, atlanta], repository_bundle, repository_memory)
    by_url = {job.job_id: job for job in result.ranked_jobs}
    assert by_url[derive_job_id("https://example.com/r")].breakdown.location_score == 100.0
    assert by_url[derive_job_id("https://example.com/h")].breakdown.location_score == 100.0
    assert by_url[derive_job_id("https://example.com/a")].breakdown.location_score == 95.0
    assert by_url[derive_job_id("https://example.com/x")].breakdown.location_score == 0.0


def test_ranking_tie_break_is_deterministic(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Tie-breaking follows final score, components, input order, and job_id."""
    first = make_job(title="Machine Learning Engineer", company="A", url="https://example.com/tie-a")
    second = make_job(title="Machine Learning Engineer", company="B", url="https://example.com/tie-b")
    result = scoring_tool([first, second], repository_bundle, repository_memory)
    assert result.ranked_jobs[0].final_score == result.ranked_jobs[1].final_score
    assert result.ranked_jobs[0].job_id == first.job_id
    assert result.top_3 == result.ranked_jobs


def test_top_3_warning_when_fewer_than_three_jobs(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Fewer than three jobs yields a warning and full top_3."""
    job = make_job(required_skills=["Python"])
    result = scoring_tool([job], repository_bundle, repository_memory)
    assert len(result.top_3) == 1
    assert result.warning is not None


def test_matched_skills_include_evidence_sources(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Every matched skill includes at least one evidence source reference."""
    job = make_job(required_skills=["Python", "FastAPI"])
    result = scoring_tool([job], repository_bundle, repository_memory)
    for evidence in result.ranked_jobs[0].matched_skill_evidence:
        if evidence.matched:
            assert evidence.evidence_sources


def test_skill_universe_uses_whole_profile(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    """Skill universe includes master skills, projects, evidence, and memory."""
    memory = CandidateMemory(
        schema_version="1.0",
        candidate_id=CANDIDATE_ID,
        facts=[make_memory_fact()],
    )
    universe = build_candidate_skill_universe(repository_bundle, memory)
    source_types = {
        source.source_type
        for sources in universe.canonical_to_sources.values()
        for source in sources
    }
    assert "master_skill" in source_types
    assert "project" in source_types
    assert "evidence" in source_types
    assert "memory" in source_types

def test_resume_tex_only_skill_contributes_to_scoring(
    tmp_path: Path,
    repository_bundle: CandidateBundle,
) -> None:
    """A skill found only in the LaTeX resume becomes auditable score evidence."""
    resume_tex = tmp_path / "resume_only.tex"
    resume_tex.write_text(
        "\n".join(
            [
                r"\documentclass{article}",
                r"\begin{document}",
                r"\section{Skills}",
                r"\begin{itemize}",
                r"\small\item{\textbf{Systems:} ResumeOnlyOrchestration}",
                r"\end{itemize}",
                r"\end{document}",
            ]
        ),
        encoding="utf-8",
    )
    empty_memory = CandidateMemory(
        schema_version="1.0",
        candidate_id=CANDIDATE_ID,
        facts=[],
    )
    job = make_job(required_skills=["ResumeOnlyOrchestration"])

    result = scoring_tool(
        [job],
        repository_bundle,
        empty_memory,
        resume_tex,
    )

    evidence = result.ranked_jobs[0].matched_skill_evidence[0]
    assert evidence.matched is True
    assert any(
        source.source_type == "resume_tex"
        and source.source_id == "resume_only.tex"
        for source in evidence.evidence_sources
    )
    assert result.resume_skill_count == 1

