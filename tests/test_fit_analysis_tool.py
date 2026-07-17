"""Tests for the evidence-grounded fit analysis tool."""

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
from src.tools.fit_analysis import FitAnalysisError, fit_analysis_tool, parse_resume_visibility
from src.tools.scoring import JobScore, ScoreBreakdown, SkillMatchEvidence, scoring_tool

JOBS_CSV = ROOT / "data" / "AI_ML_Jobs_Dataset_20.csv"
PROFILE = ROOT / "candidate" / "profile.json"
PORTFOLIO = ROOT / "candidate" / "portfolio.json"
EVIDENCE = ROOT / "candidate" / "evidence_registry.json"
MEMORY = ROOT / "memory.json"
RESUME = ROOT / "candidate" / "sample_resume.tex"
CANDIDATE_ID = "cand-mira-solenne-001"


@pytest.fixture
def repository_bundle() -> CandidateBundle:
    return load_candidate_bundle(PROFILE, PORTFOLIO, EVIDENCE)


@pytest.fixture
def repository_memory() -> CandidateMemory:
    return load_memory(MEMORY, CANDIDATE_ID)


@pytest.fixture
def top3_jobs_and_scores(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> tuple[list[Job], list]:
    jobs = load_jobs(JOBS_CSV)
    accepted = filtering_tool(jobs, repository_bundle.profile).accepted_jobs
    scores = scoring_tool(accepted, repository_bundle, repository_memory).top_3
    job_by_id = {job.job_id: job for job in accepted}
    top_jobs = [job_by_id[score.job_id] for score in scores]
    return top_jobs, scores


def make_job(**overrides: object) -> Job:
    defaults = {
        "title": "Machine Learning Engineer",
        "company": "Synthetic Co",
        "industry_domain": "Healthcare / Clinical Operations",
        "location_raw": "Remote, United States",
        "work_mode": WorkMode.REMOTE,
        "required_skills": ["Python", "PyTorch"],
        "experience_requirement_raw": "3+ years",
        "minimum_years": 3,
        "experience_parse_status": "exact",
        "experience_year_values": [3],
        "url": "https://example.com/fit-test",
    }
    data = {**defaults, **overrides}
    url = str(data["url"])
    skills = list(data["required_skills"])  # type: ignore[arg-type]
    return Job(
        job_id=derive_job_id(url),
        title=str(data["title"]),
        company=str(data["company"]),
        industry_domain=str(data["industry_domain"]),
        location_raw=str(data["location_raw"]),
        work_mode=data["work_mode"],  # type: ignore[arg-type]
        required_skills_raw="; ".join(skills),
        required_skills=skills,
        experience_requirement_raw=str(data["experience_requirement_raw"]),
        minimum_years=data["minimum_years"],  # type: ignore[arg-type]
        experience_parse_status=data["experience_parse_status"],  # type: ignore[arg-type]
        experience_year_values=list(data["experience_year_values"]),  # type: ignore[arg-type]
        job_description=str(data.get("job_description", "Requires Python and healthcare AI experience.")),
        company_details="Synthetic company.",
        url=url,
        source_row=1,
    )


def make_job_score(job: Job, **overrides: object) -> JobScore:
    defaults = {
        "rank": 1,
        "final_score": 80.0,
        "matched_required_skills": ["Python"],
        "unmatched_required_skills": ["PyTorch"],
        "matched_skill_evidence": [
            SkillMatchEvidence(
                job_skill="Python",
                matched=True,
                canonical_job_skill="python",
                canonical_candidate_skill="python",
                evidence_sources=[],
            ),
            SkillMatchEvidence(
                job_skill="PyTorch",
                matched=False,
                canonical_job_skill="pytorch",
                canonical_candidate_skill=None,
                evidence_sources=[],
            ),
        ],
        "candidate_years": 3,
        "required_minimum_years": job.minimum_years,
        "experience_parse_status": job.experience_parse_status,
        "domain_matches": ["healthcare"],
        "location_explanation": "Remote allowed.",
    }
    data = {**defaults, **overrides}
    breakdown = ScoreBreakdown(
        skills_score=70.0,
        skills_weighted=35.0,
        experience_score=100.0,
        experience_weighted=25.0,
        industry_domain_score=100.0,
        industry_domain_weighted=15.0,
        location_score=100.0,
        location_weighted=10.0,
    )
    return JobScore(
        rank=int(data["rank"]),  # type: ignore[arg-type]
        job_id=job.job_id,
        title=job.title,
        company=job.company,
        final_score=float(data["final_score"]),  # type: ignore[arg-type]
        breakdown=breakdown,
        matched_required_skills=list(data["matched_required_skills"]),  # type: ignore[arg-type]
        unmatched_required_skills=list(data["unmatched_required_skills"]),  # type: ignore[arg-type]
        matched_skill_evidence=list(data["matched_skill_evidence"]),  # type: ignore[arg-type]
        candidate_years=int(data["candidate_years"]),  # type: ignore[arg-type]
        required_minimum_years=data["required_minimum_years"],  # type: ignore[arg-type]
        experience_parse_status=str(data["experience_parse_status"]),  # type: ignore[arg-type]
        domain_matches=list(data["domain_matches"]),  # type: ignore[arg-type]
        location_explanation=str(data["location_explanation"]),  # type: ignore[arg-type]
    )


def make_memory_fact(**overrides: object) -> MemoryFact:
    defaults = {
        "fact_id": "fact-skill-001",
        "fact_type": "skill",
        "skill_tags": ["LangChain"],
        "normalized_value": "LangChain",
        "statement": "Candidate has LangChain experience.",
    }
    data = {**defaults, **overrides}
    return MemoryFact(
        fact_id=str(data["fact_id"]),
        fact_type=data["fact_type"],  # type: ignore[arg-type]
        statement=str(data["statement"]),
        normalized_value=data["normalized_value"],  # type: ignore[arg-type]
        skill_tags=list(data["skill_tags"]),  # type: ignore[arg-type]
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


def _collect_findings(result) -> list:
    findings = list(result.relevant_experience.findings)
    findings.append(result.seniority.primary_finding)
    findings.extend(result.education.findings)
    findings.extend(result.core_skills.findings)
    findings.extend(result.projects.findings)
    return findings


def test_actual_top3_workflow(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
    top3_jobs_and_scores: tuple[list[Job], list],
) -> None:
    top_jobs, top_scores = top3_jobs_and_scores
    resume_before = RESUME.read_bytes()
    bundle_before = repository_bundle.model_dump()
    memory_before = repository_memory.model_dump()

    results = [
        fit_analysis_tool(job, score, repository_bundle, repository_memory, RESUME)
        for job, score in zip(top_jobs, top_scores, strict=True)
    ]

    assert len(results) == 3
    assert [result.job_id for result in results] == [score.job_id for score in top_scores]
    for result in results:
        assert result.relevant_experience.findings
        assert result.seniority.primary_finding
        assert result.education.findings
        assert result.core_skills.findings
        assert result.projects.findings
        for heading in (
            "Relevant Experience",
            "Seniority",
            "Education",
            "Core Skills",
            "Projects",
        ):
            assert heading in result.formatted_text
        assert result.formatted_text.startswith("Tell me why this job is a good fit for me.")

    assert RESUME.read_bytes() == resume_before
    assert repository_bundle.model_dump() == bundle_before
    assert repository_memory.model_dump() == memory_before


def test_citation_integrity(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
    top3_jobs_and_scores: tuple[list[Job], list],
) -> None:
    job, score = top3_jobs_and_scores[0][0], top3_jobs_and_scores[1][0]
    result = fit_analysis_tool(job, score, repository_bundle, repository_memory, RESUME)
    findings = _collect_findings(result)
    structured_count = sum(len(finding.citations) for finding in findings)

    for finding in findings:
        if finding.status in {"aligned", "improvement_needed", "genuine_gap"}:
            assert finding.citations
        if finding.status == "genuine_gap":
            assert any(citation.source_type == "job_posting" for citation in finding.citations)
        assert len(finding.citations) == len(_dedupe(finding.citations))

    assert result.evidence_citation_count == len(_dedupe([c for f in findings for c in f.citations]))


def _dedupe(citations: list) -> list:
    seen: set[tuple[str, str, str, str | None]] = set()
    out = []
    for citation in citations:
        key = (citation.source_type, citation.source_id, citation.source_field, citation.evidence_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(citation)
    return out


def test_skill_groups_are_non_overlapping(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
    top3_jobs_and_scores: tuple[list[Job], list],
) -> None:
    result = fit_analysis_tool(
        top3_jobs_and_scores[0][0],
        top3_jobs_and_scores[1][0],
        repository_bundle,
        repository_memory,
        RESUME,
    )
    aligned = set(result.core_skills.aligned_skills)
    evidenced = set(result.core_skills.evidenced_elsewhere_skills)
    gaps = set(result.core_skills.genuine_gaps)
    assert aligned.isdisjoint(evidenced)
    assert aligned.isdisjoint(gaps)
    assert evidenced.isdisjoint(gaps)


def test_evidence_elsewhere_and_genuine_gap_actions(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    job = make_job(required_skills=["Python", "LangChain"])
    score = make_job_score(
        job,
        matched_required_skills=["Python"],
        unmatched_required_skills=["LangChain"],
        matched_skill_evidence=[
            SkillMatchEvidence(
                job_skill="Python",
                matched=True,
                canonical_job_skill="python",
                canonical_candidate_skill="python",
                evidence_sources=[],
            ),
            SkillMatchEvidence(
                job_skill="LangChain",
                matched=False,
                canonical_job_skill="langchain",
                canonical_candidate_skill=None,
                evidence_sources=[],
            ),
        ],
    )
    result = fit_analysis_tool(job, score, repository_bundle, repository_memory, RESUME)
    assert "LangChain" in result.core_skills.genuine_gaps
    assert all(action.action_type != "add_evidenced_skill" for action in result.tailoring_actions if action.target_id == "LangChain")
    assert any(action.action_type == "preserve_genuine_gap" for action in result.tailoring_actions)


def test_memory_only_skill_is_evidence_elsewhere(
    repository_bundle: CandidateBundle,
) -> None:
    memory = CandidateMemory(
        schema_version="1.0",
        candidate_id=CANDIDATE_ID,
        facts=[make_memory_fact(skill_tags=["LangChain"], normalized_value="LangChain")],
    )
    job = make_job(required_skills=["LangChain"])
    score = make_job_score(
        job,
        matched_required_skills=["LangChain"],
        unmatched_required_skills=[],
        matched_skill_evidence=[
            SkillMatchEvidence(
                job_skill="LangChain",
                matched=True,
                canonical_job_skill="langchain",
                canonical_candidate_skill="langchain",
                evidence_sources=[],
            )
        ],
    )
    result = fit_analysis_tool(job, score, repository_bundle, memory, RESUME)
    assert "LangChain" in result.core_skills.evidenced_elsewhere_skills or "LangChain" in result.core_skills.aligned_skills


def test_non_skill_candidate_fact_not_used_as_skill(
    repository_bundle: CandidateBundle,
) -> None:
    memory = CandidateMemory(
        schema_version="1.0",
        candidate_id=CANDIDATE_ID,
        facts=[
            make_memory_fact(
                fact_id="fact-001",
                fact_type="candidate_fact",
                skill_tags=[],
                normalized_value="Prefers healthcare roles",
                statement="Prefers healthcare roles",
            )
        ],
    )
    job = make_job(required_skills=["Prefers healthcare roles"])
    score = make_job_score(
        job,
        matched_required_skills=[],
        unmatched_required_skills=["Prefers healthcare roles"],
        matched_skill_evidence=[
            SkillMatchEvidence(
                job_skill="Prefers healthcare roles",
                matched=False,
                canonical_job_skill="prefers healthcare roles",
                canonical_candidate_skill=None,
                evidence_sources=[],
            )
        ],
    )
    result = fit_analysis_tool(job, score, repository_bundle, memory, RESUME)
    assert "Prefers healthcare roles" in result.core_skills.genuine_gaps


def test_project_swap_rules(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    job = make_job(
        industry_domain="Manufacturing / Industrial Quality",
        required_skills=["PyTorch", "Computer Vision", "Docker", "Python"],
        job_description="Industrial computer vision deployment role.",
        url="https://example.com/vision-role",
    )
    score = make_job_score(
        job,
        matched_required_skills=["PyTorch", "Python", "Docker"],
        unmatched_required_skills=["Computer Vision"],
        matched_skill_evidence=[
            SkillMatchEvidence(job_skill=s, matched=(s != "Computer Vision"), canonical_job_skill=s.casefold(), canonical_candidate_skill=s.casefold() if s != "Computer Vision" else None, evidence_sources=[])
            for s in job.required_skills
        ],
    )
    result = fit_analysis_tool(job, score, repository_bundle, repository_memory, RESUME)
    assert len(result.projects.current_project_comparisons) == 3
    assert all(not item.on_base_resume for item in result.projects.replacement_comparisons)
    if result.project_swap_recommended:
        swap = result.projects.swap_suggestion
        assert swap is not None
        assert swap.remove_project_id in {p.project_id for p in repository_bundle.base_resume_projects()}
        assert swap.add_project_id in {p.project_id for p in repository_bundle.swap_available_projects()}
        assert swap.score_improvement >= 10
    else:
        assert any("best available evidence set" in finding.summary for finding in result.projects.findings)


def test_experience_actions_limited_to_editable_bullets(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    job = make_job(required_skills=["UnderwaterSkillXYZ"])
    score = make_job_score(
        job,
        matched_required_skills=[],
        unmatched_required_skills=["UnderwaterSkillXYZ"],
        matched_skill_evidence=[
            SkillMatchEvidence(
                job_skill="UnderwaterSkillXYZ",
                matched=False,
                canonical_job_skill="underwaterskillxyz",
                canonical_candidate_skill=None,
                evidence_sources=[],
            )
        ],
    )
    result = fit_analysis_tool(job, score, repository_bundle, repository_memory, RESUME)
    revise_actions = [
        action for action in result.tailoring_actions if action.action_type == "revise_editable_experience_bullet"
    ]
    assert len(revise_actions) <= 2
    for action in revise_actions:
        assert action.target_id in {"exp-primary-bullet-1", "exp-primary-bullet-2"}


def test_seniority_and_education_cases(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    aligned_job = make_job(minimum_years=3, experience_parse_status="exact")
    aligned_score = make_job_score(aligned_job, required_minimum_years=3, experience_parse_status="exact")
    aligned = fit_analysis_tool(aligned_job, aligned_score, repository_bundle, repository_memory, RESUME)
    assert aligned.seniority.primary_finding.status == "aligned"

    low_job = make_job(minimum_years=6, experience_parse_status="exact", url="https://example.com/low")
    low_score = make_job_score(
        low_job,
        required_minimum_years=6,
        experience_parse_status="exact",
        breakdown=ScoreBreakdown(
            skills_score=50,
            skills_weighted=25,
            experience_score=50,
            experience_weighted=12.5,
            industry_domain_score=60,
            industry_domain_weighted=9,
            location_score=100,
            location_weighted=10,
        ),
    )
    low = fit_analysis_tool(low_job, low_score, repository_bundle, repository_memory, RESUME)
    assert low.seniority.primary_finding.status == "improvement_needed"

    ambiguous_job = make_job(
        title="Staff Machine Learning Engineer",
        minimum_years=None,
        experience_parse_status="unspecified",
        experience_requirement_raw="Staff-level role",
        experience_year_values=[],
        url="https://example.com/ambiguous",
    )
    ambiguous_score = make_job_score(
        ambiguous_job,
        required_minimum_years=None,
        experience_parse_status="unspecified",
    )
    ambiguous = fit_analysis_tool(ambiguous_job, ambiguous_score, repository_bundle, repository_memory, RESUME)
    assert "ambiguous" in ambiguous.seniority.primary_finding.summary.lower() or "staff" in ambiguous.seniority.primary_finding.summary.lower()

    edu_job = make_job(
        job_description="Requires a master's degree in data science.",
        url="https://example.com/edu",
    )
    edu_score = make_job_score(edu_job)
    edu = fit_analysis_tool(edu_job, edu_score, repository_bundle, repository_memory, RESUME)
    assert any(finding.status == "aligned" for finding in edu.education.findings)

    no_req = fit_analysis_tool(make_job(url="https://example.com/noreq"), make_job_score(make_job(url="https://example.com/noreq")), repository_bundle, repository_memory, RESUME)
    assert any(finding.status == "informational" for finding in no_req.education.findings)


def test_determinism_and_input_immutability(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
    top3_jobs_and_scores: tuple[list[Job], list],
) -> None:
    job, score = top3_jobs_and_scores[0][0], top3_jobs_and_scores[1][0]
    job_before = job.model_dump()
    score_before = score.model_dump()
    first = fit_analysis_tool(job, score, repository_bundle, repository_memory, RESUME)
    second = fit_analysis_tool(job, score, repository_bundle, repository_memory, RESUME)
    assert first.model_dump() == second.model_dump()
    assert job.model_dump() == job_before
    assert score.model_dump() == score_before


def test_job_id_mismatch_raises(
    repository_bundle: CandidateBundle,
    repository_memory: CandidateMemory,
) -> None:
    job = make_job(url="https://example.com/a")
    score = make_job_score(make_job(url="https://example.com/b"))
    with pytest.raises(FitAnalysisError, match="does not match"):
        fit_analysis_tool(job, score, repository_bundle, repository_memory, RESUME)


def test_resume_visibility_parser_reads_skills_and_projects() -> None:
    visibility = parse_resume_visibility(RESUME)
    assert "Python" in visibility.skills_text
    assert visibility.experience_bullet_texts
    assert any("CarePath" in entry for entry in visibility.project_entries)
