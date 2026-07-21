"""Focused tests for the evidence-enforcing Cover Letter Tool."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError
from pypdf import PdfWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.memory import CandidateMemory, MemoryFact, MemoryProvenance
from src.services.candidate_loader import load_candidate_bundle
from src.services.jobs_loader import load_jobs
from src.services.memory_loader import load_memory
from src.services.resume_finalizer import FinalizedResumeResult
from src.tools.cover_letter import (
    CoverLetterCitation,
    CoverLetterCompilationResult,
    CoverLetterEvidenceError,
    CoverLetterFinalizedResumeError,
    CoverLetterInputMismatchError,
    CoverLetterOnePageConstraintError,
    CoverLetterOutputError,
    CoverLetterParagraph,
    CoverLetterPlan,
    CoverLetterPlanError,
    CoverLetterSkillItem,
    compile_cover_letter_pdf,
    cover_letter_tool,
    latex_escape,
)
from src.tools.filtering import filtering_tool
from src.tools.fit_analysis import fit_analysis_tool
from src.tools.scoring import normalize_skill, scoring_tool

JOBS_CSV = ROOT / "data" / "AI_ML_Jobs_Dataset_20.csv"
PROFILE = ROOT / "candidate" / "profile.json"
PORTFOLIO = ROOT / "candidate" / "portfolio.json"
EVIDENCE = ROOT / "candidate" / "evidence_registry.json"
MEMORY = ROOT / "memory.json"
BASE_TEX = ROOT / "candidate" / "sample_resume.tex"
BASE_PDF = ROOT / "candidate" / "sample_resume.pdf"
FIXED_DATE = date(2026, 7, 17)


@pytest.fixture(scope="module")
def workflow():
    bundle = load_candidate_bundle(PROFILE, PORTFOLIO, EVIDENCE)
    memory = load_memory(MEMORY, bundle.profile.candidate_id)
    jobs = load_jobs(JOBS_CSV)
    accepted = filtering_tool(jobs, bundle.profile).accepted_jobs
    scores = scoring_tool(accepted, bundle, memory).top_3
    job_by_id = {job.job_id: job for job in accepted}
    triples = []
    for score in scores:
        job = job_by_id[score.job_id]
        analysis = fit_analysis_tool(job, score, bundle, memory, BASE_TEX)
        triples.append((job, score, analysis))
    return bundle, memory, triples


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_finalized(
    directory: Path,
    job,
    *,
    revision: int = 1,
    page_count: int = 1,
) -> FinalizedResumeResult:
    directory.mkdir(parents=True, exist_ok=True)
    before = directory / "resume_before.pdf"
    tex = directory / "resume_after.tex"
    pdf = directory / "resume_after.pdf"
    log = directory / "resume_change_log.json"
    shutil.copyfile(BASE_PDF, before)
    if page_count == 1:
        shutil.copyfile(BASE_PDF, pdf)
    else:
        writer = PdfWriter()
        for _ in range(page_count):
            writer.add_blank_page(width=612, height=792)
        with pdf.open("wb") as handle:
            writer.write(handle)
    tex.write_text(f"% approved {job.job_id} revision {revision}\n", encoding="utf-8")
    log.write_text(
        json.dumps({"job_id": job.job_id, "revision_round": revision}) + "\n",
        encoding="utf-8",
    )
    paths = {
        "resume_before.pdf": before,
        "resume_after.tex": tex,
        "resume_after.pdf": pdf,
        "resume_change_log.json": log,
    }
    return FinalizedResumeResult(
        job_id=job.job_id,
        title=job.title,
        company=job.company,
        approved_revision_round=revision,
        destination_dir=directory,
        source_base_pdf_path=before,
        source_draft_tex_path=tex,
        source_draft_pdf_path=pdf,
        source_change_log_path=log,
        resume_before_path=before,
        resume_after_tex_path=tex,
        resume_after_pdf_path=pdf,
        resume_change_log_path=log,
        page_count=page_count,
        copied_file_sha256={name: _hash(path) for name, path in paths.items()},
    )


def _job_citation(job, field: str = "job_description") -> CoverLetterCitation:
    return CoverLetterCitation(
        source_type="job_posting",
        source_id=job.job_id,
        source_field=field,
    )


def _bullet_citation() -> CoverLetterCitation:
    return CoverLetterCitation(
        source_type="experience_bullet",
        source_id="exp-primary-bullet-2",
        source_field="text",
        evidence_id="EV-EXP-BULLET-002",
    )


def _project_citation() -> CoverLetterCitation:
    return CoverLetterCitation(
        source_type="portfolio_project",
        source_id="proj-carepath-rag",
        source_field="short_description",
        evidence_id="EV-PROJ-001",
    )


def _skill_citation(skill: str) -> CoverLetterCitation:
    canonical = normalize_skill(skill, has_vector_search=True)
    if canonical in {"python", "sql", "bash"}:
        source_id = "EV-SKILL-LANG"
    elif canonical in {
        "retrieval augmented generation",
        "embeddings",
        "vector search",
        "prompt engineering",
    }:
        source_id = "EV-SKILL-GENAI"
    elif canonical in {
        "mlops",
        "docker",
        "mlflow",
        "model monitoring",
        "ci cd",
        "aws",
    }:
        source_id = "EV-SKILL-MLOPS"
    elif canonical in {"rest api", "fastapi", "git", "pytest", "postgresql"}:
        source_id = "EV-SKILL-SYSTEMS"
    else:
        source_id = "EV-SKILL-ML"
    return CoverLetterCitation(
        source_type="evidence_registry",
        source_id=source_id,
        source_field="supported_skills",
        evidence_id=source_id,
    )


def _hook(job) -> str:
    return " ".join(job.company_details.split()[:12]).rstrip(".,;:")


def _valid_skills(job) -> list[str]:
    preferred = {
        "Chickasaw Nation Industries": ["Python", "REST APIs", "RAG", "Docker"],
        "Camden Property Trust": ["Python", "REST APIs", "RAG", "MLOps"],
        "Flash AI": ["Python", "RAG", "Embeddings", "NLP"],
        "BlackLine": ["RAG", "AWS", "MLOps"],
    }
    return preferred[job.company]


def _paragraph(job, *, text: str | None = None) -> CoverLetterParagraph:
    return CoverLetterParagraph(
        text=text
        or (
            "My experience includes delivering a retrieval-augmented knowledge API with "
            "FastAPI, embeddings, and vector search, grounded in controlled evaluation. "
            "The role's focus on reliable AI workflows aligns with my CarePath project, "
            "which applied retrieval and API design to synthetic documents."
        ),
        reason="Map directly evidenced retrieval and project work to the posting.",
        citations=[_job_citation(job), _bullet_citation(), _project_citation()],
    )


def _valid_plan(job, *, paragraphs: int = 1) -> CoverLetterPlan:
    body = [_paragraph(job)]
    if paragraphs == 2:
        body.append(
            CoverLetterParagraph(
                text=(
                    "I also bring practical model-release discipline through containerized "
                    "services, repeatable tests, and monitored evaluation. That combination "
                    "supports the posting's emphasis on maintainable delivery while staying "
                    "grounded in my documented ModelWatch project and professional experience."
                ),
                reason="Connect documented MLOps practices to delivery requirements.",
                citations=[
                    _job_citation(job),
                    CoverLetterCitation(
                        source_type="experience_bullet",
                        source_id="exp-primary-bullet-3",
                        source_field="text",
                        evidence_id="EV-EXP-BULLET-003",
                    ),
                    CoverLetterCitation(
                        source_type="portfolio_project",
                        source_id="proj-model-watch",
                        source_field="short_description",
                        evidence_id="EV-PROJ-003",
                    ),
                ],
            )
        )
    return CoverLetterPlan(
        job_id=job.job_id,
        company_hook_phrase=_hook(job),
        company_hook_source_field="company_details",
        body_paragraphs=body,
        skills=[
            CoverLetterSkillItem(skill=skill, citations=[_skill_citation(skill)])
            for skill in _valid_skills(job)
        ],
        closing_sentence=(
            "I welcome the opportunity to discuss how this evidence-grounded experience "
            "can support your team."
        ),
        plan_rationale="Use only supplied evidence and the validated company hook.",
        letter_date=FIXED_DATE,
    )


def _fake_compiler(tex_path: Path, *, timeout_seconds: int = 60):
    del timeout_seconds
    pdf_path = tex_path.with_suffix(".pdf")
    shutil.copyfile(BASE_PDF, pdf_path)
    return CoverLetterCompilationResult(
        command=["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        return_code=0,
        pdf_path=pdf_path,
        page_count=1,
        stdout_tail="simulated",
        stderr_tail="",
    )


def _call(
    workflow,
    tmp_path: Path,
    monkeypatch,
    *,
    triple_index: int = 0,
    plan: CoverLetterPlan | None = None,
    finalized: FinalizedResumeResult | None = None,
    memory: CandidateMemory | None = None,
):
    bundle, repository_memory, triples = workflow
    job, score, analysis = triples[triple_index]
    monkeypatch.setattr("src.tools.cover_letter.compile_cover_letter_pdf", _fake_compiler)
    return cover_letter_tool(
        job,
        score,
        analysis,
        bundle,
        memory or repository_memory,
        finalized or _make_finalized(tmp_path / "finalized", job),
        tmp_path / "letter",
        plan or _valid_plan(job),
    )


def test_actual_top3_generate_exactly_three_real_one_page_letters(
    workflow,
    tmp_path: Path,
) -> None:
    bundle, memory, triples = workflow
    results = []
    for job, score, analysis in triples:
        result = cover_letter_tool(
            job,
            score,
            analysis,
            bundle,
            memory,
            _make_finalized(tmp_path / "finalized" / job.job_id, job),
            tmp_path / "letters" / job.job_id,
            _valid_plan(job),
        )
        results.append(result)
    assert len(results) == 3
    assert [result.job_id for result in results] == [
        score.job_id for _, score, _ in triples
    ]
    assert all(result.page_count == 1 for result in results)
    assert len(list((tmp_path / "letters").glob("*/cover_letter.pdf"))) == 3


def test_required_structure_one_and_two_paragraphs(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, _, triples = workflow
    job = triples[0][0]
    for count in (1, 2):
        result = _call(
            workflow,
            tmp_path / str(count),
            monkeypatch,
            plan=_valid_plan(job, paragraphs=count),
        )
        tex = result.tex_path.read_text(encoding="utf-8")
        persona = bundle.profile.persona
        assert persona.full_name in tex
        assert persona.email in tex
        assert persona.phone in tex
        assert f"{persona.city}, {persona.state}" in tex
        assert persona.github in tex
        assert "Dear Hiring Manager," in tex
        assert job.title in tex and job.company in tex
        assert latex_escape(_hook(job)) in tex
        assert r"\textbf{Relevant skills:}" in tex
        assert result.paragraph_count == count
        assert "Sincerely," in tex
        assert "Use only supplied evidence" not in tex


def test_plan_shape_hook_and_closing_validation(workflow) -> None:
    job = workflow[2][0][0]
    valid = _valid_plan(job)
    with pytest.raises(ValidationError, match="1 or 2"):
        CoverLetterPlan(**{**valid.model_dump(), "body_paragraphs": []})
    with pytest.raises(ValidationError, match="1 or 2"):
        CoverLetterPlan(
            **{**valid.model_dump(), "body_paragraphs": valid.body_paragraphs * 3}
        )
    with pytest.raises(ValidationError, match="between 3 and 8"):
        CoverLetterPlan(**{**valid.model_dump(), "skills": valid.skills[:2]})
    with pytest.raises(ValidationError, match="candidate_name"):
        CoverLetterPlan(**valid.model_dump(), candidate_name="Untrusted Name")


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("description", "exact normalized substring"),
        ("invented", "exact normalized substring"),
        ("short", "at least 4 meaningful"),
    ],
)
def test_company_hook_rejections(
    workflow,
    tmp_path: Path,
    monkeypatch,
    kind: str,
    message: str,
) -> None:
    job = workflow[2][0][0]
    phrase = {
        "description": "Develop AI applications that improve federal workflows",
        "invented": "industry leading products for every global customer",
        "short": "technology and services",
    }[kind]
    plan = _valid_plan(job).model_copy(update={"company_hook_phrase": phrase})
    with pytest.raises((CoverLetterPlanError, CoverLetterEvidenceError), match=message):
        _call(workflow, tmp_path, monkeypatch, plan=plan)


def test_normalized_company_hook_is_accepted(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = workflow[2][0][0]
    phrase = _hook(job).upper() + "..."
    result = _call(
        workflow,
        tmp_path,
        monkeypatch,
        plan=_valid_plan(job).model_copy(update={"company_hook_phrase": phrase}),
    )
    assert result.company_hook_phrase == phrase


def test_evidence_ids_projects_experience_metrics_and_education(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = workflow[2][0][0]
    valid = _valid_plan(job)
    bad_evidence = valid.body_paragraphs[0].model_copy(
        update={
            "citations": [
                _job_citation(job),
                _bullet_citation().model_copy(update={"evidence_id": "EV-NOT-REAL"}),
                _project_citation(),
            ]
        }
    )
    with pytest.raises(CoverLetterEvidenceError, match="Unknown evidence"):
        _call(
            workflow,
            tmp_path / "evidence",
            monkeypatch,
            plan=valid.model_copy(update={"body_paragraphs": [bad_evidence]}),
        )
    for citation, match in (
        (_project_citation().model_copy(update={"source_id": "proj-missing"}), "project"),
        (_bullet_citation().model_copy(update={"source_id": "bullet-missing"}), "bullet"),
        (_project_citation().model_copy(update={"source_field": "missing_field"}), "field"),
    ):
        bad = valid.body_paragraphs[0].model_copy(
            update={"citations": [_job_citation(job), _bullet_citation(), citation]}
        )
        with pytest.raises(CoverLetterEvidenceError, match=match):
            _call(
                workflow,
                tmp_path / match,
                monkeypatch,
                plan=valid.model_copy(update={"body_paragraphs": [bad]}),
            )

    unsupported = valid.body_paragraphs[0].model_copy(
        update={"text": valid.body_paragraphs[0].text + " I improved results by 99%."}
    )
    with pytest.raises(CoverLetterEvidenceError, match="unsupported numeric"):
        _call(
            workflow,
            tmp_path / "number",
            monkeypatch,
            plan=valid.model_copy(update={"body_paragraphs": [unsupported]}),
        )

    supported = valid.body_paragraphs[0].model_copy(
        update={"text": valid.body_paragraphs[0].text + " The pilot reduced lookup time by 36%."}
    )
    assert _call(
        workflow,
        tmp_path / "supported",
        monkeypatch,
        plan=valid.model_copy(update={"body_paragraphs": [supported]}),
    ).page_count == 1

    phd = valid.body_paragraphs[0].model_copy(
        update={
            "text": valid.body_paragraphs[0].text
            + " I earned a PhD that informs this work.",
            "citations": [
                *valid.body_paragraphs[0].citations,
                CoverLetterCitation(
                    source_type="education",
                    source_id="edu-ms-001",
                    source_field="degree",
                    evidence_id="EV-EDU-001",
                ),
            ],
        }
    )
    with pytest.raises(CoverLetterEvidenceError, match="doctoral"):
        _call(
            workflow,
            tmp_path / "phd",
            monkeypatch,
            plan=valid.model_copy(update={"body_paragraphs": [phd]}),
        )


def test_target_company_employment_and_genuine_gap_rejected(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    job, _, analysis = workflow[2][0]
    valid = _valid_plan(job)
    employed = valid.body_paragraphs[0].model_copy(
        update={"text": valid.body_paragraphs[0].text + f" I work at {job.company} today."}
    )
    with pytest.raises(CoverLetterEvidenceError, match="target company"):
        _call(
            workflow,
            tmp_path / "company",
            monkeypatch,
            plan=valid.model_copy(update={"body_paragraphs": [employed]}),
        )
    gap = analysis.core_skills.genuine_gaps[0]
    gap_skill = valid.skills[0].model_copy(
        update={"skill": gap, "citations": [_skill_citation("Python")]}
    )
    with pytest.raises(CoverLetterEvidenceError, match="genuine-gap"):
        _call(
            workflow,
            tmp_path / "gap",
            monkeypatch,
            plan=valid.model_copy(update={"skills": [gap_skill, *valid.skills[1:]]}),
        )


def test_skill_support_duplicates_job_only_and_memory_type(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, repository_memory, triples = workflow
    job, score, analysis = triples[0]
    valid = _valid_plan(job)
    duplicate = valid.skills[1].model_copy(
        update={"skill": "Python", "citations": [_skill_citation("Python")]}
    )
    with pytest.raises(CoverLetterPlanError, match="Duplicate canonical"):
        _call(
            workflow,
            tmp_path / "duplicate",
            monkeypatch,
            plan=valid.model_copy(update={"skills": [valid.skills[0], duplicate, *valid.skills[2:]]}),
        )
    job_only = valid.skills[0].model_copy(
        update={"citations": [_job_citation(job, "required_skills")]}
    )
    with pytest.raises(CoverLetterEvidenceError, match="candidate-evidence"):
        _call(
            workflow,
            tmp_path / "job-only",
            monkeypatch,
            plan=valid.model_copy(update={"skills": [job_only, *valid.skills[1:]]}),
        )

    def fact(fact_type: str) -> MemoryFact:
        return MemoryFact(
            fact_id=f"fact-langchain-{fact_type}",
            fact_type=fact_type,
            statement="Candidate has LangChain experience.",
            normalized_value="LangChain",
            skill_tags=["LangChain"],
            evidence_refs=["EV-PROJ-001"],
            provenance=MemoryProvenance(
                source="candidate_review",
                review_round=1,
                run_id="run-cover-letter-test",
                reviewer_role="reviewer",
            ),
            created_at=datetime(2026, 7, 17, tzinfo=timezone.utc),
            applied_in_run=True,
        )

    job_with_skill = job.model_copy(
        update={
            "required_skills": [*job.required_skills, "LangChain"],
            "required_skills_raw": job.required_skills_raw + "; LangChain",
        }
    )
    core = analysis.core_skills.model_copy(
        update={
            "evidenced_elsewhere_skills": [
                *analysis.core_skills.evidenced_elsewhere_skills,
                "LangChain",
            ],
            "genuine_gaps": [
                gap for gap in analysis.core_skills.genuine_gaps if gap != "LangChain"
            ],
        }
    )
    analysis_with_skill = analysis.model_copy(update={"core_skills": core})
    for fact_type, accepted in (("skill", True), ("candidate_fact", False)):
        memory_fact = fact(fact_type)
        memory = CandidateMemory(
            schema_version=repository_memory.schema_version,
            candidate_id=bundle.profile.candidate_id,
            facts=[memory_fact],
        )
        citation = CoverLetterCitation(
            source_type="memory_fact",
            source_id=memory_fact.fact_id,
            source_field="skill_tags",
        )
        plan = valid.model_copy(
            update={
                "skills": [
                    CoverLetterSkillItem(skill="LangChain", citations=[citation]),
                    *valid.skills[1:],
                ]
            }
        )
        finalized = _make_finalized(tmp_path / fact_type / "finalized", job_with_skill)
        monkeypatch.setattr(
            "src.tools.cover_letter.compile_cover_letter_pdf", _fake_compiler
        )
        call = lambda: cover_letter_tool(
            job_with_skill,
            score,
            analysis_with_skill,
            bundle,
            memory,
            finalized,
            tmp_path / fact_type / "letter",
            plan,
        )
        if accepted:
            assert call().skill_count == 4
        else:
            with pytest.raises(CoverLetterEvidenceError, match="candidate_fact|supports"):
                call()


def test_finalized_resume_gate_failures(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = workflow[2][0][0]
    valid = _make_finalized(tmp_path / "valid", job)
    assert _call(
        workflow,
        tmp_path / "accepted",
        monkeypatch,
        finalized=valid,
    ).approved_resume_revision == 1
    with pytest.raises(CoverLetterInputMismatchError, match="finalized_resume"):
        _call(
            workflow,
            tmp_path / "mismatch",
            monkeypatch,
            finalized=valid.model_copy(update={"job_id": "wrong"}),
        )
    missing_pdf = _make_finalized(tmp_path / "missing-pdf", job)
    missing_pdf.resume_after_pdf_path.unlink()
    with pytest.raises(CoverLetterFinalizedResumeError, match="missing"):
        _call(
            workflow,
            tmp_path / "missing-pdf-call",
            monkeypatch,
            finalized=missing_pdf,
        )
    missing_log = _make_finalized(tmp_path / "missing-log", job)
    missing_log.resume_change_log_path.unlink()
    with pytest.raises(CoverLetterFinalizedResumeError, match="missing"):
        _call(
            workflow,
            tmp_path / "missing-log-call",
            monkeypatch,
            finalized=missing_log,
        )
    unapproved = valid.model_copy(update={"approved_revision_round": -1})
    with pytest.raises(CoverLetterFinalizedResumeError, match="approved revision"):
        _call(
            workflow,
            tmp_path / "unapproved",
            monkeypatch,
            finalized=unapproved,
        )
    two_page = _make_finalized(tmp_path / "two-page", job, page_count=2)
    with pytest.raises(CoverLetterFinalizedResumeError, match="not marked"):
        _call(
            workflow,
            tmp_path / "two-page-call",
            monkeypatch,
            finalized=two_page,
        )


def test_latex_safety_compiler_arguments_and_cleanup(tmp_path: Path, monkeypatch) -> None:
    escaped = latex_escape(r"A&B 50% $x #1 a_b {c} ~ ^ \input{evil}")
    for token in (
        r"\&",
        r"\%",
        r"\$",
        r"\#",
        r"\_",
        r"\{",
        r"\}",
        r"\textasciitilde{}",
        r"\textasciicircum{}",
        r"\textbackslash{}input",
    ):
        assert token in escaped

    tex = tmp_path / "cover_letter.tex"
    tex.write_text(
        "\\documentclass{article}\\begin{document}Safe\\end{document}\n",
        encoding="utf-8",
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        shutil.copyfile(BASE_PDF, tmp_path / "cover_letter.pdf")
        for suffix in (".aux", ".log", ".out"):
            (tmp_path / f"cover_letter{suffix}").write_text("temporary", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("src.tools.cover_letter.subprocess.run", fake_run)
    result = compile_cover_letter_pdf(tex)
    command, kwargs = calls[0]
    assert "-shell-escape" not in command
    assert kwargs["shell"] is False
    assert result.page_count == 1
    assert not any((tmp_path / f"cover_letter{suffix}").exists() for suffix in (".aux", ".log", ".out"))


def test_raw_latex_plan_text_is_neutralized(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = workflow[2][0][0]
    valid = _valid_plan(job)
    injected = valid.body_paragraphs[0].model_copy(
        update={
            "text": valid.body_paragraphs[0].text
            + r" This plain-text note includes \input{evil} without execution."
        }
    )
    result = _call(
        workflow,
        tmp_path,
        monkeypatch,
        plan=valid.model_copy(update={"body_paragraphs": [injected]}),
    )
    tex = result.tex_path.read_text(encoding="utf-8")
    assert r"\textbackslash{}input\{evil\}" in tex
    assert r"\input{evil}" not in tex


def test_simulated_two_page_cover_letter_raises(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fail_two_pages(tex_path: Path, *, timeout_seconds: int = 60):
        del tex_path, timeout_seconds
        raise CoverLetterOnePageConstraintError(
            "Cover letter must be exactly one page; generated 2 pages"
        )

    monkeypatch.setattr(
        "src.tools.cover_letter.compile_cover_letter_pdf", fail_two_pages
    )
    bundle, memory, triples = workflow
    job, score, analysis = triples[0]
    with pytest.raises(CoverLetterOnePageConstraintError, match="2 pages"):
        cover_letter_tool(
            job,
            score,
            analysis,
            bundle,
            memory,
            _make_finalized(tmp_path / "finalized", job),
            tmp_path / "letter",
            _valid_plan(job),
        )


def test_output_evidence_determinism_no_overwrite_and_no_input_mutation(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, memory, triples = workflow
    job, score, analysis = triples[0]
    plan = _valid_plan(job)
    finalized_one = _make_finalized(tmp_path / "finalized-one", job)
    finalized_two = _make_finalized(tmp_path / "finalized-two", job)
    before = [
        value.model_dump()
        for value in (job, score, analysis, bundle, memory, finalized_one, plan)
    ]
    monkeypatch.setattr("src.tools.cover_letter.compile_cover_letter_pdf", _fake_compiler)
    first = cover_letter_tool(
        job,
        score,
        analysis,
        bundle,
        memory,
        finalized_one,
        tmp_path / "one",
        plan,
    )
    second = cover_letter_tool(
        job,
        score,
        analysis,
        bundle,
        memory,
        finalized_two,
        tmp_path / "two",
        plan,
    )
    assert {path.name for path in (tmp_path / "one").iterdir()} == {
        "cover_letter.tex",
        "cover_letter.pdf",
        "cover_letter_evidence.json",
    }
    assert first.plan_digest == second.plan_digest
    assert first.tex_path.read_bytes() == second.tex_path.read_bytes()
    evidence = json.loads(first.evidence_log_path.read_text(encoding="utf-8"))
    assert evidence["company_hook"]["citation"]["source_field"] == "company_details"
    assert evidence["body_paragraphs"][0]["citations"]
    assert evidence["skills"][0]["citations"]
    assert evidence["latex_sha256"] == first.tex_sha256
    assert evidence["pdf_sha256"] == first.pdf_sha256
    assert first.evidence_log_path.read_bytes().endswith(b"\n")
    assert not first.evidence_log_path.read_bytes().startswith(b"\xef\xbb\xbf")
    with pytest.raises(CoverLetterOutputError, match="overwrite"):
        cover_letter_tool(
            job,
            score,
            analysis,
            bundle,
            memory,
            finalized_one,
            tmp_path / "one",
            plan,
        )
    assert before == [
        value.model_dump()
        for value in (job, score, analysis, bundle, memory, finalized_one, plan)
    ]

def test_opening_sentence_handles_complete_company_hook_grammatically():
    from types import SimpleNamespace

    from src.tools.cover_letter import _render_opening_sentence

    job = SimpleNamespace(
        title="AI Engineer",
        company="BlackLine",
    )
    plan = SimpleNamespace(
        company_hook_phrase=(
            "BlackLine provides cloud software that automates and controls accounting"
        )
    )

    opening = _render_opening_sentence(job, plan)

    assert opening.startswith(
        "I am excited to apply for the AI Engineer position at BlackLine."
    )
    assert (
        "the company description highlights the following focus: "
        "BlackLine provides cloud software that automates and controls accounting."
        in opening
    )
    assert "Your work in" not in opening
    assert "is especially compelling to me" not in opening
    assert opening.count(plan.company_hook_phrase) == 1

