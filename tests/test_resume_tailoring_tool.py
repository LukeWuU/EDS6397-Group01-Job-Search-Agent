"""Focused tests for the evidence-enforcing Resume Tailoring Tool."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.memory import CandidateMemory, MemoryFact, MemoryProvenance
from src.services.candidate_loader import load_candidate_bundle
from src.services.jobs_loader import load_jobs
from src.services.memory_loader import load_memory
from src.tools.filtering import filtering_tool
from src.tools.fit_analysis import fit_analysis_tool
from src.tools.resume_tailoring import (
    CompilationResult,
    EditCitation,
    ExperienceBulletEdit,
    OnePageConstraintError,
    ProjectSwapEdit,
    ResumeEditCategory,
    ResumeEditPlan,
    ResumeEditPlanError,
    ResumeEvidenceError,
    ResumeOutputError,
    ResumeTemplateError,
    SkillEditOperation,
    SkillSectionEdit,
    SummaryEdit,
    latex_escape,
    resume_tailoring_tool,
)
from src.tools.scoring import scoring_tool

JOBS_CSV = ROOT / "data" / "AI_ML_Jobs_Dataset_20.csv"
PROFILE = ROOT / "candidate" / "profile.json"
PORTFOLIO = ROOT / "candidate" / "portfolio.json"
EVIDENCE = ROOT / "candidate" / "evidence_registry.json"
MEMORY = ROOT / "memory.json"
BASE_TEX = ROOT / "candidate" / "sample_resume.tex"
BASE_PDF = ROOT / "candidate" / "sample_resume.pdf"


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


def _job_citation(job) -> EditCitation:
    return EditCitation(
        source_type="job_posting",
        source_id=job.job_id,
        source_field="required_skills",
        supported_claim=job.required_skills_raw,
    )


def _bullet_citation(bullet_id: str, evidence_id: str) -> EditCitation:
    return EditCitation(
        source_type="experience_bullet",
        source_id=bullet_id,
        source_field="text",
        evidence_id=evidence_id,
        supported_claim=bullet_id,
    )


def _valid_plan(job, analysis, bundle) -> ResumeEditPlan:
    swap = analysis.projects.swap_suggestion
    project_swap = None
    if swap is not None:
        remove_project = next(
            project for project in bundle.all_projects()
            if project.project_id == swap.remove_project_id
        )
        add_project = next(
            project for project in bundle.all_projects()
            if project.project_id == swap.add_project_id
        )
        project_swap = ProjectSwapEdit(
            remove_project_id=remove_project.project_id,
            add_project_id=add_project.project_id,
            reason=swap.reason,
            citations=[
                _job_citation(job),
                EditCitation(
                    source_type="portfolio_project",
                    source_id=remove_project.project_id,
                    source_field="project_id",
                    evidence_id=remove_project.evidence_ids[0],
                ),
                EditCitation(
                    source_type="portfolio_project",
                    source_id=add_project.project_id,
                    source_field="project_id",
                    evidence_id=add_project.evidence_ids[0],
                ),
                EditCitation(
                    source_type="fit_analysis",
                    source_id=analysis.job_id,
                    source_field="projects.swap_suggestion",
                    supported_claim=swap.reason,
                ),
            ],
        )
    return ResumeEditPlan(
        job_id=job.job_id,
        professional_summary=SummaryEdit(
            new_text=(
                "AI Engineer with experience delivering Python retrieval-augmented APIs, "
                "embeddings, vector search, and evaluated model systems."
            ),
            reason="Emphasize evidenced fit for the AI Engineer role.",
            citations=[
                EditCitation(
                    source_type="job_posting",
                    source_id=job.job_id,
                    source_field="title",
                    supported_claim=job.title,
                ),
                EditCitation(
                    source_type="experience",
                    source_id="exp-ml-engineer-001",
                    source_field="job_title",
                    evidence_id="EV-EXP-001",
                    supported_claim="Machine Learning Engineer",
                ),
            ],
        ),
        experience_bullet_edits=[
            ExperienceBulletEdit(
                bullet_id="exp-primary-bullet-1",
                new_text=(
                    "Built Python and scikit-learn risk models over SQL data pipelines, "
                    "raising validated recall by 14% while holding the false-positive rate constant."
                ),
                reason="Preserve the supported metric while emphasizing relevant data tooling.",
                citations=[
                    _job_citation(job),
                    _bullet_citation("exp-primary-bullet-1", "EV-EXP-BULLET-001"),
                ],
            ),
            ExperienceBulletEdit(
                bullet_id="exp-primary-bullet-2",
                new_text=(
                    "Delivered a retrieval-augmented knowledge API with FastAPI, embeddings, "
                    "and vector search, reducing median analyst lookup time by 36% in a controlled pilot."
                ),
                reason="Surface the supported retrieval and API evidence.",
                citations=[
                    _job_citation(job),
                    _bullet_citation("exp-primary-bullet-2", "EV-EXP-BULLET-002"),
                ],
            ),
        ],
        skill_section_edits=[],
        project_swap=project_swap,
        plan_rationale="Apply only evidence-backed edits and the Fit Analysis project recommendation.",
    )


def _camden(workflow):
    bundle, memory, triples = workflow
    job, score, analysis = next(
        triple for triple in triples if triple[0].company == "Camden Property Trust"
    )
    assert analysis.project_swap_recommended
    return bundle, memory, job, score, analysis


def _no_swap(workflow):
    bundle, memory, triples = workflow
    job, score, analysis = next(
        triple for triple in triples if not triple[2].project_swap_recommended
    )
    return bundle, memory, job, score, analysis


def _fake_compiler(tex_path: Path, *, timeout_seconds: int = 60) -> CompilationResult:
    del timeout_seconds
    pdf_path = tex_path.with_suffix(".pdf")
    shutil.copy2(BASE_PDF, pdf_path)
    return CompilationResult(
        command=["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        return_code=0,
        pdf_path=pdf_path,
        page_count=1,
        stdout_tail="simulated successful compile",
        stderr_tail="",
    )


def _call(workflow, tmp_path: Path, plan: ResumeEditPlan, monkeypatch=None, **kwargs):
    bundle, memory, job, score, analysis = _camden(workflow)
    if monkeypatch is not None:
        monkeypatch.setattr(
            "src.tools.resume_tailoring.compile_resume_pdf",
            _fake_compiler,
        )
    return resume_tailoring_tool(
        job,
        score,
        analysis,
        bundle,
        memory,
        BASE_TEX,
        BASE_PDF,
        tmp_path,
        plan,
        **kwargs,
    )


def test_actual_camden_workflow_compiles_one_page_and_preserves_inputs(
    workflow,
    tmp_path: Path,
) -> None:
    bundle, memory, job, score, analysis = _camden(workflow)
    plan = _valid_plan(job, analysis, bundle)
    base_tex_before = BASE_TEX.read_bytes()
    base_pdf_before = BASE_PDF.read_bytes()
    dumps_before = [
        model.model_dump()
        for model in (job, score, analysis, bundle, memory, plan)
    ]

    result = _call(workflow, tmp_path, plan)

    assert result.page_count == 1
    assert result.compilation.page_count == 1
    assert result.draft_pdf_path.is_file()
    assert result.draft_tex_path.parent == tmp_path.resolve()
    assert result.project_swap_change_count == 1
    assert result.protected_regions_unchanged
    assert BASE_TEX.read_bytes() == base_tex_before
    assert BASE_PDF.read_bytes() == base_pdf_before
    assert dumps_before == [
        model.model_dump()
        for model in (job, score, analysis, bundle, memory, plan)
    ]
    assert {path.name for path in tmp_path.iterdir()} == {
        "resume_before.pdf",
        "resume_draft_r0.tex",
        "resume_draft_r0.pdf",
        "change_log_r0.json",
    }


def test_exact_edit_counts_change_log_and_real_project_rendering(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, memory, job, score, analysis = _camden(workflow)
    plan = _valid_plan(job, analysis, bundle)
    result = _call(workflow, tmp_path, plan, monkeypatch)
    tex = result.draft_tex_path.read_text(encoding="utf-8")
    log = json.loads(result.change_log_path.read_text(encoding="utf-8"))
    carepath = next(p for p in bundle.all_projects() if p.project_id == "proj-carepath-rag")

    assert result.summary_change_count == 1
    assert result.experience_bullet_change_count == 2
    assert result.skill_change_count == 0
    assert result.project_swap_change_count == 1
    assert result.change_count == 4
    assert tex.count(r"\resumeEntry{") == 7  # two education, two experience, three projects
    assert carepath.name in tex
    assert carepath.measurable_result.replace("%", r"\%") in tex
    assert "GridPulse Load Forecaster" not in tex
    assert len([c for c in log["changes"] if c["category"] == "experience_bullet"]) == 2
    assert log["deterministic_plan_digest"] == result.deterministic_plan_digest
    for change in result.changes:
        assert change.before is not None
        assert change.after
        assert change.reason
        assert change.citations


def test_plan_requires_exactly_two_distinct_bullets(workflow) -> None:
    bundle, _, job, _, analysis = _camden(workflow)
    valid = _valid_plan(job, analysis, bundle)
    with pytest.raises(ValidationError, match="exactly two"):
        ResumeEditPlan(
            **{
                **valid.model_dump(),
                "experience_bullet_edits": valid.experience_bullet_edits[:1],
            }
        )
    with pytest.raises(ValidationError, match="distinct"):
        ResumeEditPlan(
            **{
                **valid.model_dump(),
                "experience_bullet_edits": [
                    valid.experience_bullet_edits[0],
                    valid.experience_bullet_edits[0],
                ],
            }
        )
    with pytest.raises(ValidationError, match="exactly two"):
        ResumeEditPlan(
            **{
                **valid.model_dump(),
                "experience_bullet_edits": [
                    *valid.experience_bullet_edits,
                    valid.experience_bullet_edits[0].model_copy(
                        update={"bullet_id": "exp-primary-bullet-3"}
                    ),
                ],
            }
        )


@pytest.mark.parametrize(
    "bad_id",
    ["exp-primary-bullet-3", "exp-intern-bullet-1"],
)
def test_noneditable_and_internship_bullets_are_rejected(
    workflow,
    tmp_path: Path,
    bad_id: str,
) -> None:
    bundle, _, job, _, analysis = _camden(workflow)
    plan = _valid_plan(job, analysis, bundle)
    evidence_id = {
        "exp-primary-bullet-3": "EV-EXP-BULLET-003",
        "exp-intern-bullet-1": "EV-EXP-BULLET-004",
    }[bad_id]
    first = plan.experience_bullet_edits[0].model_copy(
        update={
            "bullet_id": bad_id,
            "citations": [_job_citation(job), _bullet_citation(bad_id, evidence_id)],
        }
    )
    bad_plan = plan.model_copy(
        update={"experience_bullet_edits": [first, plan.experience_bullet_edits[1]]}
    )
    with pytest.raises(ResumeEditPlanError, match="two editable primary"):
        _call(workflow, tmp_path, bad_plan)


def test_nonexistent_evidence_and_unsupported_metrics_are_rejected(
    workflow,
    tmp_path: Path,
) -> None:
    bundle, _, job, _, analysis = _camden(workflow)
    plan = _valid_plan(job, analysis, bundle)
    bad_citation = _bullet_citation("exp-primary-bullet-1", "EV-NOT-REAL")
    bad_plan = plan.model_copy(
        update={
            "experience_bullet_edits": [
                plan.experience_bullet_edits[0].model_copy(
                    update={"citations": [_job_citation(job), bad_citation]}
                ),
                plan.experience_bullet_edits[1],
            ]
        }
    )
    with pytest.raises(ResumeEvidenceError, match="Unknown evidence"):
        _call(workflow, tmp_path, bad_plan)

    numeric_plan = plan.model_copy(
        update={
            "experience_bullet_edits": [
                plan.experience_bullet_edits[0].model_copy(
                    update={"new_text": "Built Python models that improved recall by 99%."}
                ),
                plan.experience_bullet_edits[1],
            ]
        }
    )
    with pytest.raises(ResumeEvidenceError, match="unsupported numeric"):
        _call(workflow, tmp_path, numeric_plan)


def test_genuine_gap_and_target_company_claims_are_rejected(
    workflow,
    tmp_path: Path,
) -> None:
    bundle, _, job, _, analysis = _camden(workflow)
    plan = _valid_plan(job, analysis, bundle)
    gap = analysis.core_skills.genuine_gaps[0]
    gap_plan = plan.model_copy(
        update={
            "professional_summary": plan.professional_summary.model_copy(
                update={
                    "new_text": f"AI Engineer experienced with Python and {gap} systems."
                }
            )
        }
    )
    with pytest.raises(ResumeEvidenceError, match="genuine-gap"):
        _call(workflow, tmp_path, gap_plan)

    company_plan = plan.model_copy(
        update={
            "professional_summary": plan.professional_summary.model_copy(
                update={
                    "new_text": f"AI Engineer at {job.company} delivering Python systems."
                }
            )
        }
    )
    with pytest.raises(ResumeEvidenceError, match="target company"):
        _call(workflow, tmp_path, company_plan)


def test_surface_alignment_accepted_and_non_equivalent_rejected(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, _, job, _, analysis = _camden(workflow)
    plan = _valid_plan(job, analysis, bundle)
    skill_edit = SkillSectionEdit(
        operation=SkillEditOperation.SURFACE_ALIGN,
        skill="NLP",
        display_skill="Natural Language Processing",
        reason="Use the posting's supported surface form.",
        citations=[
            _job_citation(job),
            EditCitation(
                source_type="master_skill",
                source_id="master_skills.ml_and_data",
                source_field="master_skills.ml_and_data",
                evidence_id="EV-SKILL-ML",
            ),
        ],
    )
    aligned_plan = plan.model_copy(update={"skill_section_edits": [skill_edit]})
    result = _call(workflow, tmp_path, aligned_plan, monkeypatch)
    tex = result.draft_tex_path.read_text(encoding="utf-8")
    assert "Natural Language Processing" in tex
    assert result.skill_change_count == 1

    bad_edit = skill_edit.model_copy(update={"display_skill": "Computer Vision"})
    bad_plan = plan.model_copy(update={"skill_section_edits": [bad_edit]})
    with pytest.raises(ResumeEditPlanError, match="not canonically equivalent"):
        _call(workflow, tmp_path / "bad", bad_plan)


def test_duplicate_and_genuine_gap_skill_edits_are_rejected(
    workflow,
    tmp_path: Path,
) -> None:
    bundle, _, job, _, analysis = _camden(workflow)
    plan = _valid_plan(job, analysis, bundle)
    duplicate = SkillSectionEdit(
        operation=SkillEditOperation.ADD_EVIDENCED_SKILL,
        skill="Python",
        display_skill="Python",
        reason="Invalid duplicate.",
        citations=[
            _job_citation(job),
            EditCitation(
                source_type="master_skill",
                source_id="master_skills.languages",
                source_field="master_skills.languages",
                evidence_id="EV-SKILL-LANG",
            ),
        ],
    )
    duplicate_analysis = analysis.model_copy(
        update={
            "core_skills": analysis.core_skills.model_copy(
                update={
                    "evidenced_elsewhere_skills": [
                        *analysis.core_skills.evidenced_elsewhere_skills,
                        "Python",
                    ]
                }
            )
        }
    )
    with pytest.raises(ResumeEditPlanError, match="already displayed"):
        resume_tailoring_tool(
            job,
            _camden(workflow)[3],
            duplicate_analysis,
            bundle,
            _camden(workflow)[1],
            BASE_TEX,
            BASE_PDF,
            tmp_path / "duplicate",
            plan.model_copy(update={"skill_section_edits": [duplicate]}),
        )

    gap = analysis.core_skills.genuine_gaps[0]
    gap_edit = duplicate.model_copy(
        update={"skill": gap, "display_skill": gap}
    )
    with pytest.raises(ResumeEvidenceError, match="genuine-gap"):
        _call(
            workflow,
            tmp_path / "gap",
            plan.model_copy(update={"skill_section_edits": [gap_edit]}),
        )


def _memory_fact(fact_type: str = "skill") -> MemoryFact:
    return MemoryFact(
        fact_id="fact-langchain-001",
        fact_type=fact_type,
        statement="Candidate has LangChain experience.",
        normalized_value="LangChain",
        skill_tags=["LangChain"] if fact_type == "skill" else [],
        evidence_refs=["EV-PROJ-001"],
        provenance=MemoryProvenance(
            source="candidate_review",
            review_round=1,
            run_id="run-test",
            reviewer_role="reviewer",
        ),
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        applied_in_run=True,
    )


def test_memory_only_skill_requires_skill_fact(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, _, job, score, analysis = _camden(workflow)
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
    skill_edit = SkillSectionEdit(
        operation=SkillEditOperation.ADD_EVIDENCED_SKILL,
        skill="LangChain",
        display_skill="LangChain",
        category=r"ML, Data \& GenAI:",
        reason="Add a reviewed skill that is evidenced outside the base resume.",
        citations=[
            _job_citation(job_with_skill),
            EditCitation(
                source_type="memory_fact",
                source_id="fact-langchain-001",
                source_field="skill_tags",
                supported_claim="LangChain",
            ),
        ],
    )
    skill_memory = CandidateMemory(
        schema_version="1.0",
        candidate_id=bundle.profile.candidate_id,
        facts=[_memory_fact("skill")],
    )
    plan = _valid_plan(job_with_skill, analysis_with_skill, bundle).model_copy(
        update={"skill_section_edits": [skill_edit]}
    )
    monkeypatch.setattr("src.tools.resume_tailoring.compile_resume_pdf", _fake_compiler)
    result = resume_tailoring_tool(
        job_with_skill,
        score,
        analysis_with_skill,
        bundle,
        skill_memory,
        BASE_TEX,
        BASE_PDF,
        tmp_path,
        plan,
    )
    assert "LangChain" in result.draft_tex_path.read_text(encoding="utf-8")

    fact_memory = CandidateMemory(
        schema_version="1.0",
        candidate_id=bundle.profile.candidate_id,
        facts=[_memory_fact("candidate_fact")],
    )
    with pytest.raises(ResumeEvidenceError, match="No candidate evidence|not a skill"):
        resume_tailoring_tool(
            job_with_skill,
            score,
            analysis_with_skill,
            bundle,
            fact_memory,
            BASE_TEX,
            BASE_PDF,
            tmp_path / "fact",
            plan,
        )


def test_project_swap_rules_match_fit_analysis(
    workflow,
    tmp_path: Path,
) -> None:
    bundle, _, job, _, analysis = _camden(workflow)
    plan = _valid_plan(job, analysis, bundle)
    with pytest.raises(ResumeEditPlanError, match="required"):
        _call(workflow, tmp_path / "missing", plan.model_copy(update={"project_swap": None}))

    assert plan.project_swap is not None
    wrong_remove = plan.project_swap.model_copy(update={"remove_project_id": "proj-model-watch"})
    with pytest.raises(ResumeEditPlanError, match="exactly match"):
        _call(
            workflow,
            tmp_path / "wrong-remove",
            plan.model_copy(update={"project_swap": wrong_remove}),
        )
    wrong_add = plan.project_swap.model_copy(update={"add_project_id": "proj-support-nlp"})
    with pytest.raises(ResumeEditPlanError, match="exactly match"):
        _call(
            workflow,
            tmp_path / "wrong-add",
            plan.model_copy(update={"project_swap": wrong_add}),
        )

    no_bundle, no_memory, no_job, no_score, no_analysis = _no_swap(workflow)
    no_plan = _valid_plan(no_job, no_analysis, no_bundle)
    attempted = ProjectSwapEdit(
        remove_project_id="proj-grid-forecast",
        add_project_id="proj-carepath-rag",
        reason="Invalid attempted swap.",
        citations=[
            _job_citation(no_job),
            EditCitation(
                source_type="portfolio_project",
                source_id="proj-grid-forecast",
                source_field="project_id",
                evidence_id="EV-PROJ-006",
            ),
            EditCitation(
                source_type="portfolio_project",
                source_id="proj-carepath-rag",
                source_field="project_id",
                evidence_id="EV-PROJ-001",
            ),
            EditCitation(
                source_type="fit_analysis",
                source_id=no_analysis.job_id,
                source_field="projects.swap_suggestion",
            ),
        ],
    )
    with pytest.raises(ResumeEditPlanError, match="must be None"):
        resume_tailoring_tool(
            no_job,
            no_score,
            no_analysis,
            no_bundle,
            no_memory,
            BASE_TEX,
            BASE_PDF,
            tmp_path / "no-swap",
            no_plan.model_copy(update={"project_swap": attempted}),
        )


def test_latex_escape_and_required_anchor_failures(
    workflow,
    tmp_path: Path,
) -> None:
    escaped = latex_escape(r"A&B 50% $x #1 a_b {c} ~ ^ \ end")
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
        r"\textbackslash{}",
    ):
        assert token in escaped

    bundle, memory, job, score, analysis = _camden(workflow)
    plan = _valid_plan(job, analysis, bundle)
    missing_tex = tmp_path / "missing.tex"
    missing_tex.write_text(
        BASE_TEX.read_text(encoding="utf-8").replace("% AGENT-EDIT-TARGET: summary", ""),
        encoding="utf-8",
    )
    with pytest.raises(ResumeTemplateError, match="exactly once"):
        resume_tailoring_tool(
            job, score, analysis, bundle, memory,
            missing_tex, BASE_PDF, tmp_path / "missing-out", plan,
        )

    duplicate_tex = tmp_path / "duplicate.tex"
    duplicate_tex.write_text(
        BASE_TEX.read_text(encoding="utf-8")
        + "\n% AGENT-EDIT-TARGET: experience-bullet-1\n",
        encoding="utf-8",
    )
    with pytest.raises(ResumeTemplateError, match="exactly once"):
        resume_tailoring_tool(
            job, score, analysis, bundle, memory,
            duplicate_tex, BASE_PDF, tmp_path / "duplicate-out", plan,
        )


def test_revision_rules_names_and_collision_protection(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, _, job, _, analysis = _camden(workflow)
    plan = _valid_plan(job, analysis, bundle)
    with pytest.raises(ResumeEditPlanError, match="requires nonempty"):
        _call(workflow, tmp_path / "r1-missing", plan, revision_round=1)
    with pytest.raises(ResumeEditPlanError, match="0, 1, or 2"):
        _call(workflow, tmp_path / "r3", plan, revision_round=3)

    result0 = _call(workflow, tmp_path, plan, monkeypatch, revision_round=0)
    result1 = _call(
        workflow,
        tmp_path,
        plan,
        monkeypatch,
        revision_round=1,
        review_feedback="Keep the edits concise.",
    )
    assert result0.draft_tex_path.name == "resume_draft_r0.tex"
    assert result1.draft_tex_path.name == "resume_draft_r1.tex"
    assert result1.change_log_path.name == "change_log_r1.json"
    with pytest.raises(ResumeOutputError, match="overwrite"):
        _call(workflow, tmp_path, plan, monkeypatch, revision_round=0)


def test_simulated_two_page_compile_raises(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, _, job, _, analysis = _camden(workflow)
    plan = _valid_plan(job, analysis, bundle)

    def fail_two_pages(tex_path: Path, *, timeout_seconds: int = 60):
        del tex_path, timeout_seconds
        raise OnePageConstraintError("Tailored resume must be exactly one page; generated 2 pages")

    monkeypatch.setattr("src.tools.resume_tailoring.compile_resume_pdf", fail_two_pages)
    with pytest.raises(OnePageConstraintError, match="2 pages"):
        _call(workflow, tmp_path, plan)


def test_deterministic_plan_digest_and_tex_content(
    workflow,
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle, _, job, _, analysis = _camden(workflow)
    plan = _valid_plan(job, analysis, bundle)
    first = _call(workflow, tmp_path / "one", plan, monkeypatch)
    second = _call(workflow, tmp_path / "two", plan, monkeypatch)
    assert first.deterministic_plan_digest == second.deterministic_plan_digest
    assert first.draft_tex_path.read_bytes() == second.draft_tex_path.read_bytes()
    first_log = json.loads(first.change_log_path.read_text(encoding="utf-8"))
    second_log = json.loads(second.change_log_path.read_text(encoding="utf-8"))
    for log in (first_log, second_log):
        for key in (
            "draft_tex_path",
            "draft_pdf_path",
            "change_log_path",
            "base_resume_pdf_path",
            "compilation",
        ):
            log.pop(key)
    assert first_log == second_log

