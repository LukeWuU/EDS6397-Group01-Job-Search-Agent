"""Focused tests for approved resume finalization."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.resume_finalizer import (
    FinalizationError,
    UnapprovedFinalizationError,
    finalize_approved_resume,
)
from src.tools.resume_tailoring import (
    CompilationResult,
    ResumeChange,
    ResumeEditCategory,
    ResumeTailoringResult,
)

BASE_PDF = ROOT / "candidate" / "sample_resume.pdf"


def make_draft(
    directory: Path,
    job_id: str = "job-a",
    revision_round: int = 0,
    feedback: str | None = None,
) -> ResumeTailoringResult:
    directory.mkdir(parents=True, exist_ok=True)
    before_pdf = directory / "resume_before.pdf"
    draft_pdf = directory / f"resume_draft_r{revision_round}.pdf"
    draft_tex = directory / f"resume_draft_r{revision_round}.tex"
    change_log = directory / f"change_log_r{revision_round}.json"
    shutil.copyfile(BASE_PDF, before_pdf)
    shutil.copyfile(BASE_PDF, draft_pdf)
    draft_tex.write_text(f"% tailored {job_id} r{revision_round}\n", encoding="utf-8")
    tex_hash = hashlib.sha256(draft_tex.read_bytes()).hexdigest()
    protected_hash = hashlib.sha256(b"protected").hexdigest()
    changes = [
        ResumeChange(
            category=ResumeEditCategory.PROFESSIONAL_SUMMARY,
            target_id="summary",
            before="before",
            after="after",
            reason="reason",
            citations=[],
        ),
        ResumeChange(
            category=ResumeEditCategory.EXPERIENCE_BULLET,
            target_id="exp-primary-bullet-1",
            before="before 1",
            after="after 1",
            reason="reason",
            citations=[],
        ),
        ResumeChange(
            category=ResumeEditCategory.EXPERIENCE_BULLET,
            target_id="exp-primary-bullet-2",
            before="before 2",
            after="after 2",
            reason="reason",
            citations=[],
        ),
    ]
    result = ResumeTailoringResult(
        job_id=job_id,
        title="AI Engineer",
        company="Fictional Company",
        revision_round=revision_round,
        review_feedback=feedback,
        edit_categories=[
            ResumeEditCategory.PROFESSIONAL_SUMMARY,
            ResumeEditCategory.EXPERIENCE_BULLET,
        ],
        changes=changes,
        change_count=3,
        summary_change_count=1,
        experience_bullet_change_count=2,
        skill_change_count=0,
        project_swap_change_count=0,
        base_resume_pdf_path=before_pdf,
        draft_tex_path=draft_tex,
        draft_pdf_path=draft_pdf,
        change_log_path=change_log,
        compilation=CompilationResult(
            command=["pdflatex"],
            return_code=0,
            pdf_path=draft_pdf,
            page_count=1,
            stdout_tail="",
            stderr_tail="",
        ),
        page_count=1,
        evidence_citation_count=0,
        base_tex_sha256=hashlib.sha256(b"base").hexdigest(),
        tailored_tex_sha256=tex_hash,
        protected_regions_sha256_before=protected_hash,
        protected_regions_sha256_after=protected_hash,
        protected_regions_unchanged=True,
        deterministic_plan_digest=hashlib.sha256(job_id.encode()).hexdigest(),
    )
    change_log.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def test_finalization_requires_explicit_approval(tmp_path: Path) -> None:
    draft = make_draft(tmp_path / "source")
    with pytest.raises(UnapprovedFinalizationError):
        finalize_approved_resume(draft, tmp_path / "final")


def test_finalization_copies_exactly_four_files_and_one_page(tmp_path: Path) -> None:
    draft = make_draft(tmp_path / "source", revision_round=1, feedback="Approved update")
    result = finalize_approved_resume(draft, tmp_path / "final", approved=True)
    assert result.approved_revision_round == 1
    assert result.page_count == 1
    assert {path.name for path in result.destination_dir.iterdir()} == {
        "resume_before.pdf",
        "resume_after.tex",
        "resume_after.pdf",
        "resume_change_log.json",
    }
    source_by_name = {
        "resume_before.pdf": draft.base_resume_pdf_path,
        "resume_after.tex": draft.draft_tex_path,
        "resume_after.pdf": draft.draft_pdf_path,
        "resume_change_log.json": draft.change_log_path,
    }
    for name, source in source_by_name.items():
        assert (result.destination_dir / name).read_bytes() == source.read_bytes()
        assert result.copied_file_sha256[name] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_existing_file_is_not_overwritten(tmp_path: Path) -> None:
    draft = make_draft(tmp_path / "source")
    destination = tmp_path / "final"
    destination.mkdir()
    existing = destination / "resume_after.pdf"
    existing.write_bytes(b"keep")
    with pytest.raises(FinalizationError, match="overwrite"):
        finalize_approved_resume(draft, destination, approved=True)
    assert existing.read_bytes() == b"keep"


def test_protected_hash_missing_pdf_and_log_fail_clearly(tmp_path: Path) -> None:
    draft = make_draft(tmp_path / "protected")
    unsafe = draft.model_copy(update={"protected_regions_unchanged": False})
    with pytest.raises(FinalizationError, match="Protected-region"):
        finalize_approved_resume(unsafe, tmp_path / "unsafe-final", approved=True)

    missing_pdf = make_draft(tmp_path / "missing-pdf")
    missing_pdf.draft_pdf_path.unlink()
    with pytest.raises(FinalizationError, match="missing"):
        finalize_approved_resume(missing_pdf, tmp_path / "missing-pdf-final", approved=True)

    missing_log = make_draft(tmp_path / "missing-log")
    missing_log.change_log_path.unlink()
    with pytest.raises(FinalizationError, match="missing"):
        finalize_approved_resume(missing_log, tmp_path / "missing-log-final", approved=True)
