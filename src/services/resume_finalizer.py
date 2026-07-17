"""Copy an explicitly approved tailored resume into its final job directory."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader

from src.tools.resume_tailoring import ResumeTailoringResult


class FinalizationError(Exception):
    """Raised when an approved draft cannot be finalized safely."""


class UnapprovedFinalizationError(FinalizationError):
    """Raised when finalization lacks explicit workflow approval."""


class FinalizedResumeResult(BaseModel):
    """Exact copied artifacts and hashes for one approved resume revision."""

    job_id: str
    title: str
    company: str
    approved_revision_round: int
    destination_dir: Path
    source_base_pdf_path: Path
    source_draft_tex_path: Path
    source_draft_pdf_path: Path
    source_change_log_path: Path
    resume_before_path: Path
    resume_after_tex_path: Path
    resume_after_pdf_path: Path
    resume_change_log_path: Path
    page_count: int
    copied_file_sha256: dict[str, str]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_approved_draft(draft: ResumeTailoringResult) -> None:
    if not draft.protected_regions_unchanged:
        raise FinalizationError("Protected-region verification failed for approved draft")
    if (
        draft.protected_regions_sha256_before
        != draft.protected_regions_sha256_after
    ):
        raise FinalizationError("Protected-region hashes do not match")
    sources = (
        draft.base_resume_pdf_path,
        draft.draft_tex_path,
        draft.draft_pdf_path,
        draft.change_log_path,
    )
    symbolic = [path for path in sources if path.is_symlink()]
    if symbolic:
        raise FinalizationError(
            "Approved resume sources must not be symbolic links: "
            + ", ".join(str(path) for path in symbolic)
        )
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FinalizationError(
            "Approved resume source file is missing: "
            + ", ".join(str(path) for path in missing)
        )
    if _sha256(draft.draft_tex_path) != draft.tailored_tex_sha256:
        raise FinalizationError("Approved draft LaTeX hash no longer matches its result")
    try:
        change_log = json.loads(draft.change_log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"Approved change log is unreadable: {exc}") from exc
    required_log_values = {
        "job_id": draft.job_id,
        "revision_round": draft.revision_round,
        "tailored_tex_sha256": draft.tailored_tex_sha256,
        "protected_regions_unchanged": True,
    }
    for key, expected in required_log_values.items():
        if change_log.get(key) != expected:
            raise FinalizationError(
                f"Approved change log field {key!r} does not match the draft result"
            )
    try:
        source_pages = len(PdfReader(str(draft.draft_pdf_path)).pages)
    except Exception as exc:
        raise FinalizationError("Approved draft PDF is unreadable") from exc
    if source_pages != 1:
        raise FinalizationError(
            f"Approved draft PDF must be exactly one page; found {source_pages}"
        )


def finalize_approved_resume(
    approved_draft: ResumeTailoringResult,
    destination_dir: Path,
    *,
    approved: bool = False,
) -> FinalizedResumeResult:
    """Copy an explicitly approved one-page draft without recompiling it."""
    if not approved:
        raise UnapprovedFinalizationError(
            "Resume finalization requires explicit approval from Human Review"
        )
    _validate_approved_draft(approved_draft)

    destination_dir = destination_dir.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    targets = {
        "resume_before.pdf": destination_dir / "resume_before.pdf",
        "resume_after.tex": destination_dir / "resume_after.tex",
        "resume_after.pdf": destination_dir / "resume_after.pdf",
        "resume_change_log.json": destination_dir / "resume_change_log.json",
    }
    collisions = [path for path in targets.values() if path.exists()]
    if collisions:
        raise FinalizationError(
            "Refusing to overwrite existing final resume file: "
            + ", ".join(str(path) for path in collisions)
        )

    sources = {
        "resume_before.pdf": approved_draft.base_resume_pdf_path,
        "resume_after.tex": approved_draft.draft_tex_path,
        "resume_after.pdf": approved_draft.draft_pdf_path,
        "resume_change_log.json": approved_draft.change_log_path,
    }
    try:
        for name in targets:
            shutil.copyfile(sources[name], targets[name], follow_symlinks=True)
    except OSError as exc:
        raise FinalizationError(f"Failed to copy approved resume artifacts: {exc}") from exc

    for name in targets:
        if _sha256(targets[name]) != _sha256(sources[name]):
            raise FinalizationError(f"Copied bytes do not match source for {name}")
    try:
        page_count = len(PdfReader(str(targets["resume_after.pdf"])).pages)
    except Exception as exc:
        raise FinalizationError("Final resume_after.pdf is unreadable") from exc
    if page_count != 1:
        raise FinalizationError(
            f"Final resume_after.pdf must be exactly one page; found {page_count}"
        )

    return FinalizedResumeResult(
        job_id=approved_draft.job_id,
        title=approved_draft.title,
        company=approved_draft.company,
        approved_revision_round=approved_draft.revision_round,
        destination_dir=destination_dir,
        source_base_pdf_path=approved_draft.base_resume_pdf_path,
        source_draft_tex_path=approved_draft.draft_tex_path,
        source_draft_pdf_path=approved_draft.draft_pdf_path,
        source_change_log_path=approved_draft.change_log_path,
        resume_before_path=targets["resume_before.pdf"],
        resume_after_tex_path=targets["resume_after.tex"],
        resume_after_pdf_path=targets["resume_after.pdf"],
        resume_change_log_path=targets["resume_change_log.json"],
        page_count=page_count,
        copied_file_sha256={name: _sha256(path) for name, path in targets.items()},
    )


__all__ = [
    "FinalizationError",
    "FinalizedResumeResult",
    "UnapprovedFinalizationError",
    "finalize_approved_resume",
]
