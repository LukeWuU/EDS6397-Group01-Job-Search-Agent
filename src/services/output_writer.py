"""Deterministic final-folder packaging for job and fit-analysis records."""

from __future__ import annotations

import json
from pathlib import Path

from src.models.job import Job
from src.tools.fit_analysis import FitAnalysisResult
from src.tools.scoring import JobScore


class OutputPackagingError(Exception):
    """Raised when final output packaging would be unsafe or destructive."""


def _safe_target(folder: Path, filename: str) -> Path:
    if folder.is_symlink():
        raise OutputPackagingError("Final job folder must not be a symbolic link")
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    target = (folder / filename).resolve()
    if target.parent != folder:
        raise OutputPackagingError("Output path escapes the supplied final job folder")
    if target.exists() or target.is_symlink():
        raise OutputPackagingError(f"Refusing to overwrite existing output: {target}")
    return target


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_job_details(
    final_job_folder: Path,
    job: Job,
    job_score: JobScore,
) -> Path:
    """Write complete posting details and deterministic score information."""
    if job.job_id != job_score.job_id:
        raise OutputPackagingError("Job and deterministic score IDs do not match")
    path = _safe_target(final_job_folder, "job_details.json")
    payload = {
        "job_id": job.job_id,
        "title": job.title,
        "company": job.company,
        "industry_domain": job.industry_domain,
        "location": job.location_raw,
        "required_skills": job.required_skills,
        "experience_requirement": job.experience_requirement_raw,
        "job_description": job.job_description,
        "company_details": job.company_details,
        "url": job.url,
        "deterministic_score": {
            "rank": job_score.rank,
            "final_score": job_score.final_score,
            "breakdown": job_score.breakdown.model_dump(mode="json"),
        },
    }
    _write_json(path, payload)
    return path


def write_fit_analysis_files(
    final_job_folder: Path,
    fit_analysis: FitAnalysisResult,
) -> tuple[Path, Path]:
    """Write the user-facing and structured Fit Analysis representations."""
    text_path = _safe_target(final_job_folder, "fit_analysis.txt")
    json_path = _safe_target(final_job_folder, "fit_analysis.json")
    text_path.write_text(
        fit_analysis.formatted_text.rstrip("\n") + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_json(json_path, fit_analysis.model_dump(mode="json"))
    return text_path, json_path


__all__ = [
    "OutputPackagingError",
    "write_fit_analysis_files",
    "write_job_details",
]
