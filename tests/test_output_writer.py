"""Focused tests for deterministic final-folder packaging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.services.candidate_loader import load_candidate_bundle
from src.services.jobs_loader import load_jobs
from src.services.memory_loader import load_memory
from src.services.output_writer import (
    OutputPackagingError,
    write_fit_analysis_files,
    write_job_details,
)
from src.tools.filtering import filtering_tool
from src.tools.fit_analysis import fit_analysis_tool
from src.tools.scoring import scoring_tool

ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    bundle = load_candidate_bundle(
        ROOT / "candidate/profile.json",
        ROOT / "candidate/portfolio.json",
        ROOT / "candidate/evidence_registry.json",
    )
    memory = load_memory(ROOT / "memory.json", bundle.profile.candidate_id)
    jobs = load_jobs(ROOT / "data/AI_ML_Jobs_Dataset_20.csv")
    accepted = filtering_tool(jobs, bundle.profile).accepted_jobs
    score = scoring_tool(accepted, bundle, memory).top_3[0]
    job = next(item for item in accepted if item.job_id == score.job_id)
    analysis = fit_analysis_tool(
        job,
        score,
        bundle,
        memory,
        ROOT / "candidate/sample_resume.tex",
    )
    return job, score, analysis


def test_writes_complete_utf8_files_inside_supplied_folder(tmp_path):
    job, score, analysis = _inputs()
    folder = tmp_path / "job-safe"
    details = write_job_details(folder, job, score)
    text_path, json_path = write_fit_analysis_files(folder, analysis)

    payload = json.loads(details.read_text(encoding="utf-8"))
    assert payload["job_id"] == job.job_id
    assert payload["deterministic_score"]["rank"] == score.rank
    assert payload["deterministic_score"]["final_score"] == score.final_score
    assert payload["deterministic_score"]["breakdown"] == score.breakdown.model_dump(
        mode="json"
    )
    for path in (details, text_path, json_path):
        assert path.parent == folder.resolve()
        raw = path.read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf")
        assert raw.endswith(b"\n")


def test_refuses_overwrite_without_touching_existing_file(tmp_path):
    job, score, analysis = _inputs()
    folder = tmp_path / "job-safe"
    path = write_job_details(folder, job, score)
    before = path.read_bytes()
    with pytest.raises(OutputPackagingError, match="overwrite"):
        write_job_details(folder, job, score)
    assert path.read_bytes() == before

    write_fit_analysis_files(folder, analysis)
    with pytest.raises(OutputPackagingError, match="overwrite"):
        write_fit_analysis_files(folder, analysis)
    assert not (tmp_path / "fit_analysis.json").exists()
