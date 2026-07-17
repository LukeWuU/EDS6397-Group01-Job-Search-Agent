"""Tests for candidate bundle loading and integrity validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.candidate_loader import CandidateIntegrityError, load_candidate_bundle

PROFILE = ROOT / "candidate" / "profile.json"
PORTFOLIO = ROOT / "candidate" / "portfolio.json"
EVIDENCE = ROOT / "candidate" / "evidence_registry.json"


def test_candidate_bundle_loads_with_expected_invariants() -> None:
    """Repository candidate inputs satisfy assignment integrity rules."""
    bundle = load_candidate_bundle(PROFILE, PORTFOLIO, EVIDENCE)

    assert bundle.profile.candidate_id == "cand-mira-solenne-001"
    assert len(bundle.all_projects()) == 8
    assert len({project.domain for project in bundle.all_projects()}) >= 3
    assert len(bundle.base_resume_projects()) == 3
    assert set(bundle.profile.base_resume_project_ids) == {
        project.project_id for project in bundle.base_resume_projects()
    }

    primary = next(entry for entry in bundle.profile.experience if entry.is_primary_role)
    internship = next(entry for entry in bundle.profile.experience if not entry.is_primary_role)

    assert len(primary.bullets) == 3
    assert len(internship.bullets) == 2
    assert sum(1 for bullet in primary.bullets if bullet.editable_for_job_tailoring) == 2


def test_all_evidence_references_resolve() -> None:
    """Every referenced evidence ID exists in the registry."""
    bundle = load_candidate_bundle(PROFILE, PORTFOLIO, EVIDENCE)
    evidence_ids = set(bundle.all_evidence_ids())

    referenced: set[str] = set()
    for education in bundle.profile.education:
        referenced.update(education.evidence_ids)
    for experience in bundle.profile.experience:
        referenced.update(experience.evidence_ids)
        for bullet in experience.bullets:
            referenced.update(bullet.evidence_ids)
    for project in bundle.all_projects():
        referenced.update(project.evidence_ids)

    assert referenced.issubset(evidence_ids)


def test_every_master_skill_has_evidence() -> None:
    """Each master skill is supported by at least one evidence record."""
    bundle = load_candidate_bundle(PROFILE, PORTFOLIO, EVIDENCE)

    for skill in bundle.all_master_skills():
        assert bundle.get_skill_evidence(skill), f"missing evidence for {skill!r}"


def test_swap_available_projects_are_returned_correctly() -> None:
    """Non-base-resume projects are available for swapping."""
    bundle = load_candidate_bundle(PROFILE, PORTFOLIO, EVIDENCE)

    swap_ids = {project.project_id for project in bundle.swap_available_projects()}
    base_ids = {project.project_id for project in bundle.base_resume_projects()}

    assert len(swap_ids) == 5
    assert swap_ids.isdisjoint(base_ids)
    assert swap_ids == {
        "proj-carepath-rag",
        "proj-vision-inspect",
        "proj-ledger-anomaly",
        "proj-catalog-recs",
        "proj-support-nlp",
    }
    assert "proj-grid-forecast" in base_ids
    assert "proj-carepath-rag" in swap_ids


def test_cloud_and_mlops_alias_loads_from_json_key() -> None:
    """The cloud_and_mLOps JSON key maps to a conventional Python attribute."""
    bundle = load_candidate_bundle(PROFILE, PORTFOLIO, EVIDENCE)

    assert bundle.profile.master_skills.cloud_and_mlops == [
        "AWS",
        "MLOps",
        "Docker",
        "MLflow",
        "Model Monitoring",
        "CI/CD",
    ]


def test_dangling_evidence_reference_raises_integrity_error(
    tmp_path: Path,
) -> None:
    """Dangling evidence references fail validation."""
    import json

    profile_data = json.loads(PROFILE.read_text(encoding="utf-8"))
    profile_data["education"][0]["evidence_ids"] = ["EV-MISSING"]
    bad_profile = tmp_path / "profile.json"
    bad_profile.write_text(json.dumps(profile_data), encoding="utf-8")

    with pytest.raises(CandidateIntegrityError, match="dangling evidence references"):
        load_candidate_bundle(bad_profile, PORTFOLIO, EVIDENCE)
