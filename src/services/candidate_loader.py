"""Deterministic candidate bundle loader with integrity validation."""

from __future__ import annotations

import json
from pathlib import Path

from src.models.bundle import CandidateBundle
from src.models.candidate import CandidateProfile, ProjectPortfolio
from src.models.evidence import EvidenceRegistry


class CandidateIntegrityError(Exception):
    """Raised when candidate profile, portfolio, or evidence data is inconsistent."""


def _load_json(path: Path) -> dict:
    """Load a UTF-8 JSON document."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_referenced_evidence_ids(bundle: CandidateBundle) -> set[str]:
    """Collect every evidence ID referenced by profile and portfolio records."""
    referenced: set[str] = set()

    for education in bundle.profile.education:
        referenced.update(education.evidence_ids)
    for experience in bundle.profile.experience:
        referenced.update(experience.evidence_ids)
        for bullet in experience.bullets:
            referenced.update(bullet.evidence_ids)
    for project in bundle.portfolio.projects:
        referenced.update(project.evidence_ids)

    return referenced


def _validate_candidate_bundle(bundle: CandidateBundle) -> None:
    """Apply assignment-specific candidate integrity checks."""
    profile = bundle.profile
    portfolio = bundle.portfolio
    evidence = bundle.evidence

    if profile.candidate_id != portfolio.candidate_id:
        raise CandidateIntegrityError("candidate_id mismatch between profile and portfolio")
    if profile.candidate_id != evidence.candidate_id:
        raise CandidateIntegrityError("candidate_id mismatch between profile and evidence")

    if len(portfolio.projects) < 6:
        raise CandidateIntegrityError("portfolio must contain at least 6 projects")

    domains = {project.domain for project in portfolio.projects}
    if len(domains) < 2:
        raise CandidateIntegrityError("portfolio must cover at least 2 domains")

    base_resume_projects = bundle.base_resume_projects()
    if len(base_resume_projects) != 3:
        raise CandidateIntegrityError("portfolio must contain exactly 3 base-resume projects")

    if len(profile.base_resume_project_ids) != 3:
        raise CandidateIntegrityError(
            "profile.base_resume_project_ids must contain exactly 3 project IDs"
        )

    flagged_ids = {project.project_id for project in base_resume_projects}
    profile_ids = set(profile.base_resume_project_ids)
    if flagged_ids != profile_ids:
        raise CandidateIntegrityError(
            "base_resume_project_ids must match projects marked on_base_resume"
        )

    primary_roles = [entry for entry in profile.experience if entry.is_primary_role]
    internships = [entry for entry in profile.experience if not entry.is_primary_role]

    if len(primary_roles) != 1:
        raise CandidateIntegrityError("profile must contain exactly one primary role")
    if len(internships) != 1:
        raise CandidateIntegrityError("profile must contain exactly one internship role")

    primary_bullets = primary_roles[0].bullets
    intern_bullets = internships[0].bullets

    if len(primary_bullets) != 3:
        raise CandidateIntegrityError("primary role must contain exactly 3 bullets")
    if len(intern_bullets) != 2:
        raise CandidateIntegrityError("internship role must contain exactly 2 bullets")

    editable_primary = [
        bullet for bullet in primary_bullets if bullet.editable_for_job_tailoring
    ]
    if len(editable_primary) != 2:
        raise CandidateIntegrityError(
            "primary role must contain exactly 2 editable bullets"
        )

    evidence_ids = bundle.all_evidence_ids()
    if len(evidence_ids) != len(set(evidence_ids)):
        raise CandidateIntegrityError("evidence IDs must be unique")

    evidence_lookup = {record.evidence_id: record for record in evidence.evidence_records}
    referenced_ids = _collect_referenced_evidence_ids(bundle)

    dangling = referenced_ids - set(evidence_lookup)
    if dangling:
        raise CandidateIntegrityError(
            "dangling evidence references found: " + ", ".join(sorted(dangling))
        )

    for skill in bundle.all_master_skills():
        if not bundle.get_skill_evidence(skill):
            raise CandidateIntegrityError(
                f"master skill {skill!r} has no supporting evidence"
            )


def load_candidate_bundle(
    profile_path: Path,
    portfolio_path: Path,
    evidence_path: Path,
) -> CandidateBundle:
    """Load and validate the candidate profile, portfolio, and evidence registry."""
    profile = CandidateProfile.model_validate(_load_json(profile_path))
    portfolio = ProjectPortfolio.model_validate(_load_json(portfolio_path))
    evidence = EvidenceRegistry.model_validate(_load_json(evidence_path))

    bundle = CandidateBundle(profile=profile, portfolio=portfolio, evidence=evidence)
    _validate_candidate_bundle(bundle)
    return bundle
