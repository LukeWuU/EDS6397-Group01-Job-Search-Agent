"""Combined candidate bundle with deterministic lookup helpers."""

from __future__ import annotations

from pydantic import BaseModel

from src.models.candidate import CandidateProfile, PortfolioProject, ProjectPortfolio
from src.models.evidence import EvidenceRecord, EvidenceRegistry


class CandidateBundle(BaseModel):
    """Validated candidate profile, portfolio, and evidence registry."""

    profile: CandidateProfile
    portfolio: ProjectPortfolio
    evidence: EvidenceRegistry

    def all_master_skills(self) -> list[str]:
        """Return all master skills as a deduplicated ordered list."""
        skills: list[str] = []
        seen: set[str] = set()
        categories = (
            self.profile.master_skills.languages,
            self.profile.master_skills.ml_and_data,
            self.profile.master_skills.generative_ai,
            self.profile.master_skills.cloud_and_mlops,
            self.profile.master_skills.systems_and_tools,
        )
        for category in categories:
            for skill in category:
                key = skill.casefold()
                if key in seen:
                    continue
                seen.add(key)
                skills.append(skill)
        return skills

    def all_projects(self) -> list[PortfolioProject]:
        """Return all portfolio projects in source order."""
        return list(self.portfolio.projects)

    def base_resume_projects(self) -> list[PortfolioProject]:
        """Return projects currently placed on the base resume."""
        return [project for project in self.portfolio.projects if project.on_base_resume]

    def swap_available_projects(self) -> list[PortfolioProject]:
        """Return portfolio projects available for resume swapping."""
        return [project for project in self.portfolio.projects if not project.on_base_resume]

    def all_evidence_ids(self) -> list[str]:
        """Return all evidence IDs in registry order."""
        return [record.evidence_id for record in self.evidence.evidence_records]

    def get_evidence(self, evidence_id: str) -> EvidenceRecord | None:
        """Look up a single evidence record by ID."""
        for record in self.evidence.evidence_records:
            if record.evidence_id == evidence_id:
                return record
        return None

    def get_skill_evidence(self, skill: str) -> list[EvidenceRecord]:
        """Return evidence records that support a given skill."""
        target = skill.casefold()
        return [
            record
            for record in self.evidence.evidence_records
            if any(supported.casefold() == target for supported in record.supported_skills)
        ]
