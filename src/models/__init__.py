"""Typed domain models for jobs, candidates, evidence, and memory."""

from src.models.bundle import CandidateBundle
from src.models.candidate import (
    CandidateProfile,
    Education,
    ExperienceBullet,
    ExperienceEntry,
    MasterSkills,
    Persona,
    PortfolioProject,
    Preferences,
    ProjectPortfolio,
)
from src.models.evidence import EvidenceRecord, EvidenceRegistry
from src.models.job import ExperienceParseStatus, Job, WorkMode
from src.models.memory import CandidateMemory, MemoryFact, MemoryProvenance

__all__ = [
    "CandidateBundle",
    "CandidateMemory",
    "CandidateProfile",
    "Education",
    "EvidenceRecord",
    "EvidenceRegistry",
    "ExperienceBullet",
    "ExperienceEntry",
    "ExperienceParseStatus",
    "Job",
    "MasterSkills",
    "MemoryFact",
    "MemoryProvenance",
    "Persona",
    "PortfolioProject",
    "Preferences",
    "ProjectPortfolio",
    "WorkMode",
]
