"""Typed candidate profile and portfolio models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Persona(BaseModel):
    """Fictional candidate identity and contact details."""

    full_name: str
    city: str
    state: str
    phone: str
    email: str
    github: str
    fictional: bool


class Preferences(BaseModel):
    """Candidate job-search preferences."""

    preferred_locations: list[str]
    remote_preference: str
    onsite_policy: str
    years_of_experience: int
    excluded_companies: list[str]
    target_job_titles: list[str]
    target_domains: list[str]


class Education(BaseModel):
    """Education entry with evidence references."""

    education_id: str
    institution: str
    location: str
    degree: str
    graduation_date: str
    gpa: str | None = None
    evidence_ids: list[str]


class ExperienceBullet(BaseModel):
    """Single resume bullet with editability metadata."""

    bullet_id: str
    text: str
    editable_for_job_tailoring: bool = False
    immutable_for_job_tailoring: bool = False
    evidence_ids: list[str]


class ExperienceEntry(BaseModel):
    """Professional or internship experience entry."""

    experience_id: str
    job_title: str
    employer: str
    location: str
    start_date: str
    end_date: str
    is_primary_role: bool
    fictional_employer: bool
    evidence_ids: list[str]
    bullets: list[ExperienceBullet]


class MasterSkills(BaseModel):
    """Master skills grouped by category."""

    model_config = ConfigDict(populate_by_name=True)

    languages: list[str]
    ml_and_data: list[str]
    generative_ai: list[str]
    cloud_and_mlops: list[str] = Field(alias="cloud_and_mLOps")
    systems_and_tools: list[str]


class EvidenceSummaryCategory(BaseModel):
    """Evidence ID grouping used in the profile summary."""

    category: str
    evidence_ids: list[str]


class CandidateProfile(BaseModel):
    """Full candidate profile loaded from profile.json."""

    schema_version: str
    candidate_id: str
    persona: Persona
    preferences: Preferences
    education: list[Education]
    experience: list[ExperienceEntry]
    master_skills: MasterSkills
    base_resume_project_ids: list[str]
    evidence_summary: list[EvidenceSummaryCategory]


class PortfolioProject(BaseModel):
    """Portfolio project with resume placement metadata."""

    project_id: str
    name: str
    year: int
    domain: str
    industry: str
    short_description: str
    problem_solved: str
    technology_stack: list[str]
    skills_demonstrated: list[str]
    measurable_result: str
    evidence_ids: list[str]
    on_base_resume: bool


class ProjectPortfolio(BaseModel):
    """Project portfolio loaded from portfolio.json."""

    schema_version: str
    candidate_id: str
    projects: list[PortfolioProject]
