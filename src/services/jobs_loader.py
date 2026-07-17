"""Deterministic CSV job loader."""

from __future__ import annotations

import csv
from pathlib import Path

from src.models.job import (
    Job,
    derive_job_id,
    parse_experience_requirement,
    parse_required_skills,
    parse_work_mode,
)

CSV_HEADERS: dict[str, str] = {
    "Job Title": "title",
    "Company": "company",
    "Industry/Domain": "industry_domain",
    "Location": "location_raw",
    "Required Skills": "required_skills_raw",
    "Years of Experience Required": "experience_requirement_raw",
    "Job Description (10-20 lines; include responsibilities and qualifications)": "job_description",
    "Company Details (2-3 lines about the company; used for cover letters)": "company_details",
    "URL": "url",
}

REQUIRED_FIELDS = tuple(CSV_HEADERS.values())


class JobsLoaderError(Exception):
    """Raised when the jobs CSV cannot be loaded or validated."""


def _require_field(row_number: int, field_name: str, value: str | None) -> str:
    """Return a trimmed required field or raise a row-specific error."""
    if value is None or not str(value).strip():
        raise JobsLoaderError(
            f"Row {row_number}: missing required value for {field_name!r}"
        )
    return str(value).strip()


def load_jobs(path: Path) -> list[Job]:
    """Load and validate all jobs from the assignment CSV dataset."""
    if not path.is_file():
        raise JobsLoaderError(f"Jobs CSV not found: {path}")

    jobs: list[Job] = []
    seen_urls: set[str] = set()
    seen_job_ids: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise JobsLoaderError("Jobs CSV is missing a header row")

        missing_headers = [header for header in CSV_HEADERS if header not in reader.fieldnames]
        if missing_headers:
            raise JobsLoaderError(
                "Jobs CSV is missing required columns: "
                + ", ".join(missing_headers)
            )

        for row_index, row in enumerate(reader, start=2):
            mapped = {
                model_field: _require_field(row_index, csv_header, row.get(csv_header))
                for csv_header, model_field in CSV_HEADERS.items()
            }

            url = mapped["url"]
            if url in seen_urls:
                raise JobsLoaderError(f"Row {row_index}: duplicate URL {url!r}")
            seen_urls.add(url)

            job_id = derive_job_id(url)
            if job_id in seen_job_ids:
                raise JobsLoaderError(f"Row {row_index}: duplicate job_id {job_id!r}")
            seen_job_ids.add(job_id)

            required_skills = parse_required_skills(mapped["required_skills_raw"])
            if not required_skills:
                raise JobsLoaderError(
                    f"Row {row_index}: required skills parsed to an empty list"
                )

            status, minimum_years, year_values = parse_experience_requirement(
                mapped["experience_requirement_raw"]
            )

            jobs.append(
                Job(
                    job_id=job_id,
                    title=mapped["title"],
                    company=mapped["company"],
                    industry_domain=mapped["industry_domain"],
                    location_raw=mapped["location_raw"],
                    work_mode=parse_work_mode(mapped["location_raw"]),
                    required_skills_raw=mapped["required_skills_raw"],
                    required_skills=required_skills,
                    experience_requirement_raw=mapped["experience_requirement_raw"],
                    minimum_years=minimum_years,
                    experience_parse_status=status,
                    experience_year_values=year_values,
                    job_description=mapped["job_description"],
                    company_details=mapped["company_details"],
                    url=url,
                    source_row=row_index,
                )
            )

    return jobs
