"""Read-only local preflight checks for the production CLI."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import ollama
from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader

from src.config import AppConfig
from src.services.candidate_loader import load_candidate_bundle
from src.services.jobs_loader import load_jobs
from src.services.memory_loader import load_memory


class PreflightStatus(StrEnum):
    """Allowed read-only check outcomes."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class PreflightCheck(BaseModel):
    """One named preflight outcome."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: PreflightStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class PreflightResult(BaseModel):
    """Complete read-only preflight report."""

    model_config = ConfigDict(extra="forbid")

    repository_root: Path
    checks: list[PreflightCheck]
    candidate_id: str | None = None
    job_count: int | None = None
    resume_page_count: int | None = None
    ollama_model: str
    output_root: Path
    output_root_empty: bool | None = None

    @property
    def passed_count(self) -> int:
        return sum(check.status == PreflightStatus.PASS for check in self.checks)

    @property
    def warning_count(self) -> int:
        return sum(check.status == PreflightStatus.WARNING for check in self.checks)

    @property
    def failed_count(self) -> int:
        return sum(check.status == PreflightStatus.FAIL for check in self.checks)

    @property
    def succeeded(self) -> bool:
        return self.failed_count == 0


class PathSafetyError(ValueError):
    """Raised when writable paths violate repository isolation rules."""


_PROTECTED_OUTPUT_DIRECTORIES = (
    "candidate",
    "data",
    "src",
    "tests",
    ".git",
    ".venv",
)

_COMPLETED_JOB_FILENAMES = {
    "job_details.json",
    "resume_after.pdf",
    "resume_change_log.json",
    "cover_letter.pdf",
    "cover_letter_evidence.json",
}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_run_paths(
    repository_root: Path,
    output_root: Path,
    memory_path: Path,
) -> tuple[Path, Path]:
    """Resolve and validate writable paths without creating or deleting anything."""
    repository_root = repository_root.resolve()
    raw_output = output_root.expanduser()
    if raw_output.is_symlink():
        raise PathSafetyError("Output root must not be a symbolic link")
    resolved_output = raw_output.resolve()
    resolved_memory = memory_path.expanduser().resolve()

    if resolved_output == repository_root:
        raise PathSafetyError("Output root must not equal the repository root")
    if not _is_relative_to(resolved_output, repository_root):
        raise PathSafetyError(
            "Output root must remain inside the chosen repository/run area"
        )
    for directory_name in _PROTECTED_OUTPUT_DIRECTORIES:
        protected = (repository_root / directory_name).resolve()
        if resolved_output == protected or _is_relative_to(resolved_output, protected):
            raise PathSafetyError(
                f"Output root must not equal or be nested inside {directory_name}/"
            )
    default_outputs = (repository_root / "outputs").resolve()
    if resolved_memory == default_outputs or _is_relative_to(
        resolved_memory, default_outputs
    ):
        raise PathSafetyError("Memory path must not be inside outputs/")
    if resolved_memory == resolved_output or _is_relative_to(
        resolved_memory, resolved_output
    ):
        raise PathSafetyError("Memory path must not be inside the output root")
    return resolved_output, resolved_memory


def find_completed_output_files(output_root: Path) -> list[Path]:
    """Return completed-job artifacts that the runtime must never overwrite."""
    if not output_root.exists():
        return []
    if output_root.is_symlink() or not output_root.is_dir():
        return [output_root]
    collisions: list[Path] = []
    for child in sorted(output_root.iterdir(), key=lambda item: item.name.casefold()):
        if not child.is_dir() or not child.name.startswith("job-"):
            continue
        for filename in sorted(_COMPLETED_JOB_FILENAMES):
            candidate = child / filename
            if candidate.exists():
                collisions.append(candidate)
    return collisions


def _check(
    name: str,
    status: PreflightStatus,
    message: str,
    **details: Any,
) -> PreflightCheck:
    return PreflightCheck(
        name=name,
        status=status,
        message=message,
        details=details,
    )


def _response_content(response: Any) -> str:
    message = (
        response.get("message")
        if isinstance(response, dict)
        else getattr(response, "message", None)
    )
    if message is None:
        return ""
    content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", "")
    )
    return content if isinstance(content, str) else ""


def _installed_model_names(response: Any) -> list[str]:
    models = (
        response.get("models", [])
        if isinstance(response, dict)
        else getattr(response, "models", [])
    )
    names: list[str] = []
    for model in models or []:
        name = (
            model.get("model") or model.get("name")
            if isinstance(model, dict)
            else getattr(model, "model", None) or getattr(model, "name", None)
        )
        if name:
            names.append(str(name))
    return names


def _chat_is_preflight_ok(content: str) -> bool:
    normalized = re.sub(r"[^A-Z]+", "_", content.upper()).strip("_")
    return normalized == "PREFLIGHT_OK" or "PREFLIGHT_OK" in normalized


def run_preflight(
    *,
    config: AppConfig,
    repository_root: Path,
    jobs_path: Path,
    profile_path: Path,
    portfolio_path: Path,
    evidence_path: Path,
    base_resume_tex_path: Path,
    base_resume_pdf_path: Path,
    memory_path: Path,
    output_root: Path,
    ollama_client_factory: Callable[..., Any] = ollama.Client,
    command_runner: Callable[..., Any] = subprocess.run,
    python_version: Sequence[int] | None = None,
) -> PreflightResult:
    """Run local read-only validation without creating an agent or trace."""
    repository_root = repository_root.resolve()
    paths = {
        "jobs": jobs_path.expanduser().resolve(),
        "profile": profile_path.expanduser().resolve(),
        "portfolio": portfolio_path.expanduser().resolve(),
        "evidence": evidence_path.expanduser().resolve(),
        "base_resume_tex": base_resume_tex_path.expanduser().resolve(),
        "base_resume_pdf": base_resume_pdf_path.expanduser().resolve(),
        "memory": memory_path.expanduser().resolve(),
    }
    checks: list[PreflightCheck] = []
    candidate_id: str | None = None
    job_count: int | None = None
    resume_page_count: int | None = None
    output_root_empty: bool | None = None

    version = tuple(python_version or sys.version_info[:3])
    if version >= (3, 13):
        checks.append(
            _check(
                "Python version",
                PreflightStatus.PASS,
                f"Python {'.'.join(str(item) for item in version[:3])}",
            )
        )
    else:
        checks.append(
            _check(
                "Python version",
                PreflightStatus.FAIL,
                f"Python 3.13 or newer is required; found {version}",
            )
        )

    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        checks.append(
            _check(
                "Repository inputs",
                PreflightStatus.FAIL,
                "Missing required files: " + ", ".join(missing),
                missing=missing,
            )
        )
    else:
        checks.append(
            _check(
                "Repository inputs",
                PreflightStatus.PASS,
                "All required repository input files exist",
            )
        )

    try:
        jobs = load_jobs(paths["jobs"])
        job_count = len(jobs)
        if 20 <= job_count <= 25:
            checks.append(
                _check(
                    "Jobs dataset",
                    PreflightStatus.PASS,
                    f"{job_count} jobs",
                    job_count=job_count,
                )
            )
        else:
            checks.append(
                _check(
                    "Jobs dataset",
                    PreflightStatus.FAIL,
                    f"Expected 20–25 jobs; found {job_count}",
                    job_count=job_count,
                )
            )
    except Exception as exc:
        checks.append(
            _check(
                "Jobs dataset",
                PreflightStatus.FAIL,
                f"Jobs dataset failed validation: {exc}",
            )
        )

    bundle = None
    try:
        bundle = load_candidate_bundle(
            paths["profile"],
            paths["portfolio"],
            paths["evidence"],
        )
        candidate_id = bundle.profile.candidate_id
        checks.append(
            _check(
                "Candidate bundle",
                PreflightStatus.PASS,
                f"Candidate {candidate_id}",
                candidate_id=candidate_id,
            )
        )
    except Exception as exc:
        checks.append(
            _check(
                "Candidate bundle",
                PreflightStatus.FAIL,
                f"Candidate bundle failed integrity validation: {exc}",
            )
        )

    if bundle is None:
        checks.append(
            _check(
                "Memory",
                PreflightStatus.FAIL,
                "Memory cannot be validated until the candidate bundle loads",
            )
        )
    else:
        try:
            memory = load_memory(paths["memory"], bundle.profile.candidate_id)
            checks.append(
                _check(
                    "Memory",
                    PreflightStatus.PASS,
                    f"Candidate {memory.candidate_id}; {len(memory.facts)} fact(s)",
                    candidate_id=memory.candidate_id,
                    fact_count=len(memory.facts),
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    "Memory",
                    PreflightStatus.FAIL,
                    f"Persistent memory failed validation: {exc}",
                )
            )

    try:
        paths["base_resume_tex"].read_text(encoding="utf-8")
        checks.append(
            _check(
                "Resume TEX",
                PreflightStatus.PASS,
                "Base resume TEX is readable",
            )
        )
    except Exception as exc:
        checks.append(
            _check(
                "Resume TEX",
                PreflightStatus.FAIL,
                f"Base resume TEX is not readable: {exc}",
            )
        )

    try:
        resume_page_count = len(PdfReader(str(paths["base_resume_pdf"])).pages)
        status = (
            PreflightStatus.PASS
            if resume_page_count == 1
            else PreflightStatus.FAIL
        )
        checks.append(
            _check(
                "Resume PDF",
                status,
                f"{resume_page_count} page(s)",
                page_count=resume_page_count,
            )
        )
    except Exception as exc:
        checks.append(
            _check(
                "Resume PDF",
                PreflightStatus.FAIL,
                f"Base resume PDF is unreadable: {exc}",
            )
        )

    try:
        completed = command_runner(
            ["pdflatex", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
            shell=False,
        )
        version_text = (completed.stdout or completed.stderr or "").strip()
        if completed.returncode == 0 and version_text:
            checks.append(
                _check(
                    "pdflatex",
                    PreflightStatus.PASS,
                    version_text.splitlines()[0][:300],
                )
            )
        else:
            checks.append(
                _check(
                    "pdflatex",
                    PreflightStatus.FAIL,
                    f"pdflatex version command returned {completed.returncode}",
                )
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        checks.append(
            _check(
                "pdflatex",
                PreflightStatus.FAIL,
                f"pdflatex is unavailable: {exc}",
            )
        )

    ollama_client = None
    installed_models: list[str] = []
    try:
        ollama_client = ollama_client_factory(host=config.ollama_host, timeout=8.0)
        listed = ollama_client.list()
        installed_models = _installed_model_names(listed)
        checks.append(
            _check(
                "Ollama service",
                PreflightStatus.PASS,
                f"Local service reachable at {config.ollama_host}",
            )
        )
    except Exception as exc:
        checks.append(
            _check(
                "Ollama service",
                PreflightStatus.FAIL,
                f"Local Ollama service is unreachable: {type(exc).__name__}",
            )
        )

    model_available = config.ollama_model in installed_models
    if ollama_client is None:
        checks.append(
            _check(
                "Ollama model",
                PreflightStatus.FAIL,
                f"Cannot verify configured model {config.ollama_model}",
            )
        )
    elif model_available:
        checks.append(
            _check(
                "Ollama model",
                PreflightStatus.PASS,
                config.ollama_model,
                installed_models=installed_models,
            )
        )
    else:
        checks.append(
            _check(
                "Ollama model",
                PreflightStatus.FAIL,
                f"Configured model is not installed: {config.ollama_model}",
                installed_models=installed_models,
            )
        )

    if ollama_client is None or not model_available:
        checks.append(
            _check(
                "Ollama chat",
                PreflightStatus.FAIL,
                "Minimal chat cannot run until the service and model checks pass",
            )
        )
    else:
        try:
            response = ollama_client.chat(
                model=config.ollama_model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Reply with exactly PREFLIGHT_OK and no other text."
                        ),
                    }
                ],
                think=False,
                stream=False,
                options={
                    "temperature": 0,
                    "num_ctx": min(config.ollama_num_ctx, 2048),
                },
                keep_alive=config.ollama_keep_alive,
            )
            content = _response_content(response)
            if _chat_is_preflight_ok(content):
                checks.append(
                    _check(
                        "Ollama chat",
                        PreflightStatus.PASS,
                        "PREFLIGHT_OK",
                    )
                )
            else:
                checks.append(
                    _check(
                        "Ollama chat",
                        PreflightStatus.FAIL,
                        "Minimal chat did not return the required acknowledgement",
                    )
                )
        except Exception as exc:
            checks.append(
                _check(
                    "Ollama chat",
                    PreflightStatus.FAIL,
                    f"Minimal local chat failed: {type(exc).__name__}",
                )
            )

    if not config.langfuse_enabled:
        checks.append(
            _check(
                "Langfuse disabled",
                PreflightStatus.PASS,
                "Credentials are not required and no network call was made",
            )
        )
    else:
        missing_credentials = []
        if not config.langfuse_public_key.strip():
            missing_credentials.append("public key")
        if not config.langfuse_secret_key.strip():
            missing_credentials.append("secret key")
        if not config.langfuse_base_url.strip():
            missing_credentials.append("base URL")
        if missing_credentials:
            checks.append(
                _check(
                    "Langfuse configuration",
                    PreflightStatus.FAIL,
                    "Missing " + ", ".join(missing_credentials),
                )
            )
        else:
            checks.append(
                _check(
                    "Langfuse configuration",
                    PreflightStatus.PASS,
                    "Enabled credentials are configured; no trace was sent",
                )
            )

    try:
        resolved_output, _ = validate_run_paths(
            repository_root,
            output_root,
            paths["memory"],
        )
        if resolved_output.exists():
            if not resolved_output.is_dir():
                raise PathSafetyError("Output root exists but is not a directory")
            output_root_empty = not any(resolved_output.iterdir())
            description = (
                "safe and empty"
                if output_root_empty
                else "safe but currently nonempty"
            )
            status = (
                PreflightStatus.PASS
                if output_root_empty
                else PreflightStatus.WARNING
            )
        else:
            output_root_empty = True
            description = "safe and does not yet exist"
            status = PreflightStatus.PASS
        checks.append(
            _check(
                "Output root safety",
                status,
                description,
                output_root=str(resolved_output),
                empty=output_root_empty,
            )
        )
    except PathSafetyError as exc:
        resolved_output = output_root.expanduser().resolve()
        checks.append(
            _check(
                "Output root safety",
                PreflightStatus.FAIL,
                str(exc),
                output_root=str(resolved_output),
            )
        )

    return PreflightResult(
        repository_root=repository_root,
        checks=checks,
        candidate_id=candidate_id,
        job_count=job_count,
        resume_page_count=resume_page_count,
        ollama_model=config.ollama_model,
        output_root=resolved_output,
        output_root_empty=output_root_empty,
    )


def format_preflight_result(result: PreflightResult) -> str:
    """Format a compact table-like report with deterministic totals."""
    lines = [
        f"[{check.status.value}] {check.name}: {check.message}"
        for check in result.checks
    ]
    lines.extend(
        [
            "",
            "Summary:",
            (
                f"{result.passed_count} passed, "
                f"{result.warning_count} warnings, "
                f"{result.failed_count} failed"
            ),
        ]
    )
    return "\n".join(lines)


__all__ = [
    "PathSafetyError",
    "PreflightCheck",
    "PreflightResult",
    "PreflightStatus",
    "find_completed_output_files",
    "format_preflight_result",
    "run_preflight",
    "validate_run_paths",
]
