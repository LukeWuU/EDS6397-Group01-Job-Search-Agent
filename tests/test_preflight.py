"""Focused read-only preflight and path-safety tests."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from src.config import AppConfig
from src.services.preflight import (
    OLLAMA_PREFLIGHT_TIMEOUT_SECONDS,
    PathSafetyError,
    PreflightStatus,
    run_preflight,
    validate_run_paths,
)

ROOT = Path(__file__).resolve().parents[1]


def _copy_inputs(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    (repo / "candidate").mkdir(parents=True)
    (repo / "data").mkdir()
    for name in (
        "profile.json",
        "portfolio.json",
        "evidence_registry.json",
        "sample_resume.tex",
        "sample_resume.pdf",
    ):
        shutil.copyfile(ROOT / "candidate" / name, repo / "candidate" / name)
    shutil.copyfile(
        ROOT / "data/AI_ML_Jobs_Dataset_20.csv",
        repo / "data/AI_ML_Jobs_Dataset_20.csv",
    )
    shutil.copyfile(ROOT / "memory.json", repo / "memory.json")
    return {
        "repository_root": repo,
        "jobs_path": repo / "data/AI_ML_Jobs_Dataset_20.csv",
        "profile_path": repo / "candidate/profile.json",
        "portfolio_path": repo / "candidate/portfolio.json",
        "evidence_path": repo / "candidate/evidence_registry.json",
        "base_resume_tex_path": repo / "candidate/sample_resume.tex",
        "base_resume_pdf_path": repo / "candidate/sample_resume.pdf",
        "memory_path": repo / "memory.json",
        "output_root": repo / "outputs",
    }


class FakeOllama:
    def __init__(
        self,
        *,
        reachable=True,
        model=True,
        chat=True,
        chat_exception: Exception | None = None,
    ):
        self.reachable = reachable
        self.model = model
        self.chat_ok = chat
        self.chat_exception = chat_exception
        self.chat_calls = []

    def list(self):
        if not self.reachable:
            raise ConnectionError("offline")
        return {"models": [{"model": "qwen3:8b"}] if self.model else []}

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        if self.chat_exception is not None:
            raise self.chat_exception
        if not self.chat_ok:
            raise RuntimeError("chat failed")
        return {"message": {"content": "PREFLIGHT_OK"}}


def _factory(fake):
    def create(**kwargs):
        assert kwargs["timeout"] == OLLAMA_PREFLIGHT_TIMEOUT_SECONDS == 60.0
        return fake

    return create


def _runner(*args, **kwargs):
    assert args[0] == ["pdflatex", "--version"]
    assert kwargs["shell"] is False
    return SimpleNamespace(returncode=0, stdout="pdfTeX 3.14\n", stderr="")


def _run(paths, fake=None, config=None, runner=_runner):
    return run_preflight(
        config=config or AppConfig(langfuse_enabled=False),
        ollama_client_factory=_factory(fake or FakeOllama()),
        command_runner=runner,
        python_version=(3, 13, 2),
        **paths,
    )


def _status(result, name):
    return next(check.status for check in result.checks if check.name == name)


def _hashes(paths):
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
        if name.endswith("_path") and path.is_file()
    }


def test_all_checks_pass_without_modifying_inputs_or_outputs(tmp_path):
    paths = _copy_inputs(tmp_path)
    before = _hashes(paths)
    fake = FakeOllama()
    result = _run(
        paths,
        fake,
        config=AppConfig(
            langfuse_enabled=False,
            ollama_keep_alive="7m",
        ),
    )
    assert result.succeeded
    assert result.job_count == 20
    assert result.candidate_id == "cand-mira-solenne-001"
    assert result.resume_page_count == 1
    assert result.output_root_empty is True
    assert not paths["output_root"].exists()
    assert _hashes(paths) == before
    call = fake.chat_calls[0]
    assert call["think"] is False and call["stream"] is False
    assert call["options"]["temperature"] == 0
    assert call["options"]["num_ctx"] <= 2048
    assert call["options"]["num_predict"] == 16
    assert call["keep_alive"] == "7m"
    assert _status(result, "Ollama chat") == PreflightStatus.PASS


def test_missing_csv_invalid_candidate_and_memory_mismatch_fail(tmp_path):
    missing = _copy_inputs(tmp_path / "missing")
    missing["jobs_path"].unlink()
    assert _status(_run(missing), "Jobs dataset") == PreflightStatus.FAIL

    invalid = _copy_inputs(tmp_path / "invalid")
    profile = json.loads(invalid["profile_path"].read_text(encoding="utf-8"))
    profile["base_resume_project_ids"] = ["bad"]
    invalid["profile_path"].write_text(json.dumps(profile), encoding="utf-8")
    assert _status(_run(invalid), "Candidate bundle") == PreflightStatus.FAIL

    mismatch = _copy_inputs(tmp_path / "mismatch")
    memory = json.loads(mismatch["memory_path"].read_text(encoding="utf-8"))
    memory["candidate_id"] = "other-candidate"
    mismatch["memory_path"].write_text(json.dumps(memory), encoding="utf-8")
    assert _status(_run(mismatch), "Memory") == PreflightStatus.FAIL


def test_two_page_resume_and_missing_pdflatex_fail(tmp_path):
    paths = _copy_inputs(tmp_path)
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)
    with paths["base_resume_pdf_path"].open("wb") as handle:
        writer.write(handle)

    def missing_runner(*args, **kwargs):
        raise FileNotFoundError("pdflatex")

    result = _run(paths, runner=missing_runner)
    assert _status(result, "Resume PDF") == PreflightStatus.FAIL
    assert _status(result, "pdflatex") == PreflightStatus.FAIL


@pytest.mark.parametrize(
    ("fake", "failed_check"),
    [
        (FakeOllama(reachable=False), "Ollama service"),
        (FakeOllama(model=False), "Ollama model"),
        (FakeOllama(chat=False), "Ollama chat"),
    ],
)
def test_ollama_failure_modes_are_reported(tmp_path, fake, failed_check):
    result = _run(_copy_inputs(tmp_path), fake)
    assert _status(result, failed_check) == PreflightStatus.FAIL


def test_simulated_ollama_timeout_remains_a_read_only_failure(
    tmp_path,
    monkeypatch,
):
    class ReadTimeout(Exception):
        pass

    source_memory = ROOT / "memory.json"
    memory_before = hashlib.sha256(source_memory.read_bytes()).hexdigest()
    source_outputs = ROOT / "outputs"
    outputs_before = (
        sorted(str(path.relative_to(source_outputs)) for path in source_outputs.rglob("*"))
        if source_outputs.exists()
        else []
    )
    monkeypatch.setattr(
        "src.observability.tracing.build_agent_tracer",
        lambda *args, **kwargs: pytest.fail("Preflight must not create a trace"),
    )
    paths = _copy_inputs(tmp_path)
    before = _hashes(paths)

    result = _run(
        paths,
        FakeOllama(chat_exception=ReadTimeout("simulated timeout")),
    )

    assert _status(result, "Ollama chat") == PreflightStatus.FAIL
    chat_check = next(
        check for check in result.checks if check.name == "Ollama chat"
    )
    assert "ReadTimeout" in chat_check.message
    assert not paths["output_root"].exists()
    assert _hashes(paths) == before
    assert hashlib.sha256(source_memory.read_bytes()).hexdigest() == memory_before
    outputs_after = (
        sorted(str(path.relative_to(source_outputs)) for path in source_outputs.rglob("*"))
        if source_outputs.exists()
        else []
    )
    assert outputs_after == outputs_before


def test_langfuse_configuration_is_local_only(tmp_path):
    paths = _copy_inputs(tmp_path)
    disabled = _run(
        paths,
        config=AppConfig(
            langfuse_enabled=False,
            langfuse_public_key="",
            langfuse_secret_key="",
        ),
    )
    assert _status(disabled, "Langfuse disabled") == PreflightStatus.PASS

    enabled = _run(
        paths,
        config=AppConfig(
            langfuse_enabled=True,
            langfuse_public_key="",
            langfuse_secret_key="",
            langfuse_base_url="",
        ),
    )
    assert _status(enabled, "Langfuse configuration") == PreflightStatus.FAIL


def test_path_safety_accepts_normal_output_and_rejects_protected_paths(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    memory = repo / "memory.json"
    memory.write_text("{}", encoding="utf-8")
    output, resolved_memory = validate_run_paths(repo, repo / "outputs", memory)
    assert output == (repo / "outputs").resolve()
    assert resolved_memory == memory.resolve()

    with pytest.raises(PathSafetyError, match="repository root"):
        validate_run_paths(repo, repo, memory)
    for directory in ("candidate", "data", "src", "tests", ".git", ".venv"):
        with pytest.raises(PathSafetyError, match=directory.replace(".", r"\.")):
            validate_run_paths(repo, repo / directory / "generated", memory)
    with pytest.raises(PathSafetyError, match="inside outputs"):
        validate_run_paths(repo, repo / "outputs", repo / "outputs/memory.json")
    with pytest.raises(PathSafetyError, match="inside outputs"):
        validate_run_paths(repo, repo / "artifacts", repo / "outputs/memory.json")
    with pytest.raises(PathSafetyError, match="inside the output"):
        validate_run_paths(repo, repo / "artifacts", repo / "artifacts/memory.json")
    with pytest.raises(PathSafetyError, match="inside"):
        validate_run_paths(repo, tmp_path / "outside", memory)


def test_output_symbolic_link_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "real-output"
    target.mkdir()
    link = repo / "outputs"
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        pytest.skip("Symbolic links are unavailable in this environment")
    with pytest.raises(PathSafetyError, match="symbolic"):
        validate_run_paths(repo, link, repo / "memory.json")
