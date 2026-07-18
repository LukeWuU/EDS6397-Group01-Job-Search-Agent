"""Focused production CLI tests without a real model or network."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

import src.cli as cli
from src.agent.state import AgentRunResult
from src.config import AppConfig
from src.services.preflight import (
    PreflightCheck,
    PreflightResult,
    PreflightStatus,
)


@pytest.mark.parametrize(
    "argv",
    [["--help"], ["preflight", "--help"], ["run", "--help"]],
)
def test_cli_help_succeeds(argv):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 0


def _preflight(repo: Path, *, passed: bool) -> PreflightResult:
    return PreflightResult(
        repository_root=repo,
        checks=[
            PreflightCheck(
                name="mock",
                status=PreflightStatus.PASS if passed else PreflightStatus.FAIL,
                message="mocked",
            )
        ],
        ollama_model="qwen3:8b",
        output_root=repo / "outputs",
        output_root_empty=True,
    )


def _result(*, completed=True, trace_url=None) -> AgentRunResult:
    return AgentRunResult(
        run_id="run-cli-test",
        completed=completed,
        failure_reason=None if completed else "scripted failure",
        model_name="qwen3:8b",
        model_call_count=7,
        tool_call_count=11,
        invalid_tool_attempt_count=1,
        tool_execution_records=[],
        top_3_job_ids=["job-a", "job-b", "job-c"],
        top_3_scores={"job-a": 90.0, "job-b": 80.0, "job-c": 70.0},
        fit_analysis_count=3,
        draft_resume_count=3,
        pause_count=1,
        learned_memory_fact_ids=["fact-1"],
        finalized_resume_count=3,
        cover_letter_count=3,
        output_folders={"job-a": Path("outputs/job-a")},
        trace_id="trace-1",
        trace_url=trace_url,
        state_summary={"phase": "completed" if completed else "failed"},
    )


def _run_args(repo: Path, *extra: str) -> list[str]:
    return [
        "run",
        "--output-root",
        str(repo / "outputs"),
        "--memory",
        str(repo / "memory.json"),
        *extra,
    ]


def _repo(tmp_path: Path, monkeypatch) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "memory.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli, "REPOSITORY_ROOT", repo)
    return repo


def test_preflight_failure_prevents_runtime_construction(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli, "_preflight_for_args", lambda *args: _preflight(repo, passed=False)
    )
    monkeypatch.setattr(
        cli,
        "OllamaChatModelClient",
        lambda *_: pytest.fail("model client must not be constructed"),
    )
    assert cli.main(_run_args(repo, "--yes")) == 1


def test_negative_start_confirmation_prevents_runtime_construction(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli, "_preflight_for_args", lambda *args: _preflight(repo, passed=True)
    )
    monkeypatch.setattr(builtins, "input", lambda prompt: "no")
    monkeypatch.setattr(
        cli,
        "OllamaChatModelClient",
        lambda *_: pytest.fail("model client must not be constructed"),
    )
    assert cli.main(_run_args(repo)) == 0


def test_yes_constructs_each_dependency_once_and_invokes_runtime_once(
    tmp_path, monkeypatch, capsys
):
    repo = _repo(tmp_path, monkeypatch)
    counts = {"client": 0, "tracer": 0, "provider": 0, "runtime": 0}
    objects = {}

    def client(config):
        counts["client"] += 1
        objects["client"] = object()
        return objects["client"]

    def tracer(config):
        counts["tracer"] += 1
        objects["tracer"] = object()
        return objects["tracer"]

    def provider():
        counts["provider"] += 1
        objects["provider"] = object()
        return objects["provider"]

    def runtime(**kwargs):
        counts["runtime"] += 1
        assert kwargs["client"] is objects["client"]
        assert kwargs["tracer"] is objects["tracer"]
        assert kwargs["review_decision_provider"] is objects["provider"]
        assert kwargs["progress_callback"] is print
        assert cli.runtime_module.MAX_MODEL_CALLS == 12
        assert cli.runtime_module.MAX_TOOL_CALLS == 18
        kwargs["progress_callback"]("Agent phase: filtering")
        return _result()

    monkeypatch.setattr(
        builtins,
        "input",
        lambda _: pytest.fail("--yes must skip only the start prompt"),
    )
    monkeypatch.setattr(cli, "OllamaChatModelClient", client)
    monkeypatch.setattr(cli, "build_agent_tracer", tracer)
    monkeypatch.setattr(cli, "ConsoleReviewDecisionProvider", provider)
    monkeypatch.setattr(cli, "run_job_search_agent", runtime)
    code = cli.main(
        _run_args(
            repo,
            "--yes",
            "--skip-preflight",
            "--max-model-calls",
            "12",
            "--max-tool-calls",
            "18",
        )
    )
    assert code == 0
    assert counts == {"client": 1, "tracer": 1, "provider": 1, "runtime": 1}
    output = capsys.readouterr().out
    assert "[WARNING] Preflight skipped" in output
    assert "Agent phase: filtering" in output
    assert "Agent run completed" in output
    assert "Human Review pauses: 1" in output
    assert "Trace URL: not available" in output
    assert "Trace visibility: disabled" in output
    assert cli.runtime_module.MAX_MODEL_CALLS == 40
    assert cli.runtime_module.MAX_TOOL_CALLS == 60


def test_failed_result_returns_one_and_prints_failure_summary(
    tmp_path, monkeypatch, capsys
):
    repo = _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "OllamaChatModelClient", lambda _: object())
    monkeypatch.setattr(cli, "build_agent_tracer", lambda _: object())
    monkeypatch.setattr(cli, "ConsoleReviewDecisionProvider", lambda: object())
    monkeypatch.setattr(cli, "run_job_search_agent", lambda **kwargs: _result(completed=False))
    assert cli.main(_run_args(repo, "--yes", "--skip-preflight")) == 1
    output = capsys.readouterr().out
    assert "Agent run failed" in output
    assert "Failure: scripted failure" in output
    assert "Phase: failed" in output


def test_keyboard_interrupt_returns_130(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "OllamaChatModelClient", lambda _: object())
    monkeypatch.setattr(cli, "build_agent_tracer", lambda _: object())
    monkeypatch.setattr(cli, "ConsoleReviewDecisionProvider", lambda: object())

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_job_search_agent", interrupt)
    assert cli.main(_run_args(repo, "--yes", "--skip-preflight")) == 130


@pytest.mark.parametrize(
    ("config", "trace_url", "expected_url", "expected_visibility"),
    [
        (
            AppConfig(langfuse_enabled=False),
            None,
            "Trace URL: not available",
            "Trace visibility: disabled",
        ),
        (
            AppConfig(langfuse_enabled=True, langfuse_public_trace=False),
            None,
            "Trace URL: not publicly available",
            "Trace visibility: private",
        ),
        (
            AppConfig(langfuse_enabled=True, langfuse_public_trace=True),
            "https://trace.local/actual-trace",
            "Trace URL: https://trace.local/actual-trace",
            "Trace visibility: public",
        ),
    ],
)
def test_trace_visibility_output(
    config,
    trace_url,
    expected_url,
    expected_visibility,
    capsys,
):
    config.langfuse_public_key = "must-not-print-public-key"
    config.langfuse_secret_key = "must-not-print-secret-key"
    cli._print_success(_result(trace_url=trace_url), config)
    output = capsys.readouterr().out
    assert expected_url in output
    assert expected_visibility in output
    assert "must-not-print-public-key" not in output
    assert "must-not-print-secret-key" not in output


def test_completed_output_files_are_refused_before_construction(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path, monkeypatch)
    completed = repo / "outputs/job-existing"
    completed.mkdir(parents=True)
    (completed / "resume_after.pdf").write_bytes(b"existing")
    monkeypatch.setattr(
        cli,
        "OllamaChatModelClient",
        lambda *_: pytest.fail("model client must not be constructed"),
    )
    assert cli.main(_run_args(repo, "--yes", "--skip-preflight")) == 1
