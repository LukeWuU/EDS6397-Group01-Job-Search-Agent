"""Tests for application configuration loading."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig, load_config


def test_documented_defaults_without_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults match .env.example when no overrides are present."""
    for key in (
        "OLLAMA_HOST",
        "OLLAMA_MODEL",
        "OLLAMA_NUM_CTX",
        "OLLAMA_NUM_PREDICT",
        "OLLAMA_REQUEST_TIMEOUT_SECONDS",
        "OLLAMA_TEMPERATURE",
        "OLLAMA_KEEP_ALIVE",
        "LANGFUSE_ENABLED",
        "LANGFUSE_PUBLIC_TRACE",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_BASE_URL",
        "LANGFUSE_TRACING_ENVIRONMENT",
        "PROJECT_ENV",
        "LOG_LEVEL",
    ):
        monkeypatch.delenv(key, raising=False)

    config = load_config(env_path=ROOT / "__missing__.env")

    assert config.ollama_host == "http://localhost:11434"
    assert config.ollama_model == "qwen3:8b"
    assert config.ollama_num_ctx == 8192
    assert config.ollama_num_predict == 1024
    assert config.ollama_request_timeout_seconds == 600.0
    assert config.ollama_temperature == 0.0
    assert config.ollama_keep_alive == "10m"
    assert config.langfuse_enabled is False
    assert config.langfuse_public_trace is False
    assert config.langfuse_base_url == "https://us.cloud.langfuse.com"
    assert config.project_env == "development"
    assert config.log_level == "INFO"


def test_boolean_integer_and_float_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment values are converted to typed settings safely."""
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "512")
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "45.5")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.5")
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_TRACE", "yes")

    config = load_config(env_path=ROOT / "missing.env")

    assert config.ollama_num_ctx == 4096
    assert config.ollama_num_predict == 512
    assert config.ollama_request_timeout_seconds == 45.5
    assert config.ollama_temperature == 0.5
    assert config.langfuse_enabled is True
    assert config.langfuse_public_trace is True

    monkeypatch.setenv("LANGFUSE_PUBLIC_TRACE", "false")
    assert load_config(env_path=ROOT / "missing.env").langfuse_public_trace is False


def test_invalid_public_trace_boolean_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_TRACE", "sometimes")
    with pytest.raises(ValueError, match="Invalid boolean value"):
        load_config(env_path=ROOT / "missing.env")


def test_env_example_documents_safe_public_trace_default() -> None:
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "LANGFUSE_PUBLIC_TRACE=false" in example
    assert "LANGFUSE_PUBLIC_KEY=\n" in example
    assert "LANGFUSE_SECRET_KEY=\n" in example
    assert "OLLAMA_NUM_PREDICT=1024" in example
    assert "OLLAMA_REQUEST_TIMEOUT_SECONDS=600" in example


def test_invalid_context_size_is_rejected() -> None:
    """Non-positive context sizes are rejected."""
    with pytest.raises(ValidationError):
        AppConfig(ollama_num_ctx=0)

    with pytest.raises(ValidationError):
        AppConfig(ollama_num_ctx=-1)


@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_generation_limit_is_rejected(value: int) -> None:
    with pytest.raises(ValidationError, match="ollama_num_predict"):
        AppConfig(ollama_num_predict=value)


@pytest.mark.parametrize("value", [0, -0.1])
def test_nonpositive_request_timeout_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError, match="ollama_request_timeout_seconds"):
        AppConfig(ollama_request_timeout_seconds=value)


@pytest.mark.parametrize(
    ("environment_name", "value"),
    [
        ("OLLAMA_NUM_PREDICT", "not-an-integer"),
        ("OLLAMA_REQUEST_TIMEOUT_SECONDS", "not-a-number"),
    ],
)
def test_malformed_generation_safety_environment_values_fail_clearly(
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    value: str,
) -> None:
    monkeypatch.setenv(environment_name, value)
    with pytest.raises(ValueError):
        load_config(env_path=ROOT / "missing.env")
