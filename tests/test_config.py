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
        "OLLAMA_TEMPERATURE",
        "OLLAMA_KEEP_ALIVE",
        "LANGFUSE_ENABLED",
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
    assert config.ollama_temperature == 0.0
    assert config.ollama_keep_alive == "10m"
    assert config.langfuse_enabled is False
    assert config.langfuse_base_url == "https://us.cloud.langfuse.com"
    assert config.project_env == "development"
    assert config.log_level == "INFO"


def test_boolean_integer_and_float_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment values are converted to typed settings safely."""
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.5")
    monkeypatch.setenv("LANGFUSE_ENABLED", "true")

    config = load_config(env_path=ROOT / "missing.env")

    assert config.ollama_num_ctx == 4096
    assert config.ollama_temperature == 0.5
    assert config.langfuse_enabled is True


def test_invalid_context_size_is_rejected() -> None:
    """Non-positive context sizes are rejected."""
    with pytest.raises(ValidationError):
        AppConfig(ollama_num_ctx=0)

    with pytest.raises(ValidationError):
        AppConfig(ollama_num_ctx=-1)
