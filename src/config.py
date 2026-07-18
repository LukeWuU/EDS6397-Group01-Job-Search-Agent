"""Application configuration loaded from environment variables and optional .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, Field, field_validator


class AppConfig(BaseModel):
    """Runtime configuration for the Job Search Agent."""

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_num_ctx: int = 8192
    ollama_num_predict: int = 1024
    ollama_request_timeout_seconds: float = 600.0
    ollama_temperature: float = 0.0
    ollama_keep_alive: str = "10m"
    langfuse_enabled: bool = False
    langfuse_public_trace: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://us.cloud.langfuse.com"
    langfuse_tracing_environment: str = "development"
    project_env: str = "development"
    log_level: str = "INFO"

    @field_validator("ollama_num_ctx")
    @classmethod
    def validate_positive_context(cls, value: int) -> int:
        """Reject non-positive Ollama context window sizes."""
        if value <= 0:
            raise ValueError("ollama_num_ctx must be a positive integer")
        return value

    @field_validator("ollama_num_predict")
    @classmethod
    def validate_positive_generation_limit(cls, value: int) -> int:
        """Reject non-positive generation limits."""
        if value <= 0:
            raise ValueError("ollama_num_predict must be a positive integer")
        return value

    @field_validator("ollama_request_timeout_seconds")
    @classmethod
    def validate_positive_timeout(cls, value: float) -> float:
        """Reject non-positive production request timeouts."""
        if value <= 0:
            raise ValueError("ollama_request_timeout_seconds must be greater than zero")
        return value

    @field_validator("ollama_temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Reject negative sampling temperatures."""
        if value < 0:
            raise ValueError("ollama_temperature must be zero or greater")
        return value


_ENV_FIELD_MAP: dict[str, str] = {
    "OLLAMA_HOST": "ollama_host",
    "OLLAMA_MODEL": "ollama_model",
    "OLLAMA_NUM_CTX": "ollama_num_ctx",
    "OLLAMA_NUM_PREDICT": "ollama_num_predict",
    "OLLAMA_REQUEST_TIMEOUT_SECONDS": "ollama_request_timeout_seconds",
    "OLLAMA_TEMPERATURE": "ollama_temperature",
    "OLLAMA_KEEP_ALIVE": "ollama_keep_alive",
    "LANGFUSE_ENABLED": "langfuse_enabled",
    "LANGFUSE_PUBLIC_TRACE": "langfuse_public_trace",
    "LANGFUSE_PUBLIC_KEY": "langfuse_public_key",
    "LANGFUSE_SECRET_KEY": "langfuse_secret_key",
    "LANGFUSE_BASE_URL": "langfuse_base_url",
    "LANGFUSE_TRACING_ENVIRONMENT": "langfuse_tracing_environment",
    "PROJECT_ENV": "project_env",
    "LOG_LEVEL": "log_level",
}


def _parse_bool(value: str) -> bool:
    """Convert common boolean string representations."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _coerce_env_value(field_name: str, raw_value: str) -> Any:
    """Convert a raw environment string to the typed config value."""
    if field_name in {"ollama_num_ctx", "ollama_num_predict"}:
        return int(raw_value)
    if field_name in {
        "ollama_temperature",
        "ollama_request_timeout_seconds",
    }:
        return float(raw_value)
    if field_name in {"langfuse_enabled", "langfuse_public_trace"}:
        return _parse_bool(raw_value)
    return raw_value


def _resolve_env_path(env_path: Path | None) -> Path | None:
    """Return an existing .env path when one is available."""
    if env_path is not None:
        return env_path if env_path.is_file() else None
    default_path = Path(".env")
    return default_path if default_path.is_file() else None


def load_config(env_path: Path | None = None) -> AppConfig:
    """Load configuration using defaults, optional .env values, and process env.

    A local ``.env`` file is loaded when present. ``.env.example`` is never
    loaded automatically during normal execution.
    """
    values: dict[str, Any] = {}

    resolved_env_path = _resolve_env_path(env_path)
    if resolved_env_path is not None:
        file_values = dotenv_values(resolved_env_path)
        for env_key, field_name in _ENV_FIELD_MAP.items():
            raw_value = file_values.get(env_key)
            if raw_value is not None and str(raw_value).strip() != "":
                values[field_name] = _coerce_env_value(field_name, str(raw_value))

    for env_key, field_name in _ENV_FIELD_MAP.items():
        import os

        raw_value = os.environ.get(env_key)
        if raw_value is not None and str(raw_value).strip() != "":
            values[field_name] = _coerce_env_value(field_name, str(raw_value))

    return AppConfig(**values)


__all__ = ["AppConfig", "load_config"]
