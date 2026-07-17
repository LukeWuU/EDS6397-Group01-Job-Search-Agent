"""Normalized chat-model client for the single runtime reasoning loop."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import ollama
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import AppConfig


class ChatModelClientError(Exception):
    """Base error for model-client failures."""


class ChatModelTransportError(ChatModelClientError):
    """Raised when the local model service cannot complete a request."""


class ChatModelResponseError(ChatModelClientError):
    """Raised when a model response cannot be normalized safely."""


class NormalizedToolCall(BaseModel):
    """One normalized function call returned by a chat model."""

    model_config = ConfigDict(extra="forbid")

    id: str | None = None
    name: str
    arguments: dict[str, Any]

    @field_validator("name")
    @classmethod
    def require_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("tool-call function name must be nonempty")
        return value


class NormalizedAssistantMessage(BaseModel):
    """Provider-neutral assistant content and tool calls."""

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    tool_calls: list[NormalizedToolCall] = Field(default_factory=list)


@runtime_checkable
class ChatModelClient(Protocol):
    """Protocol used by the one production runtime and scripted tests."""

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> NormalizedAssistantMessage:
        """Return one normalized, non-streaming assistant message."""


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _normalize_arguments(raw: Any, *, function_name: str) -> dict[str, Any]:
    if raw is None:
        raise ChatModelResponseError(
            f"Tool call {function_name!r} is missing function arguments"
        )
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ChatModelResponseError(
                f"Tool call {function_name!r} has malformed JSON arguments"
            ) from exc
    if not isinstance(raw, Mapping):
        raise ChatModelResponseError(
            f"Tool call {function_name!r} arguments must be a JSON object"
        )
    return {str(key): item for key, item in raw.items()}


def normalize_assistant_message(response: Any) -> NormalizedAssistantMessage:
    """Normalize Ollama 0.6.x typed or mapping responses."""
    message = _get_value(response, "message")
    if message is None:
        raise ChatModelResponseError("Chat response is missing an assistant message")
    content = _get_value(message, "content", "") or ""
    if not isinstance(content, str):
        raise ChatModelResponseError("Assistant message content must be text")

    normalized_calls: list[NormalizedToolCall] = []
    raw_calls = _get_value(message, "tool_calls", None) or []
    if isinstance(raw_calls, (str, bytes)) or not isinstance(raw_calls, Sequence):
        raise ChatModelResponseError("Assistant tool_calls must be a sequence")
    for index, raw_call in enumerate(raw_calls):
        function = _get_value(raw_call, "function")
        if function is None:
            raise ChatModelResponseError(
                f"Tool call at index {index} is missing its function"
            )
        name = _get_value(function, "name")
        if not isinstance(name, str) or not name.strip():
            raise ChatModelResponseError(
                f"Tool call at index {index} is missing a function name"
            )
        call_id = _get_value(raw_call, "id")
        if call_id is not None and not isinstance(call_id, str):
            call_id = str(call_id)
        normalized_calls.append(
            NormalizedToolCall(
                id=call_id,
                name=name,
                arguments=_normalize_arguments(
                    _get_value(function, "arguments"),
                    function_name=name,
                ),
            )
        )
    return NormalizedAssistantMessage(content=content, tool_calls=normalized_calls)


class OllamaChatModelClient:
    """Thin Ollama 0.6.2 adapter; it intentionally contains no agent loop."""

    def __init__(
        self,
        config: AppConfig,
        *,
        client: ollama.Client | None = None,
    ) -> None:
        self._config = config
        self._client = client or ollama.Client(host=config.ollama_host)

    @property
    def model_name(self) -> str:
        return self._config.ollama_model

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> NormalizedAssistantMessage:
        try:
            response = self._client.chat(
                model=self._config.ollama_model,
                messages=list(messages),
                tools=list(tools),
                stream=False,
                think=False,
                options={
                    "num_ctx": self._config.ollama_num_ctx,
                    "temperature": self._config.ollama_temperature,
                },
                keep_alive=self._config.ollama_keep_alive,
            )
        except Exception as exc:
            raise ChatModelTransportError(
                f"Local Ollama chat request failed for model "
                f"{self._config.ollama_model!r}: {type(exc).__name__}"
            ) from exc
        return normalize_assistant_message(response)


__all__ = [
    "ChatModelClient",
    "ChatModelClientError",
    "ChatModelResponseError",
    "ChatModelTransportError",
    "NormalizedAssistantMessage",
    "NormalizedToolCall",
    "OllamaChatModelClient",
    "normalize_assistant_message",
]
