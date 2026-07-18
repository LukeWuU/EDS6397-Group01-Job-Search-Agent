"""Focused tests for normalized local-model client behavior."""

from __future__ import annotations

import pytest

from src.agent.client import (
    ChatModelResponseError,
    ChatModelTransportError,
    NormalizedAssistantMessage,
    OllamaChatModelClient,
    normalize_assistant_message,
)
from src.config import AppConfig


class FakeOllama:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_normalizes_tool_calls_and_preserves_provided_id():
    result = normalize_assistant_message(
        {
            "message": {
                "content": "calling",
                "tool_calls": [
                    {
                        "id": "call-7",
                        "function": {
                            "name": "filter_jobs",
                            "arguments": {"decision_summary": "Filter once."},
                        },
                    }
                ],
            }
        }
    )
    assert isinstance(result, NormalizedAssistantMessage)
    assert result.tool_calls[0].id == "call-7"
    assert result.tool_calls[0].name == "filter_jobs"


@pytest.mark.parametrize(
    "arguments",
    [None, "not-json", [], 42],
)
def test_rejects_missing_or_malformed_arguments(arguments):
    with pytest.raises(ChatModelResponseError):
        normalize_assistant_message(
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "filter_jobs", "arguments": arguments}}
                    ]
                }
            }
        )


def test_ollama_062_arguments_are_explicit_and_no_secret_is_exposed():
    fake = FakeOllama({"message": {"content": "ok", "tool_calls": []}})
    config = AppConfig(
        ollama_host="http://local.test",
        ollama_model="qwen-test",
        ollama_num_ctx=4096,
        ollama_temperature=0.2,
        ollama_keep_alive="3m",
        langfuse_secret_key="must-not-leak",
    )
    client = OllamaChatModelClient(config, client=fake)
    client.chat([{"role": "user", "content": "go"}], [])
    call = fake.calls[0]
    assert call["think"] is False
    assert call["stream"] is False
    assert call["options"] == {
        "num_ctx": 4096,
        "num_predict": 1024,
        "temperature": 0.2,
    }
    assert call["keep_alive"] == "3m"
    assert "must-not-leak" not in repr(call)


def test_production_client_uses_configured_transport_timeout(monkeypatch):
    fake = FakeOllama({"message": {"content": "ok", "tool_calls": []}})
    constructions = []

    def factory(**kwargs):
        constructions.append(kwargs)
        return fake

    monkeypatch.setattr("src.agent.client.ollama.Client", factory)
    config = AppConfig(
        ollama_host="http://production-local.test",
        ollama_request_timeout_seconds=321.5,
    )

    client = OllamaChatModelClient(config)
    client.chat([{"role": "user", "content": "safe test"}], [])

    assert constructions == [
        {
            "host": "http://production-local.test",
            "timeout": 321.5,
        }
    ]
    assert len(fake.calls) == 1


def test_read_timeout_is_wrapped_once_without_sensitive_text():
    class ReadTimeout(Exception):
        pass

    class TimingOutOllama:
        def __init__(self):
            self.calls = 0

        def chat(self, **kwargs):
            self.calls += 1
            raise ReadTimeout(
                "prompt=PRIVATE_PROMPT secret=PRIVATE_SECRET arguments=PRIVATE_ARGS"
            )

    fake = TimingOutOllama()
    client = OllamaChatModelClient(
        AppConfig(
            ollama_model="qwen3:8b",
            langfuse_secret_key="PRIVATE_SECRET",
        ),
        client=fake,
    )

    with pytest.raises(ChatModelTransportError) as captured:
        client.chat(
            [{"role": "user", "content": "PRIVATE_PROMPT"}],
            [{"function": {"arguments": "PRIVATE_ARGS"}}],
        )

    assert isinstance(captured.value.__cause__, ReadTimeout)
    assert str(captured.value) == "Local Ollama chat request failed: ReadTimeout"
    assert "PRIVATE_PROMPT" not in str(captured.value)
    assert "PRIVATE_SECRET" not in str(captured.value)
    assert "PRIVATE_ARGS" not in str(captured.value)
    assert fake.calls == 1
