"""Observability abstraction with offline and Langfuse 4.14 implementations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from src.config import AppConfig

_MAX_STRING = 4000
_SECRET_KEYS = (
    "secret",
    "password",
    "api_key",
    "apikey",
    "authorization",
    "token",
)


def sanitize_trace_value(value: Any, *, key: str = "") -> Any:
    """Return bounded JSON-compatible trace data without secrets or binary data."""
    if any(marker in key.casefold() for marker in _SECRET_KEYS):
        return "[REDACTED]"
    if isinstance(value, bytes):
        return f"[BINARY OMITTED: {len(value)} bytes]"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return sanitize_trace_value(value.model_dump(mode="json"), key=key)
    if isinstance(value, Mapping):
        return {
            str(item_key): sanitize_trace_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize_trace_value(item, key=key) for item in list(value)[:200]]
    if isinstance(value, str):
        return value if len(value) <= _MAX_STRING else value[:_MAX_STRING] + "[TRUNCATED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_trace_value(str(value), key=key)


class ObservationRecord(BaseModel):
    """Local audit representation shared by both tracer implementations."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str
    trace_id: str
    parent_observation_id: str | None
    name: str
    observation_type: str = "span"
    input: Any = None
    output: Any = None
    metadata: Any = None
    status: str = "started"
    error: str | None = None


class TraceRecord(BaseModel):
    """One root trace and its nested observations."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str
    trace_url: str | None = None
    root: ObservationRecord
    observations: list[ObservationRecord] = Field(default_factory=list)
    flush_count: int = 0


class TraceContext:
    """Runtime root trace handle."""

    def __init__(self, record: TraceRecord, native: Any = None) -> None:
        self.record = record
        self.native = native

    @property
    def trace_id(self) -> str:
        return self.record.trace_id

    @property
    def trace_url(self) -> str | None:
        return self.record.trace_url


class SpanContext:
    """Nested observation context manager with explicit bounded output."""

    def __init__(
        self,
        tracer: "AgentTracer",
        trace: TraceContext,
        record: ObservationRecord,
        native: Any = None,
    ) -> None:
        self.tracer = tracer
        self.trace = trace
        self.record = record
        self.native = native

    def __enter__(self) -> "SpanContext":
        return self

    def set_output(self, value: Any) -> None:
        self.record.output = sanitize_trace_value(value)

    def set_status(self, status: str) -> None:
        self.record.status = status

    def set_error(self, error: BaseException | str) -> None:
        self.record.error = str(error)
        self.record.status = "error"

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc is not None:
            self.set_error(exc)
        elif self.record.status == "started":
            self.record.status = "ok"
        self.tracer.end_span(self)
        return False


@runtime_checkable
class AgentTracer(Protocol):
    """Tracing contract used by the one runtime."""

    def start_trace(
        self,
        name: str,
        *,
        run_id: str,
        input: Any = None,
        metadata: Any = None,
    ) -> TraceContext:
        """Create the single root trace."""

    def span(
        self,
        parent: TraceContext | SpanContext,
        name: str,
        *,
        input: Any = None,
        metadata: Any = None,
        observation_type: str = "span",
    ) -> SpanContext:
        """Create one child observation."""

    def end_span(self, span: SpanContext) -> None:
        """Finish one child observation."""

    def end_trace(
        self,
        trace: TraceContext,
        *,
        output: Any = None,
        error: str | None = None,
    ) -> None:
        """Finish the root trace."""

    def flush(self) -> None:
        """Flush tracing exactly once at run end."""


def _stable_id(*parts: str, length: int) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:length]


class NoOpAgentTracer:
    """Network-free tracer that preserves the production hierarchy."""

    def __init__(self) -> None:
        self.traces: list[TraceRecord] = []
        self.flush_count = 0

    def start_trace(
        self,
        name: str,
        *,
        run_id: str,
        input: Any = None,
        metadata: Any = None,
    ) -> TraceContext:
        trace_id = _stable_id("trace", run_id, name, length=32)
        root = ObservationRecord(
            observation_id=_stable_id(trace_id, "root", length=16),
            trace_id=trace_id,
            parent_observation_id=None,
            name=name,
            observation_type="agent",
            input=sanitize_trace_value(input),
            metadata=sanitize_trace_value(metadata),
        )
        record = TraceRecord(trace_id=trace_id, root=root)
        self.traces.append(record)
        return TraceContext(record)

    def span(
        self,
        parent: TraceContext | SpanContext,
        name: str,
        *,
        input: Any = None,
        metadata: Any = None,
        observation_type: str = "span",
    ) -> SpanContext:
        trace = parent if isinstance(parent, TraceContext) else parent.trace
        parent_id = (
            parent.record.observation_id
            if isinstance(parent, SpanContext)
            else trace.record.root.observation_id
        )
        index = len(trace.record.observations) + 1
        record = ObservationRecord(
            observation_id=_stable_id(
                trace.trace_id, parent_id, name, str(index), length=16
            ),
            trace_id=trace.trace_id,
            parent_observation_id=parent_id,
            name=name,
            observation_type=observation_type,
            input=sanitize_trace_value(input),
            metadata=sanitize_trace_value(metadata),
        )
        trace.record.observations.append(record)
        return SpanContext(self, trace, record)

    def end_span(self, span: SpanContext) -> None:
        del span

    def end_trace(
        self,
        trace: TraceContext,
        *,
        output: Any = None,
        error: str | None = None,
    ) -> None:
        trace.record.root.output = sanitize_trace_value(output)
        trace.record.root.error = error
        trace.record.root.status = "error" if error else "ok"

    def flush(self) -> None:
        self.flush_count += 1
        for trace in self.traces:
            trace.flush_count = self.flush_count


class LangfuseAgentTracer(NoOpAgentTracer):
    """Langfuse 4.14 tracer retaining the same inspectable local records."""

    def __init__(self, config: AppConfig, *, client: Any = None) -> None:
        super().__init__()
        if client is None:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=config.langfuse_public_key,
                secret_key=config.langfuse_secret_key,
                base_url=config.langfuse_base_url,
                environment=config.langfuse_tracing_environment,
                tracing_enabled=True,
            )
        self._client = client

    def start_trace(
        self,
        name: str,
        *,
        run_id: str,
        input: Any = None,
        metadata: Any = None,
    ) -> TraceContext:
        local = super().start_trace(
            name, run_id=run_id, input=input, metadata=metadata
        )
        trace_id = self._client.create_trace_id(seed=run_id)
        native = self._client.start_observation(
            trace_context={"trace_id": trace_id},
            name=name,
            as_type="agent",
            input=local.record.root.input,
            metadata=local.record.root.metadata,
        )
        local.record.trace_id = trace_id
        local.record.root.trace_id = trace_id
        local.record.trace_url = self._client.get_trace_url(trace_id=trace_id)
        local.native = native
        return local

    def span(
        self,
        parent: TraceContext | SpanContext,
        name: str,
        *,
        input: Any = None,
        metadata: Any = None,
        observation_type: str = "span",
    ) -> SpanContext:
        local = super().span(
            parent,
            name,
            input=input,
            metadata=metadata,
            observation_type=observation_type,
        )
        native_parent = parent.native
        native_type = observation_type if observation_type in {
            "generation",
            "agent",
            "tool",
            "chain",
            "retriever",
            "evaluator",
            "guardrail",
            "span",
        } else "span"
        local.native = native_parent.start_observation(
            name=name,
            as_type=native_type,
            input=local.record.input,
            metadata=local.record.metadata,
        )
        return local

    def end_span(self, span: SpanContext) -> None:
        span.native.update(
            output=span.record.output,
            level="ERROR" if span.record.status == "error" else "DEFAULT",
            status_message=span.record.error,
        )
        span.native.end()

    def end_trace(
        self,
        trace: TraceContext,
        *,
        output: Any = None,
        error: str | None = None,
    ) -> None:
        super().end_trace(trace, output=output, error=error)
        trace.native.update(
            output=trace.record.root.output,
            level="ERROR" if error else "DEFAULT",
            status_message=error,
        )
        trace.native.end()

    def flush(self) -> None:
        super().flush()
        self._client.flush()


def build_agent_tracer(config: AppConfig) -> AgentTracer:
    """Build an offline tracer unless Langfuse is explicitly enabled."""
    if not config.langfuse_enabled:
        return NoOpAgentTracer()
    return LangfuseAgentTracer(config)


__all__ = [
    "AgentTracer",
    "LangfuseAgentTracer",
    "NoOpAgentTracer",
    "ObservationRecord",
    "SpanContext",
    "TraceContext",
    "TraceRecord",
    "build_agent_tracer",
    "sanitize_trace_value",
]
