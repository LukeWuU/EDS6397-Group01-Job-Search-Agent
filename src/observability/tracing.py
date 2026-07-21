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
    trace_public: bool = False
    tracing_enabled: bool = False
    flushed: bool = False
    publication_error: str | None = None
    tracing_error: str | None = None
    root: ObservationRecord
    observations: list[ObservationRecord] = Field(default_factory=list)
    flush_count: int = 0


class TraceContext:
    """Runtime root trace handle."""

    def __init__(
        self,
        record: TraceRecord,
        native: Any = None,
        native_context: Any = None,
    ) -> None:
        self.record = record
        self.native = native
        self.native_context = native_context

    @property
    def trace_id(self) -> str:
        return self.record.trace_id

    @property
    def trace_url(self) -> str | None:
        return self.record.trace_url

    @property
    def trace_public(self) -> bool:
        return self.record.trace_public

    @property
    def flushed(self) -> bool:
        return self.record.flushed

    @property
    def tracing_enabled(self) -> bool:
        return self.record.tracing_enabled

    @property
    def publication_error(self) -> str | None:
        return self.record.publication_error


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
            trace.flushed = True


class LangfuseAgentTracer(NoOpAgentTracer):
    """Langfuse 4.14 tracer retaining the same inspectable local records."""

    def __init__(self, config: AppConfig, *, client: Any = None) -> None:
        super().__init__()
        self._config = config
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
        local.record.tracing_enabled = True
        native_context = self._client.start_as_current_observation(
            name=name,
            as_type="agent",
            input=local.record.root.input,
            metadata=local.record.root.metadata,
            end_on_exit=True,
        )
        native = None
        try:
            native = native_context.__enter__()
            trace_id = self._client.get_current_trace_id()
            if not trace_id:
                raise RuntimeError(
                    "Langfuse did not provide an active root trace ID"
                )
        except BaseException as exc:
            if native is not None:
                try:
                    native_context.__exit__(
                        type(exc),
                        exc,
                        exc.__traceback__,
                    )
                except Exception:
                    pass
            raise
        local.record.trace_id = trace_id
        local.record.root.trace_id = trace_id
        local.native = native
        local.native_context = native_context
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
        try:
            span.native.update(
                output=span.record.output,
                level="ERROR" if span.record.status == "error" else "DEFAULT",
                status_message=span.record.error,
            )
            span.native.end()
        except Exception as exc:
            span.trace.record.tracing_error = (
                f"Child observation close failed: {type(exc).__name__}"
            )

    def end_trace(
        self,
        trace: TraceContext,
        *,
        output: Any = None,
        error: str | None = None,
    ) -> None:
        super().end_trace(trace, output=output, error=error)
        try:
            trace.native.update(
                output=trace.record.root.output,
                level="ERROR" if error else "DEFAULT",
                status_message=error,
            )
        except Exception as exc:
            trace.record.tracing_error = (
                f"Root observation update failed: {type(exc).__name__}"
            )

        completed = (
            isinstance(output, Mapping)
            and output.get("completed") is True
            and error is None
        )
        if (
            completed
            and self._config.langfuse_public_trace
            and trace.record.tracing_error is None
        ):
            try:
                self._client.set_current_trace_as_public()
                trace.record.trace_public = True
                trace_url = self._client.get_trace_url(trace_id=trace.trace_id)
                if not trace_url:
                    raise RuntimeError("SDK returned no trace URL")
                trace.record.trace_url = trace_url
            except Exception as exc:
                trace.record.trace_url = None
                trace.record.publication_error = (
                    f"Public trace publication failed: {type(exc).__name__}"
                )
                if not trace.record.trace_public:
                    trace.record.trace_public = False

        try:
            trace.native_context.__exit__(None, None, None)
        except Exception as exc:
            trace.record.tracing_error = (
                f"Root observation close failed: {type(exc).__name__}"
            )

    def flush(self) -> None:
        self.flush_count += 1
        try:
            self._client.flush()
        except Exception as exc:
            for trace in self.traces:
                trace.tracing_error = (
                    f"Trace flush failed: {type(exc).__name__}"
                )
                trace.flushed = False
                trace.flush_count = self.flush_count
        else:
            for trace in self.traces:
                trace.flushed = True
                trace.flush_count = self.flush_count


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
