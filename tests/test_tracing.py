"""Focused offline observability tests."""

from __future__ import annotations

from src.config import AppConfig
from src.observability.tracing import (
    LangfuseAgentTracer,
    NoOpAgentTracer,
    build_agent_tracer,
    sanitize_trace_value,
)


def test_noop_tracer_keeps_one_root_and_nested_hierarchy():
    tracer = NoOpAgentTracer()
    trace = tracer.start_trace(
        "agent_run",
        run_id="stable-run",
        input={"secret_key": "hide", "pdf": b"%PDF bytes"},
    )
    with tracer.span(trace, "human_review_pause") as review:
        with tracer.span(
            review,
            "memory_write",
            input={"decision_summary": "Persist learned candidate skill."},
        ) as memory:
            memory.set_output({"ok": True})
    tracer.end_trace(trace, output={"completed": True})
    tracer.flush()

    assert len(tracer.traces) == 1
    assert tracer.flush_count == 1
    assert trace.record.root.input["secret_key"] == "[REDACTED]"
    assert "BINARY OMITTED" in trace.record.root.input["pdf"]
    observations = trace.record.observations
    assert [item.name for item in observations] == [
        "human_review_pause",
        "memory_write",
    ]
    assert observations[1].parent_observation_id == observations[0].observation_id


def test_noop_ids_are_stable_and_disabled_builder_is_network_free():
    first = NoOpAgentTracer().start_trace("agent_run", run_id="same")
    second = NoOpAgentTracer().start_trace("agent_run", run_id="same")
    assert first.trace_id == second.trace_id
    assert isinstance(build_agent_tracer(AppConfig(langfuse_enabled=False)), NoOpAgentTracer)
    assert sanitize_trace_value({"compiler_log": "x" * 5000})["compiler_log"].endswith(
        "[TRUNCATED]"
    )


class FakeObservation:
    def __init__(self):
        self.children = []
        self.updated = None
        self.ended = 0

    def start_observation(self, **kwargs):
        child = FakeObservation()
        child.kwargs = kwargs
        self.children.append(child)
        return child

    def update(self, **kwargs):
        self.updated = kwargs

    def end(self):
        self.ended += 1


class FakeLangfuse:
    def __init__(self):
        self.root = None
        self.flushes = 0

    def create_trace_id(self, *, seed):
        return f"lf-{seed}"

    def start_observation(self, **kwargs):
        self.root = FakeObservation()
        self.root.kwargs = kwargs
        return self.root

    def get_trace_url(self, *, trace_id):
        return f"https://trace.local/{trace_id}"

    def flush(self):
        self.flushes += 1


def test_langfuse_414_lifecycle_uses_one_root_and_flush():
    fake = FakeLangfuse()
    tracer = LangfuseAgentTracer(
        AppConfig(langfuse_enabled=True),
        client=fake,
    )
    trace = tracer.start_trace("agent_run", run_id="r1")
    with tracer.span(trace, "chat_model", observation_type="generation") as span:
        span.set_output({"content": "ok"})
    tracer.end_trace(trace, output={"completed": True})
    tracer.flush()
    assert trace.trace_id == "lf-r1"
    assert fake.root.ended == 1
    assert len(fake.root.children) == 1
    assert fake.flushes == 1
