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
    assert trace.trace_url is None
    assert trace.trace_public is False
    assert trace.tracing_enabled is False
    assert trace.flushed is True
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
    def __init__(self, trace_id, *, publication_fails=False):
        self.trace_id = trace_id
        self.publication_fails = publication_fails
        self.children = []
        self.updated = None
        self.ended = 0
        self.public_calls = 0

    def start_observation(self, **kwargs):
        child = FakeObservation(self.trace_id)
        child.kwargs = kwargs
        self.children.append(child)
        return child

    def update(self, **kwargs):
        self.updated = kwargs

    def end(self):
        self.ended += 1

    def set_trace_as_public(self):
        self.public_calls += 1
        if self.publication_fails:
            raise RuntimeError("publication failed")
        return self


class FakeRootContext:
    def __init__(self, client, root):
        self.client = client
        self.root = root

    def __enter__(self):
        self.client.current_trace_id = self.root.trace_id
        return self.root

    def __exit__(self, exc_type, exc, traceback):
        self.root.end()
        self.client.current_trace_id = None
        return False


class FakeLangfuse:
    def __init__(self, *, publication_fails=False, url_fails=False):
        self.roots = []
        self.flushes = 0
        self.current_trace_id = None
        self.publication_fails = publication_fails
        self.url_fails = url_fails
        self.trace_url_arguments = []
        self.current_public_calls = 0
        self.publication_context_ids = []

    @property
    def root(self):
        return self.roots[0]

    def start_as_current_observation(self, **kwargs):
        root = FakeObservation(
            "0123456789abcdef0123456789abcdef",
            publication_fails=self.publication_fails,
        )
        root.kwargs = kwargs
        self.roots.append(root)
        return FakeRootContext(self, root)

    def get_current_trace_id(self):
        return self.current_trace_id

    def set_current_trace_as_public(self):
        self.current_public_calls += 1
        self.publication_context_ids.append(self.current_trace_id)
        if self.publication_fails:
            raise RuntimeError("publication failed")

    def get_trace_url(self, *, trace_id):
        self.trace_url_arguments.append(trace_id)
        if self.url_fails:
            raise RuntimeError("URL lookup failed")
        return f"https://trace.local/{trace_id}"

    def flush(self):
        self.flushes += 1



def test_langfuse_private_lifecycle_uses_one_root_without_public_url():
    fake = FakeLangfuse()
    tracer = LangfuseAgentTracer(
        AppConfig(langfuse_enabled=True, langfuse_public_trace=False),
        client=fake,
    )
    trace = tracer.start_trace("agent_run", run_id="r1")

    with tracer.span(
        trace,
        "chat_model",
        observation_type="generation",
    ) as span:
        span.set_output({"content": "ok"})

    tracer.end_trace(trace, output={"completed": True})
    tracer.flush()

    assert trace.trace_id == "0123456789abcdef0123456789abcdef"
    assert len(fake.roots) == 1
    assert fake.root.ended == 1
    assert len(fake.root.children) == 1
    assert fake.root.public_calls == 0
    assert fake.current_public_calls == 0
    assert fake.publication_context_ids == []
    assert fake.trace_url_arguments == []
    assert trace.trace_public is False
    assert trace.trace_url is None
    assert trace.tracing_enabled is True
    assert trace.flushed is True
    assert fake.flushes == 1


def test_successful_public_trace_uses_sdk_id_url_and_one_flush():
    fake = FakeLangfuse()
    tracer = LangfuseAgentTracer(
        AppConfig(langfuse_enabled=True, langfuse_public_trace=True),
        client=fake,
    )
    trace = tracer.start_trace("agent_run", run_id="public-run")

    with tracer.span(trace, "human_review_pause") as review:
        with tracer.span(review, "revision_call") as revision:
            revision.set_output({"ok": True})

    tracer.end_trace(trace, output={"completed": True})
    tracer.flush()

    assert len(fake.roots) == 1
    assert len(trace.record.observations) == 2
    assert {
        item.trace_id for item in trace.record.observations
    } == {trace.trace_id}

    # The client-level active-trace API is required for publication.
    assert fake.root.public_calls == 0
    assert fake.current_public_calls == 1
    assert fake.publication_context_ids == [trace.trace_id]

    assert fake.trace_url_arguments == [trace.trace_id]
    assert trace.trace_url == f"https://trace.local/{trace.trace_id}"
    assert trace.trace_public is True
    assert trace.publication_error is None
    assert fake.root.ended == 1
    assert fake.flushes == 1


def test_failed_trace_is_closed_flushed_and_never_public():
    fake = FakeLangfuse()
    tracer = LangfuseAgentTracer(
        AppConfig(langfuse_enabled=True, langfuse_public_trace=True),
        client=fake,
    )
    trace = tracer.start_trace("agent_run", run_id="failed-run")

    tracer.end_trace(
        trace,
        output={"completed": False},
        error="runtime failed",
    )
    tracer.flush()

    assert fake.root.ended == 1
    assert fake.root.public_calls == 0
    assert fake.current_public_calls == 0
    assert fake.publication_context_ids == []
    assert fake.trace_url_arguments == []
    assert trace.trace_public is False
    assert trace.trace_url is None
    assert fake.flushes == 1


def test_publication_failure_records_warning_without_fabricated_url():
    fake = FakeLangfuse(publication_fails=True)
    tracer = LangfuseAgentTracer(
        AppConfig(langfuse_enabled=True, langfuse_public_trace=True),
        client=fake,
    )
    trace = tracer.start_trace(
        "agent_run",
        run_id="publication-failure",
    )

    tracer.end_trace(trace, output={"completed": True})
    tracer.flush()

    assert fake.root.public_calls == 0
    assert fake.current_public_calls == 1
    assert fake.publication_context_ids == [trace.trace_id]
    assert fake.trace_url_arguments == []
    assert trace.trace_public is False
    assert trace.trace_url is None
    assert (
        trace.publication_error
        == "Public trace publication failed: RuntimeError"
    )
    assert fake.root.ended == 1
    assert fake.flushes == 1
