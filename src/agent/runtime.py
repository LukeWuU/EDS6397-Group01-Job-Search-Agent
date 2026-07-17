"""The one production LLM runtime and its continuous tool-calling loop."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from src.agent.client import (
    ChatModelClient,
    NormalizedAssistantMessage,
    NormalizedToolCall,
    OllamaChatModelClient,
)
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.state import (
    AgentPhase,
    AgentRunResult,
    AgentRunState,
    InvalidToolAttempt,
    StateInvariantError,
)
from src.agent.tool_registry import (
    AssignmentToolRegistry,
    ToolArgumentsError,
    ToolExecutionError,
    ToolRegistryError,
)
from src.config import AppConfig, load_config
from src.models.memory import CandidateMemory
from src.observability.tracing import (
    AgentTracer,
    SpanContext,
    TraceContext,
    build_agent_tracer,
)
from src.services.candidate_loader import load_candidate_bundle
from src.services.jobs_loader import load_jobs
from src.services.memory_loader import load_memory
from src.services.output_writer import (
    write_fit_analysis_files,
    write_job_details,
)
from src.workflow.human_review import (
    ReviewDecisionProvider,
    run_human_review_session,
)

MAX_MODEL_CALLS = 40
MAX_TOOL_CALLS = 60
MAX_CONSECUTIVE_INVALID_TURNS = 3


class AgentRuntimeError(Exception):
    """Base error for the one runtime."""


class AgentLoopLimitError(AgentRuntimeError):
    """Raised when model/tool safety limits are reached."""


class AgentModelResponseError(AgentRuntimeError):
    """Raised when the model cannot provide a usable workflow action."""


class JobSearchAgentRuntime:
    """The application's only production LLM-based agent."""

    def __init__(
        self,
        *,
        client: ChatModelClient,
        review_decision_provider: ReviewDecisionProvider,
        config: AppConfig,
        jobs_path: Path,
        profile_path: Path,
        portfolio_path: Path,
        evidence_path: Path,
        memory_path: Path,
        base_resume_tex_path: Path,
        base_resume_pdf_path: Path,
        run_workspace: Path,
        final_output_root: Path,
        tracer: AgentTracer | None = None,
        run_id: str | None = None,
    ) -> None:
        self.client = client
        self.review_decision_provider = review_decision_provider
        self.config = config
        self.jobs_path = jobs_path
        self.profile_path = profile_path
        self.portfolio_path = portfolio_path
        self.evidence_path = evidence_path
        self.memory_path = memory_path
        self.base_resume_tex_path = base_resume_tex_path
        self.base_resume_pdf_path = base_resume_pdf_path
        self.run_workspace = run_workspace.resolve()
        self.final_output_root = final_output_root.resolve()
        self.tracer = tracer or build_agent_tracer(config)
        self.run_id = run_id or f"run-{uuid.uuid4().hex}"

        self.state: AgentRunState | None = None
        self.registry: AssignmentToolRegistry | None = None
        self.trace: TraceContext | None = None
        self.conversation: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self._requested_tool_call_count = 0
        self._last_invalid_signature: str | None = None
        self._output_folders: dict[str, Path] = {}

    @property
    def model_name(self) -> str:
        return str(getattr(self.client, "model_name", self.config.ollama_model))

    def run(self) -> AgentRunResult:
        """Load inputs, run one continuous conversation, and package outputs."""
        self.run_workspace.mkdir(parents=True, exist_ok=True)
        self.final_output_root.mkdir(parents=True, exist_ok=True)
        self.trace = self.tracer.start_trace(
            "agent_run",
            run_id=self.run_id,
            input={
                "jobs_path": self.jobs_path,
                "profile_path": self.profile_path,
                "portfolio_path": self.portfolio_path,
                "evidence_path": self.evidence_path,
                "memory_path": self.memory_path,
                "final_output_root": self.final_output_root,
            },
            metadata={
                "model": self.model_name,
                "system_prompt": SYSTEM_PROMPT,
                "tool_names": list(AssignmentToolRegistry.TOOL_MODELS),
                "ollama_num_ctx": self.config.ollama_num_ctx,
                "ollama_temperature": self.config.ollama_temperature,
            },
        )
        failure: str | None = None
        loop_error: AgentLoopLimitError | None = None
        try:
            self._load_inputs()
            self._reasoning_loop()
        except AgentLoopLimitError as exc:
            failure = str(exc)
            loop_error = exc
            if self.state is not None:
                self.state.phase = AgentPhase.FAILED
                self.state.failure_reason = failure
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            if self.state is not None:
                self.state.phase = AgentPhase.FAILED
                self.state.failure_reason = failure
        provisional_result = self._build_result(failure)
        self.tracer.end_trace(
            self.trace,
            output=provisional_result.model_dump(mode="json"),
            error=failure,
        )
        self.tracer.flush()
        if loop_error is not None:
            raise loop_error
        return self._build_result(failure)

    def _load_inputs(self) -> None:
        assert self.trace is not None
        with self.tracer.span(
            self.trace,
            "input_loading",
            input={
                "jobs": self.jobs_path,
                "candidate": [
                    self.profile_path,
                    self.portfolio_path,
                    self.evidence_path,
                ],
                "memory": self.memory_path,
            },
        ) as span:
            jobs = load_jobs(self.jobs_path)
            bundle = load_candidate_bundle(
                self.profile_path,
                self.portfolio_path,
                self.evidence_path,
            )
            memory = load_memory(self.memory_path, bundle.profile.candidate_id)
            if not self.base_resume_tex_path.is_file():
                raise AgentRuntimeError(
                    f"Base resume LaTeX not found: {self.base_resume_tex_path}"
                )
            if not self.base_resume_pdf_path.is_file():
                raise AgentRuntimeError(
                    f"Base resume PDF not found: {self.base_resume_pdf_path}"
                )
            self.state = AgentRunState(
                run_id=self.run_id,
                candidate_id=bundle.profile.candidate_id,
                loaded_memory_candidate_id=memory.candidate_id,
                loaded_memory_fact_ids=[item.fact_id for item in memory.facts],
            )
            self.registry = AssignmentToolRegistry(
                state=self.state,
                jobs=jobs,
                bundle=bundle,
                memory=memory,
                base_resume_tex_path=self.base_resume_tex_path,
                base_resume_pdf_path=self.base_resume_pdf_path,
                run_workspace=self.run_workspace,
                tracer=self.tracer,
                trace=self.trace,
            )
            span.set_output(
                {
                    "job_count": len(jobs),
                    "candidate_id": bundle.profile.candidate_id,
                    "memory_fact_ids": [item.fact_id for item in memory.facts],
                }
            )

    def _reasoning_loop(self) -> None:
        assert self.state is not None
        assert self.registry is not None
        while not self.state.completed:
            if self.state.can_start_human_review():
                self._run_human_review()
                continue
            if (
                len(self.state.cover_letters) == 3
                and set(self.state.cover_letters) == set(self.state.top_3_job_ids)
            ):
                self.state.mark_completed()
                break
            self._append_state_snapshot()
            response = self._call_model(self.trace)
            self._append_assistant_message(response)
            valid_count, invalid_count = self._execute_response_calls(response)
            if not response.tool_calls:
                invalid_count = 1
                self._record_invalid(
                    tool_call=None,
                    error=(
                        "A text-only response is insufficient before completion; "
                        "request one currently valid assignment tool."
                    ),
                )
                self._append_invalid_message(
                    None,
                    "At least one valid tool call is required before completion.",
                )
            if valid_count:
                self.state.consecutive_invalid_call_count = 0
            elif invalid_count:
                self.state.consecutive_invalid_call_count += 1
                if (
                    self.state.consecutive_invalid_call_count
                    >= MAX_CONSECUTIVE_INVALID_TURNS
                ):
                    raise AgentLoopLimitError(
                        "Maximum consecutive invalid tool-call turns reached"
                    )

    def _append_state_snapshot(self) -> None:
        assert self.state is not None
        self.conversation.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "type": "current_state",
                        "state": self.state.snapshot(),
                        "instruction": (
                            "Request one or more currently valid calls from the five "
                            "assignment tools. Python owns all numerical scores."
                        ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )

    def _call_model(
        self,
        parent: TraceContext | SpanContext | None,
    ) -> NormalizedAssistantMessage:
        assert self.state is not None
        assert self.registry is not None
        assert self.trace is not None
        if self.state.model_call_count >= MAX_MODEL_CALLS:
            raise AgentLoopLimitError("Maximum model-call count reached")
        schemas = self.registry.model_schemas()
        call_number = self.state.model_call_count + 1
        with self.tracer.span(
            parent or self.trace,
            f"chat_model:{call_number}",
            input={
                "system_prompt": SYSTEM_PROMPT,
                "messages": self.conversation,
                "tools": schemas,
                "model": self.model_name,
                "configuration": {
                    "think": False,
                    "stream": False,
                    "num_ctx": self.config.ollama_num_ctx,
                    "temperature": self.config.ollama_temperature,
                },
            },
            metadata={"model_call_number": call_number},
            observation_type="generation",
        ) as span:
            response = self.client.chat(self.conversation, schemas)
            self.state.model_call_count = call_number
            span.set_output(response.model_dump(mode="json"))
        return response

    def _append_assistant_message(
        self,
        response: NormalizedAssistantMessage,
    ) -> None:
        self.conversation.append(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        **({"id": call.id} if call.id else {}),
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in response.tool_calls
                ],
            }
        )

    def _execute_response_calls(
        self,
        response: NormalizedAssistantMessage,
    ) -> tuple[int, int]:
        assert self.state is not None
        assert self.registry is not None
        valid = 0
        invalid = 0
        response_start_state = self.state.model_copy(deep=True)
        for call in response.tool_calls:
            self._requested_tool_call_count += 1
            if self._requested_tool_call_count > MAX_TOOL_CALLS:
                raise AgentLoopLimitError("Maximum requested tool-call count reached")
            try:
                baseline_arguments = self.registry.parse_arguments(
                    call.name, call.arguments
                )
                response_start_state.validate_tool_call(
                    call.name,
                    job_id=getattr(baseline_arguments, "job_id", None),
                )
                self._execute_model_tool_call(call)
            except (StateInvariantError, ToolRegistryError) as exc:
                invalid += 1
                self._record_invalid(tool_call=call, error=str(exc))
                self._append_invalid_message(call, str(exc))
            else:
                valid += 1
        return valid, invalid

    def _execute_model_tool_call(
        self,
        call: NormalizedToolCall,
        *,
        revision_round: int = 0,
        review_feedback: str | None = None,
        trace_parent: TraceContext | SpanContext | None = None,
    ):
        assert self.registry is not None
        assert self.trace is not None
        with self.tracer.span(
            trace_parent or self.trace,
            f"tool_call:{call.name}",
            input={
                "tool_call_id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            },
            metadata={
                "decision_summary": call.arguments.get("decision_summary", "")
            },
            observation_type="tool",
        ) as span:
            outcome = self.registry.execute(
                call.name,
                call.arguments,
                tool_call_id=call.id,
                revision_round=revision_round,
                review_feedback=review_feedback,
                trace_parent=span,
            )
            span.set_output(outcome.message_payload)
        self.conversation.append(
            {
                "role": "tool",
                "tool_name": call.name,
                **({"tool_call_id": call.id} if call.id else {}),
                "content": json.dumps(
                    outcome.message_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
        return outcome

    def _record_invalid(
        self,
        *,
        tool_call: NormalizedToolCall | None,
        error: str,
    ) -> None:
        assert self.state is not None
        name = tool_call.name if tool_call else "<missing_tool_call>"
        arguments = tool_call.arguments if tool_call else {}
        signature = json.dumps(
            {"name": name, "arguments": arguments},
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        self._last_invalid_signature = signature
        self.state.invalid_tool_attempts.append(
            InvalidToolAttempt(
                sequence=len(self.state.invalid_tool_attempts) + 1,
                model_call_number=self.state.model_call_count,
                tool_call_id=tool_call.id if tool_call else None,
                tool_name=name,
                arguments=arguments,
                error=error[:2000],
                valid_state=self.state.snapshot(),
            )
        )

    def _append_invalid_message(
        self,
        tool_call: NormalizedToolCall | None,
        error: str,
    ) -> None:
        assert self.state is not None
        payload = {
            "status": "invalid_tool_call",
            "requested_tool": tool_call.name if tool_call else None,
            "error": error[:2000],
            "current_valid_state": self.state.snapshot(),
            "instruction": "Correct the request using the same five-tool workflow.",
        }
        self.conversation.append(
            {
                "role": "tool" if tool_call else "user",
                **(
                    {
                        "tool_name": tool_call.name,
                        **(
                            {"tool_call_id": tool_call.id}
                            if tool_call.id
                            else {}
                        ),
                    }
                    if tool_call
                    else {}
                ),
                "content": json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ),
            }
        )

    def _run_human_review(self) -> None:
        assert self.state is not None
        assert self.registry is not None
        assert self.trace is not None
        self.state.begin_human_review()
        initial_drafts = [
            self.state.draft_resumes[job_id] for job_id in self.state.top_3_job_ids
        ]
        with self.tracer.span(
            self.trace,
            "human_review_pause",
            input={
                "job_ids": self.state.top_3_job_ids,
                "draft_revisions": {
                    item.job_id: item.revision_round for item in initial_drafts
                },
            },
        ) as review_span:

            def traced_provider(pending_drafts, session_state):
                with self.tracer.span(
                    review_span,
                    f"review_decision_round:{session_state.provider_call_count}",
                    input={
                        "pending_job_ids": session_state.pending_job_ids,
                        "pause_count": session_state.pause_count,
                    },
                ) as decision_span:
                    decisions = self.review_decision_provider(
                        pending_drafts, session_state
                    )
                    decision_span.set_output(
                        [
                            item.model_dump(mode="json")
                            if hasattr(item, "model_dump")
                            else str(item)
                            for item in decisions
                        ]
                    )
                    return decisions

            def revision_handler(
                job_id: str,
                previous,
                review_comments: str,
                updated_memory: CandidateMemory,
                next_revision_round: int,
            ):
                self.registry.set_memory(updated_memory)
                with self.tracer.span(
                    review_span,
                    f"revision_call:{job_id}:r{next_revision_round}",
                    input={
                        "job_id": job_id,
                        "review_feedback": review_comments,
                        "memory_fact_ids": [
                            item.fact_id for item in updated_memory.facts
                        ],
                    },
                ) as revision_span:
                    result = self._request_revision(
                        job_id=job_id,
                        review_comments=review_comments,
                        updated_memory=updated_memory,
                        next_revision_round=next_revision_round,
                        trace_parent=revision_span,
                    )
                    revision_span.set_output(
                        {
                            "job_id": result.job_id,
                            "revision_round": result.revision_round,
                            "page_count": result.page_count,
                        }
                    )
                    return result

            result = run_human_review_session(
                initial_drafts=initial_drafts,
                memory=self.registry.memory,
                memory_path=self.memory_path,
                final_output_root=self.final_output_root,
                decision_provider=traced_provider,
                revision_handler=revision_handler,
            )
            with self.tracer.span(
                review_span,
                "memory_write",
                input={"memory_path": self.memory_path},
            ) as memory_span:
                memory_span.set_output(
                    {
                        "learned_fact_ids": result.learned_fact_ids,
                        "final_fact_count": len(result.final_memory.facts),
                    }
                )
            with self.tracer.span(
                review_span,
                "resume_finalization",
                input={"approved_job_ids": result.initial_job_ids},
            ) as finalization_span:
                finalization_span.set_output(
                    {
                        "finalization_count": result.finalization_count,
                        "folders": [
                            item.destination_dir for item in result.finalized_resumes
                        ],
                    }
                )
            self.registry.set_memory(result.final_memory)
            self.state.apply_human_review(result)
            review_span.set_output(
                {
                    "pause_count": result.pause_count,
                    "all_approved": result.all_approved,
                    "learned_fact_ids": result.learned_fact_ids,
                    "revision_count_by_job": result.revision_count_by_job,
                }
            )
        self._package_final_folders()
        self.conversation.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "type": "human_review_outcome",
                        "summary": result.audit_summary,
                        "learned_memory_fact_ids": result.learned_fact_ids,
                        "finalized_job_ids": list(self.state.finalized_resumes),
                        "instruction": (
                            "Continue this same conversation by generating one "
                            "evidence-grounded cover letter for each Top 3 job."
                        ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )

    def _request_revision(
        self,
        *,
        job_id: str,
        review_comments: str,
        updated_memory: CandidateMemory,
        next_revision_round: int,
        trace_parent: SpanContext,
    ):
        assert self.state is not None
        self.conversation.append(
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "type": "resume_revision_request",
                        "job_id": job_id,
                        "required_revision_round": next_revision_round,
                        "exact_review_feedback": review_comments,
                        "updated_memory": [
                            {
                                "fact_id": fact.fact_id,
                                "fact_type": fact.fact_type,
                                "statement": fact.statement,
                                "normalized_value": fact.normalized_value,
                                "skill_tags": fact.skill_tags,
                            }
                            for fact in updated_memory.facts
                        ],
                        "instruction": (
                            "Use tailor_resume for this same job with a corrected "
                            "evidence-grounded edit_plan. The runtime supplies the "
                            "required revision round and exact feedback."
                        ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
        invalid_turns = 0
        while True:
            response = self._call_model(trace_parent)
            self._append_assistant_message(response)
            if len(response.tool_calls) != 1:
                error = (
                    "A review revision response must contain exactly one "
                    "tailor_resume call"
                )
                self._record_invalid(tool_call=None, error=error)
                self._append_invalid_message(None, error)
                invalid_turns += 1
            else:
                call = response.tool_calls[0]
                self._requested_tool_call_count += 1
                if self._requested_tool_call_count > MAX_TOOL_CALLS:
                    raise AgentLoopLimitError(
                        "Maximum requested tool-call count reached"
                    )
                if (
                    call.name != "tailor_resume"
                    or call.arguments.get("job_id") != job_id
                ):
                    error = (
                        "Revision must call tailor_resume for the rejected job "
                        f"{job_id!r}"
                    )
                    self._record_invalid(tool_call=call, error=error)
                    self._append_invalid_message(call, error)
                    invalid_turns += 1
                else:
                    try:
                        outcome = self._execute_model_tool_call(
                            call,
                            revision_round=next_revision_round,
                            review_feedback=review_comments,
                            trace_parent=trace_parent,
                        )
                    except (StateInvariantError, ToolRegistryError) as exc:
                        self._record_invalid(tool_call=call, error=str(exc))
                        self._append_invalid_message(call, str(exc))
                        invalid_turns += 1
                    else:
                        self.state.consecutive_invalid_call_count = 0
                        return outcome.result
            self.state.consecutive_invalid_call_count += 1
            if invalid_turns >= MAX_CONSECUTIVE_INVALID_TURNS:
                raise AgentLoopLimitError(
                    "Maximum invalid revision tool-call turns reached"
                )

    def _package_final_folders(self) -> None:
        assert self.state is not None
        assert self.registry is not None
        assert self.trace is not None
        with self.tracer.span(
            self.trace,
            "output_packaging",
            input={"job_ids": self.state.top_3_job_ids},
        ) as span:
            for job_id in self.state.top_3_job_ids:
                finalized = self.state.finalized_resumes[job_id]
                job = self.registry._job(job_id)
                score = self.registry._score(job_id)
                write_job_details(finalized.destination_dir, job, score)
                write_fit_analysis_files(
                    finalized.destination_dir,
                    self.state.fit_analyses[job_id],
                )
                self._output_folders[job_id] = finalized.destination_dir
            span.set_output(
                {
                    job_id: sorted(path.name for path in folder.iterdir())
                    for job_id, folder in self._output_folders.items()
                }
            )

    def _build_result(self, failure: str | None) -> AgentRunResult:
        state = self.state
        if state is None:
            return AgentRunResult(
                run_id=self.run_id,
                completed=False,
                failure_reason=failure or "Inputs were not loaded",
                model_name=self.model_name,
                model_call_count=0,
                tool_call_count=0,
                invalid_tool_attempt_count=0,
                tool_execution_records=[],
                top_3_job_ids=[],
                top_3_scores={},
                fit_analysis_count=0,
                draft_resume_count=0,
                pause_count=0,
                learned_memory_fact_ids=[],
                finalized_resume_count=0,
                cover_letter_count=0,
                output_folders={},
                trace_id=self.trace.trace_id if self.trace else None,
                trace_url=self.trace.trace_url if self.trace else None,
                state_summary={},
            )
        scores = {
            item.job_id: item.final_score
            for item in (state.scoring_result.top_3 if state.scoring_result else [])
        }
        learned = (
            state.human_review.learned_fact_ids if state.human_review else []
        )
        return AgentRunResult(
            run_id=self.run_id,
            completed=state.completed,
            failure_reason=failure or state.failure_reason,
            model_name=self.model_name,
            model_call_count=state.model_call_count,
            tool_call_count=state.tool_call_count,
            invalid_tool_attempt_count=len(state.invalid_tool_attempts),
            tool_execution_records=state.tool_execution_history,
            top_3_job_ids=state.top_3_job_ids,
            top_3_scores=scores,
            fit_analysis_count=len(state.fit_analyses),
            draft_resume_count=len(state.draft_resumes),
            pause_count=state.pause_count,
            learned_memory_fact_ids=learned,
            finalized_resume_count=len(state.finalized_resumes),
            cover_letter_count=len(state.cover_letters),
            output_folders=self._output_folders,
            trace_id=self.trace.trace_id if self.trace else None,
            trace_url=self.trace.trace_url if self.trace else None,
            state_summary=state.snapshot(),
        )


def run_job_search_agent(
    *,
    review_decision_provider: ReviewDecisionProvider,
    jobs_path: Path = Path("data/AI_ML_Jobs_Dataset_20.csv"),
    profile_path: Path = Path("candidate/profile.json"),
    portfolio_path: Path = Path("candidate/portfolio.json"),
    evidence_path: Path = Path("candidate/evidence_registry.json"),
    memory_path: Path = Path("memory.json"),
    base_resume_tex_path: Path = Path("candidate/sample_resume.tex"),
    base_resume_pdf_path: Path = Path("candidate/sample_resume.pdf"),
    run_workspace: Path = Path(".runtime"),
    final_output_root: Path = Path("outputs"),
    config: AppConfig | None = None,
    client: ChatModelClient | None = None,
    tracer: AgentTracer | None = None,
    run_id: str | None = None,
) -> AgentRunResult:
    """Convenience entry point for the one production runtime."""
    resolved_config = config or load_config()
    resolved_client = client or OllamaChatModelClient(resolved_config)
    runtime = JobSearchAgentRuntime(
        client=resolved_client,
        review_decision_provider=review_decision_provider,
        config=resolved_config,
        jobs_path=jobs_path,
        profile_path=profile_path,
        portfolio_path=portfolio_path,
        evidence_path=evidence_path,
        memory_path=memory_path,
        base_resume_tex_path=base_resume_tex_path,
        base_resume_pdf_path=base_resume_pdf_path,
        run_workspace=run_workspace,
        final_output_root=final_output_root,
        tracer=tracer,
        run_id=run_id,
    )
    return runtime.run()


__all__ = [
    "AgentLoopLimitError",
    "AgentModelResponseError",
    "AgentRuntimeError",
    "JobSearchAgentRuntime",
    "MAX_CONSECUTIVE_INVALID_TURNS",
    "MAX_MODEL_CALLS",
    "MAX_TOOL_CALLS",
    "run_job_search_agent",
]
