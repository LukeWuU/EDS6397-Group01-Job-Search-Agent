"""Production argparse entry point for local preflight and agent execution."""

from __future__ import annotations

import argparse
import builtins
import traceback
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import src.agent.runtime as runtime_module
from src.agent.client import OllamaChatModelClient
from src.agent.runtime import run_job_search_agent
from src.agent.state import AgentRunResult
from src.config import AppConfig, load_config
from src.observability.tracing import build_agent_tracer
from src.services.preflight import (
    PathSafetyError,
    PreflightResult,
    find_completed_output_files,
    format_preflight_result,
    run_preflight,
    validate_run_paths,
)
from src.workflow.console_review import ConsoleReviewDecisionProvider

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_JOBS = REPOSITORY_ROOT / "data" / "AI_ML_Jobs_Dataset_20.csv"
DEFAULT_PROFILE = REPOSITORY_ROOT / "candidate" / "profile.json"
DEFAULT_PORTFOLIO = REPOSITORY_ROOT / "candidate" / "portfolio.json"
DEFAULT_EVIDENCE = REPOSITORY_ROOT / "candidate" / "evidence_registry.json"
DEFAULT_BASE_RESUME_TEX = REPOSITORY_ROOT / "candidate" / "sample_resume.tex"
DEFAULT_BASE_RESUME_PDF = REPOSITORY_ROOT / "candidate" / "sample_resume.pdf"
DEFAULT_MEMORY = REPOSITORY_ROOT / "memory.json"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"


def _positive_integer(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def _add_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--portfolio", type=Path, default=DEFAULT_PORTFOLIO)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument(
        "--base-resume-tex",
        type=Path,
        default=DEFAULT_BASE_RESUME_TEX,
    )
    parser.add_argument(
        "--base-resume-pdf",
        type=Path,
        default=DEFAULT_BASE_RESUME_PDF,
    )
    parser.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    """Build the standard-library CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Evidence-grounded single-agent job search workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Validate local inputs and services without running the agent.",
    )
    _add_path_arguments(preflight_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the single Job Search Agent after local preflight.",
    )
    _add_path_arguments(run_parser)
    run_parser.add_argument(
        "--max-model-calls",
        type=_positive_integer,
        default=runtime_module.MAX_MODEL_CALLS,
    )
    run_parser.add_argument(
        "--max-tool-calls",
        type=_positive_integer,
        default=runtime_module.MAX_TOOL_CALLS,
    )
    run_parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip service checks; path and overwrite safety remain enforced.",
    )
    run_parser.add_argument(
        "--yes",
        action="store_true",
        help="Bypass only the initial run-start confirmation.",
    )
    run_parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a traceback for runtime failures.",
    )
    return parser


def _resolved_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "jobs_path": args.jobs.expanduser().resolve(),
        "profile_path": args.profile.expanduser().resolve(),
        "portfolio_path": args.portfolio.expanduser().resolve(),
        "evidence_path": args.evidence.expanduser().resolve(),
        "base_resume_tex_path": args.base_resume_tex.expanduser().resolve(),
        "base_resume_pdf_path": args.base_resume_pdf.expanduser().resolve(),
        "memory_path": args.memory.expanduser().resolve(),
        # Preserve the final path component until the safety check can reject
        # an output-directory symlink; the validator returns the resolved path.
        "output_root": args.output_root.expanduser().absolute(),
    }


def _preflight_for_args(
    args: argparse.Namespace,
    config: AppConfig,
    paths: dict[str, Path],
) -> PreflightResult:
    return run_preflight(
        config=config,
        repository_root=REPOSITORY_ROOT,
        jobs_path=paths["jobs_path"],
        profile_path=paths["profile_path"],
        portfolio_path=paths["portfolio_path"],
        evidence_path=paths["evidence_path"],
        base_resume_tex_path=paths["base_resume_tex_path"],
        base_resume_pdf_path=paths["base_resume_pdf_path"],
        memory_path=paths["memory_path"],
        output_root=paths["output_root"],
    )


@contextmanager
def _runtime_limits(max_model_calls: int, max_tool_calls: int):
    """Apply existing runtime limits for one call and always restore them."""
    previous_model = runtime_module.MAX_MODEL_CALLS
    previous_tool = runtime_module.MAX_TOOL_CALLS
    runtime_module.MAX_MODEL_CALLS = max_model_calls
    runtime_module.MAX_TOOL_CALLS = max_tool_calls
    try:
        yield
    finally:
        runtime_module.MAX_MODEL_CALLS = previous_model
        runtime_module.MAX_TOOL_CALLS = previous_tool


def _print_success(result: AgentRunResult) -> None:
    print("Agent run completed")
    print(f"Run ID: {result.run_id}")
    print(f"Model: {result.model_name}")
    print(f"Model calls: {result.model_call_count}")
    print(f"Tool calls: {result.tool_call_count}")
    print(f"Invalid tool attempts: {result.invalid_tool_attempt_count}")
    print("Top 3: " + (", ".join(result.top_3_job_ids) or "none"))
    print(f"Fit analyses: {result.fit_analysis_count}")
    print(f"Resume drafts: {result.draft_resume_count}")
    print(f"Human Review pauses: {result.pause_count}")
    print(
        "Learned memory facts: "
        + (", ".join(result.learned_memory_fact_ids) or "none")
    )
    print(f"Final resumes: {result.finalized_resume_count}")
    print(f"Cover letters: {result.cover_letter_count}")
    print("Output folders:")
    if result.output_folders:
        for job_id, folder in result.output_folders.items():
            print(f"  {job_id}: {folder}")
    else:
        print("  none")
    print(f"Trace ID: {result.trace_id or 'not available'}")
    print(f"Trace URL: {result.trace_url or 'not available'}")


def _print_failure(result: AgentRunResult) -> None:
    print("Agent run failed")
    print(f"Failure: {result.failure_reason or 'unknown failure'}")
    print(f"Phase: {result.state_summary.get('phase', 'unknown')}")
    print(f"Model calls: {result.model_call_count}")
    print(f"Tool calls: {result.tool_call_count}")
    print(f"Trace ID: {result.trace_id or 'not available'}")


def _run_preflight_command(args: argparse.Namespace, config: AppConfig) -> int:
    paths = _resolved_paths(args)
    result = _preflight_for_args(args, config, paths)
    print(format_preflight_result(result))
    return 0 if result.succeeded else 1


def _run_agent_command(args: argparse.Namespace, config: AppConfig) -> int:
    paths = _resolved_paths(args)
    try:
        resolved_output, resolved_memory = validate_run_paths(
            REPOSITORY_ROOT,
            paths["output_root"],
            paths["memory_path"],
        )
    except PathSafetyError as exc:
        print("Agent run failed")
        print(f"Failure: {exc}")
        print("Phase: preflight")
        print("Model calls: 0")
        print("Tool calls: 0")
        print("Trace ID: not available")
        return 1
    paths["output_root"] = resolved_output
    paths["memory_path"] = resolved_memory

    if args.skip_preflight:
        print(
            "[WARNING] Preflight skipped; local services and input integrity "
            "were not validated."
        )
    else:
        preflight = _preflight_for_args(args, config, paths)
        print(format_preflight_result(preflight))
        if not preflight.succeeded:
            print("Agent run aborted because preflight reported a FAIL check.")
            return 1

    collisions = find_completed_output_files(paths["output_root"])
    if collisions:
        print("Agent run failed")
        print(
            "Failure: output root contains completed job files that would "
            "conflict with a new run"
        )
        for path in collisions:
            print(f"  {path}")
        print("Phase: preflight")
        print("Model calls: 0")
        print("Tool calls: 0")
        print("Trace ID: not available")
        return 1

    print(f"Memory path that Human Review may change: {paths['memory_path']}")
    print(f"Output root that the run may change: {paths['output_root']}")
    if not args.yes:
        try:
            consent = builtins.input("Start the agent run? [y/N] ").strip().casefold()
        except EOFError:
            consent = ""
        if consent not in {"y", "yes"}:
            print("Agent run cancelled before creating the model client.")
            return 0

    client = OllamaChatModelClient(config)
    tracer = build_agent_tracer(config)
    review_provider = ConsoleReviewDecisionProvider()
    try:
        with _runtime_limits(args.max_model_calls, args.max_tool_calls):
            result = run_job_search_agent(
                review_decision_provider=review_provider,
                jobs_path=paths["jobs_path"],
                profile_path=paths["profile_path"],
                portfolio_path=paths["portfolio_path"],
                evidence_path=paths["evidence_path"],
                memory_path=paths["memory_path"],
                base_resume_tex_path=paths["base_resume_tex_path"],
                base_resume_pdf_path=paths["base_resume_pdf_path"],
                run_workspace=paths["output_root"] / ".runtime",
                final_output_root=paths["output_root"],
                config=config,
                client=client,
                tracer=tracer,
            )
    except KeyboardInterrupt:
        print("\nAgent run interrupted.")
        return 130
    except Exception as exc:
        print("Agent run failed")
        print(f"Failure: {type(exc).__name__}: {exc}")
        print("Phase: runtime")
        print("Model calls: unavailable")
        print("Tool calls: unavailable")
        print("Trace ID: not available")
        if args.debug:
            traceback.print_exc()
        return 1

    if result.completed:
        _print_success(result)
        return 0
    _print_failure(result)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and execute one requested command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(env_path=REPOSITORY_ROOT / ".env")
        if args.command == "preflight":
            return _run_preflight_command(args, config)
        return _run_agent_command(args, config)
    except KeyboardInterrupt:
        print("\nOperation interrupted.")
        return 130
    except Exception as exc:
        print(f"Command failed: {type(exc).__name__}: {exc}")
        if getattr(args, "debug", False):
            traceback.print_exc()
        return 1


__all__ = ["build_parser", "main"]
