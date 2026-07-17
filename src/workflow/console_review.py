"""Interactive console decision provider for the existing Human Review workflow."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import ValidationError

from src.tools.resume_tailoring import ResumeEditCategory, ResumeTailoringResult
from src.workflow.human_review import (
    HumanReviewSessionState,
    ResumeReviewDecision,
    ReviewDecisionType,
    ReviewFactInput,
    ReviewFactType,
)


class ConsoleReviewAbort(RuntimeError):
    """Raised when console input ends or the reviewer interrupts the session."""


class ConsoleReviewDecisionProvider:
    """Collect typed decisions within the existing single review session."""

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] | None = None,
        output_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self._input = input_fn or builtins.input
        self._output = output_fn or builtins.print
        self._session_id: str | None = None
        self.call_count = 0

    def _write(self, message: str = "") -> None:
        self._output(message)

    def _ask(self, prompt: str) -> str:
        try:
            return self._input(prompt)
        except EOFError as exc:
            raise ConsoleReviewAbort(
                "Human Review aborted because console input ended"
            ) from exc
        except KeyboardInterrupt:
            raise

    @staticmethod
    def _citation_ids(change: Any) -> list[str]:
        identifiers: list[str] = []
        for citation in change.citations:
            identifier = citation.evidence_id or citation.source_id
            if identifier and identifier not in identifiers:
                identifiers.append(identifier)
        return identifiers

    def _display_draft(self, draft: ResumeTailoringResult) -> None:
        self._write("=" * 72)
        self._write(f"Job ID: {draft.job_id}")
        self._write(f"Title: {draft.title}")
        self._write(f"Company: {draft.company}")
        self._write(f"Revision: {draft.revision_round}")
        self._write(f"Draft PDF: {draft.draft_pdf_path}")
        self._write(f"Change log: {draft.change_log_path}")
        self._write(f"Change count: {draft.change_count}")
        for index, change in enumerate(draft.changes, start=1):
            category = (
                change.category.value
                if hasattr(change.category, "value")
                else str(change.category)
            )
            self._write(f"Change {index} [{category}] {change.target_id}")
            self._write(f"  Before: {change.before}")
            self._write(f"  After: {change.after}")
            citation_ids = self._citation_ids(change)
            self._write(
                "  Evidence citations: "
                + (", ".join(citation_ids) if citation_ids else "none")
            )
            if change.category == ResumeEditCategory.PROJECT_SWAP:
                self._write(f"  Project swap before: {change.before}")
                self._write(f"  Project swap after: {change.after}")

    def _ask_decision(self, draft: ResumeTailoringResult) -> tuple[ReviewDecisionType, str]:
        while True:
            raw = self._ask(
                f"Decision for {draft.company} — {draft.title} [approve/reject]: "
            ).strip().casefold()
            if raw in {"approve", "a"}:
                comments = self._ask(
                    "Optional comments, or press Enter: "
                ).strip()
                return ReviewDecisionType.APPROVE, comments
            if raw in {"reject", "r"}:
                while True:
                    comments = self._ask(
                        "Comments explaining required changes: "
                    ).strip()
                    if comments:
                        return ReviewDecisionType.REJECT, comments
                    self._write("Reject decisions require nonempty comments.")
            self._write("Enter approve/a or reject/r.")

    def _ask_yes_no(self, prompt: str) -> bool:
        while True:
            raw = self._ask(prompt).strip().casefold()
            if raw in {"", "n", "no"}:
                return False
            if raw in {"y", "yes"}:
                return True
            self._write("Enter y/yes or n/no.")

    def _ask_fact_type(self) -> ReviewFactType:
        while True:
            raw = self._ask("Fact type [skill/candidate_fact]: ").strip().casefold()
            if raw == "skill":
                return ReviewFactType.SKILL
            if raw == "candidate_fact":
                return ReviewFactType.CANDIDATE_FACT
            self._write("Fact type must be skill or candidate_fact.")

    def _ask_nonempty(self, prompt: str) -> str:
        while True:
            value = self._ask(prompt).strip()
            if value:
                return value
            self._write("A nonempty value is required.")

    @staticmethod
    def _parse_skill_tags(raw: str) -> list[str]:
        by_key: dict[str, str] = {}
        for item in raw.split(","):
            cleaned = " ".join(item.split())
            if cleaned:
                by_key.setdefault(cleaned.casefold(), cleaned)
        return [by_key[key] for key in sorted(by_key)]

    def _collect_one_fact(self) -> ReviewFactInput:
        while True:
            fact_type = self._ask_fact_type()
            statement = self._ask_nonempty("Statement: ")
            normalized_value = self._ask("Normalized value: ").strip()
            skill_tags = self._parse_skill_tags(
                self._ask(
                    "Skill tags (comma-separated, optional for candidate_fact): "
                )
            )
            try:
                return ReviewFactInput(
                    fact_type=fact_type,
                    statement=statement,
                    normalized_value=normalized_value,
                    skill_tags=skill_tags,
                )
            except ValidationError as exc:
                self._write(f"Invalid candidate fact: {exc.errors()[0]['msg']}")

    def _collect_facts(self) -> list[ReviewFactInput]:
        if not self._ask_yes_no("Add a new candidate fact from this review? [y/N] "):
            return []
        facts = [self._collect_one_fact()]
        while self._ask_yes_no("Add another fact? [y/N] "):
            facts.append(self._collect_one_fact())
        return facts

    def __call__(
        self,
        pending_drafts: Sequence[ResumeTailoringResult],
        session_state: HumanReviewSessionState,
    ) -> Sequence[ResumeReviewDecision]:
        """Display the whole batch, then collect one explicit decision per draft."""
        if self._session_id is None:
            self._session_id = session_state.session_id
        elif self._session_id != session_state.session_id:
            raise ConsoleReviewAbort(
                "Console provider cannot be reused for a different Human Review session"
            )
        self.call_count += 1
        self._write(
            f"Human Review — session {session_state.session_id}, "
            f"decision round {session_state.provider_call_count}"
        )
        self._write("Review every pending resume before entering decisions.")
        for draft in pending_drafts:
            self._display_draft(draft)
        self._write("=" * 72)

        decisions: list[ResumeReviewDecision] = []
        for draft in pending_drafts:
            decision, comments = self._ask_decision(draft)
            learned_facts = self._collect_facts()
            decisions.append(
                ResumeReviewDecision(
                    job_id=draft.job_id,
                    decision=decision,
                    comments=comments,
                    learned_facts=learned_facts,
                )
            )
        return decisions


__all__ = [
    "ConsoleReviewAbort",
    "ConsoleReviewDecisionProvider",
]
