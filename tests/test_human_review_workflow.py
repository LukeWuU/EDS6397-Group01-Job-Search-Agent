"""Focused tests for the single continuous Human Review workflow."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.memory import CandidateMemory
from src.services.memory_loader import load_memory
from src.services.memory_store import save_memory_atomic
from src.tools.resume_tailoring import (
    CompilationResult,
    ResumeChange,
    ResumeEditCategory,
    ResumeTailoringResult,
)
from src.workflow.human_review import (
    DuplicatePauseError,
    InvalidReviewBatchError,
    ResumeReviewDecision,
    ReviewDecisionType,
    ReviewFactInput,
    ReviewFactType,
    RevisionLimitError,
    RevisionResultMismatchError,
    run_human_review_session,
)

BASE_PDF = ROOT / "candidate" / "sample_resume.pdf"


def _memory() -> CandidateMemory:
    return CandidateMemory(
        schema_version="1.0",
        candidate_id="cand-mira-solenne-001",
        facts=[],
    )


def _draft(
    directory: Path,
    job_id: str,
    revision_round: int = 0,
    feedback: str | None = None,
) -> ResumeTailoringResult:
    directory.mkdir(parents=True, exist_ok=True)
    before_pdf = directory / "resume_before.pdf"
    draft_pdf = directory / f"resume_draft_r{revision_round}.pdf"
    draft_tex = directory / f"resume_draft_r{revision_round}.tex"
    change_log = directory / f"change_log_r{revision_round}.json"
    shutil.copyfile(BASE_PDF, before_pdf)
    shutil.copyfile(BASE_PDF, draft_pdf)
    draft_tex.write_text(f"% {job_id} revision {revision_round}\n", encoding="utf-8")
    tex_hash = hashlib.sha256(draft_tex.read_bytes()).hexdigest()
    protected = hashlib.sha256(b"same protected regions").hexdigest()
    changes = [
        ResumeChange(
            category=ResumeEditCategory.PROFESSIONAL_SUMMARY,
            target_id="summary",
            before="summary before",
            after="summary after",
            reason="review plan",
            citations=[],
        ),
        ResumeChange(
            category=ResumeEditCategory.EXPERIENCE_BULLET,
            target_id="exp-primary-bullet-1",
            before="bullet before 1",
            after="bullet after 1",
            reason="review plan",
            citations=[],
        ),
        ResumeChange(
            category=ResumeEditCategory.EXPERIENCE_BULLET,
            target_id="exp-primary-bullet-2",
            before="bullet before 2",
            after="bullet after 2",
            reason="review plan",
            citations=[],
        ),
    ]
    result = ResumeTailoringResult(
        job_id=job_id,
        title="AI Engineer",
        company=f"Fictional {job_id}",
        revision_round=revision_round,
        review_feedback=feedback,
        edit_categories=[
            ResumeEditCategory.PROFESSIONAL_SUMMARY,
            ResumeEditCategory.EXPERIENCE_BULLET,
        ],
        changes=changes,
        change_count=3,
        summary_change_count=1,
        experience_bullet_change_count=2,
        skill_change_count=0,
        project_swap_change_count=0,
        base_resume_pdf_path=before_pdf,
        draft_tex_path=draft_tex,
        draft_pdf_path=draft_pdf,
        change_log_path=change_log,
        compilation=CompilationResult(
            command=["pdflatex"],
            return_code=0,
            pdf_path=draft_pdf,
            page_count=1,
            stdout_tail="",
            stderr_tail="",
        ),
        page_count=1,
        evidence_citation_count=0,
        base_tex_sha256=hashlib.sha256(b"base").hexdigest(),
        tailored_tex_sha256=tex_hash,
        protected_regions_sha256_before=protected,
        protected_regions_sha256_after=protected,
        protected_regions_unchanged=True,
        deterministic_plan_digest=hashlib.sha256(
            f"{job_id}:{revision_round}".encode()
        ).hexdigest(),
    )
    change_log.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _three_drafts(tmp_path: Path) -> list[ResumeTailoringResult]:
    return [
        _draft(tmp_path / "drafts" / job_id / "r0", job_id)
        for job_id in ("job-a", "job-b", "job-c")
    ]


def _run(tmp_path: Path, drafts, provider, handler):
    tmp_path.mkdir(parents=True, exist_ok=True)
    memory_path = tmp_path / "memory.json"
    save_memory_atomic(_memory(), memory_path)
    return run_human_review_session(
        drafts,
        _memory(),
        memory_path,
        tmp_path / "final",
        provider,
        handler,
    )


def test_requires_exactly_three_distinct_r0_drafts(tmp_path: Path) -> None:
    drafts = _three_drafts(tmp_path)
    with pytest.raises(InvalidReviewBatchError, match="exactly 3"):
        _run(tmp_path / "two", drafts[:2], lambda *_: [], lambda *_: None)
    duplicate = [
        drafts[0],
        _draft(tmp_path / "duplicate", "job-a"),
        drafts[2],
    ]
    with pytest.raises(InvalidReviewBatchError, match="distinct"):
        _run(tmp_path / "dupe", duplicate, lambda *_: [], lambda *_: None)
    r1 = _draft(tmp_path / "r1", "job-z", 1, "feedback")
    with pytest.raises(InvalidReviewBatchError, match="revision_round 0"):
        _run(tmp_path / "wrong-round", [drafts[0], drafts[1], r1], lambda *_: [], lambda *_: None)


def test_three_immediate_approvals_single_pause_and_finalization(tmp_path: Path) -> None:
    drafts = _three_drafts(tmp_path)
    calls = []

    def provider(pending, state):
        calls.append((list(pending), state))
        return [
            ResumeReviewDecision(job_id=draft.job_id, decision="approve")
            for draft in pending
        ]

    def no_revision(*_):
        raise AssertionError("revision handler must not run for approvals")

    result = _run(tmp_path, drafts, provider, no_revision)
    assert result.completed and result.all_approved
    assert result.pause_count == 1
    assert result.decision_provider_call_count == 1
    assert len(calls) == 1
    assert len(calls[0][0]) == 3
    assert all(draft.change_log_path.is_file() for draft in calls[0][0])
    assert result.finalization_count == 3
    assert result.approved_revision_by_job == {"job-a": 0, "job-b": 0, "job-c": 0}
    for finalized in result.finalized_resumes:
        assert finalized.resume_after_pdf_path.is_file()
        assert {path.name for path in finalized.destination_dir.iterdir()} == {
            "resume_before.pdf",
            "resume_after.tex",
            "resume_after.pdf",
            "resume_change_log.json",
        }

    with pytest.raises(DuplicatePauseError):
        run_human_review_session(
            drafts,
            _memory(),
            tmp_path / "memory.json",
            tmp_path / "final",
            provider,
            no_revision,
        )
    assert len(calls) == 1


def test_graphql_persists_before_b_and_c_revisions(tmp_path: Path) -> None:
    drafts = _three_drafts(tmp_path)
    provider_calls = []
    memories_seen: dict[str, list[str]] = {}
    comments_seen: dict[str, str] = {}

    def provider(pending, state):
        provider_calls.append((pending, state.pause_count))
        if len(provider_calls) == 1:
            return [
                ResumeReviewDecision(
                    job_id="job-a",
                    decision=ReviewDecisionType.APPROVE,
                    learned_facts=[
                        ReviewFactInput(
                            fact_type=ReviewFactType.SKILL,
                            statement="GraphQL is a skill I know.",
                            normalized_value="GraphQL",
                            skill_tags=["GraphQL"],
                        )
                    ],
                ),
                ResumeReviewDecision(
                    job_id="job-b",
                    decision=ReviewDecisionType.REJECT,
                    comments="Surface GraphQL in Resume B.",
                ),
                ResumeReviewDecision(
                    job_id="job-c",
                    decision=ReviewDecisionType.REJECT,
                    comments="Surface the remembered GraphQL skill in Resume C.",
                ),
            ]
        return [
            ResumeReviewDecision(job_id=draft.job_id, decision="approve")
            for draft in pending
        ]

    def revision_handler(job_id, previous, comments, updated_memory, next_round):
        memories_seen[job_id] = [
            tag
            for fact in updated_memory.facts
            if fact.fact_type == "skill"
            for tag in fact.skill_tags
        ]
        comments_seen[job_id] = comments
        persisted = load_memory(tmp_path / "memory.json", updated_memory.candidate_id)
        assert any("GraphQL" in fact.skill_tags for fact in persisted.facts)
        return _draft(
            tmp_path / "revisions" / job_id / f"r{next_round}",
            job_id,
            next_round,
            comments,
        )

    result = _run(tmp_path, drafts, provider, revision_handler)
    loaded = load_memory(tmp_path / "memory.json", result.final_memory.candidate_id)
    graphql = next(fact for fact in loaded.facts if "GraphQL" in fact.skill_tags)
    assert result.pause_count == 1
    assert len(provider_calls) == 2
    assert all(pause == 1 for _, pause in provider_calls)
    assert memories_seen == {"job-b": ["GraphQL"], "job-c": ["GraphQL"]}
    assert comments_seen == {
        "job-b": "Surface GraphQL in Resume B.",
        "job-c": "Surface the remembered GraphQL skill in Resume C.",
    }
    assert graphql.provenance.review_round == 1
    assert graphql.evidence_refs == ["job-a"]
    assert result.revision_count_by_job == {"job-a": 0, "job-b": 1, "job-c": 1}
    assert result.approved_revision_by_job == {"job-a": 0, "job-b": 1, "job-c": 1}
    assert result.finalization_count == 3
    assert any(record.learned_fact_ids == [graphql.fact_id] for record in result.review_round_records)
    assert [record.decision for record in result.review_round_records].count("reject") == 2


def test_two_revisions_allowed_but_third_is_rejected(tmp_path: Path) -> None:
    drafts = _three_drafts(tmp_path)
    call_count = 0
    rounds = []

    def provider(pending, _state):
        nonlocal call_count
        call_count += 1
        decisions = []
        for draft in pending:
            if draft.job_id == "job-a":
                decisions.append(
                    ResumeReviewDecision(
                        job_id="job-a",
                        decision="reject",
                        comments=f"Reject round {draft.revision_round}.",
                    )
                )
            else:
                decisions.append(ResumeReviewDecision(job_id=draft.job_id, decision="approve"))
        return decisions

    def handler(job_id, _previous, comments, _memory, next_round):
        rounds.append(next_round)
        return _draft(tmp_path / "revisions" / f"r{next_round}", job_id, next_round, comments)

    with pytest.raises(RevisionLimitError, match="third revision"):
        _run(tmp_path, drafts, provider, handler)
    assert rounds == [1, 2]
    assert call_count == 3


@pytest.mark.parametrize("mismatch", ["job", "round", "feedback"])
def test_revision_result_mismatch_is_rejected(
    tmp_path: Path,
    mismatch: str,
) -> None:
    drafts = _three_drafts(tmp_path)

    def provider(pending, _state):
        return [
            ResumeReviewDecision(
                job_id=draft.job_id,
                decision="reject" if draft.job_id == "job-a" else "approve",
                comments="Please revise." if draft.job_id == "job-a" else "",
            )
            for draft in pending
        ]

    def handler(job_id, _previous, comments, _memory, next_round):
        return _draft(
            tmp_path / "bad-revision",
            "wrong-job" if mismatch == "job" else job_id,
            2 if mismatch == "round" else next_round,
            "wrong feedback" if mismatch == "feedback" else comments,
        )

    with pytest.raises(RevisionResultMismatchError):
        _run(tmp_path, drafts, provider, handler)


def test_missing_decision_and_reject_without_comments_fail(tmp_path: Path) -> None:
    drafts = _three_drafts(tmp_path)
    with pytest.raises(InvalidReviewBatchError, match="every pending"):
        _run(
            tmp_path / "missing",
            drafts,
            lambda pending, _: [
                ResumeReviewDecision(job_id=draft.job_id, decision="approve")
                for draft in pending[:2]
            ],
            lambda *_: None,
        )
    with pytest.raises(ValidationError, match="nonempty comments"):
        ResumeReviewDecision(job_id="job-a", decision="reject")
