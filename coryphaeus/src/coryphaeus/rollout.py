"""Rollouts: k attempts at one question, shaped exactly as GRPO will want them.

The group — not the individual rollout — is the unit, because GRPO's advantage is relative *within*
a group for the same question. :func:`group_advantages` already emits that, so the phase-3 trainer
plugs into this harness instead of reshaping it.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass, field

from .datasets.loaders import Question
from .orchestrate import ExecutionResult, execute
from .reward import Score, score_answer, score_failure
from .schema import Workflow
from .workers.base import Worker
from .workers.governor import Governor
from .workers.registry import WorkerRegistry
from .workflow import DepDefault, parse


@dataclass(slots=True)
class RolloutRecord:
    """One attempt: what the policy emitted, what happened, and what it scored."""

    question_id: str
    rollout_index: int
    arm: str
    raw: str
    score: Score
    workflow: Workflow | None = None
    parse_reason: str | None = None
    parse_detail: str = ""
    repaired: bool = False
    execution: ExecutionResult | None = None
    advantage: float | None = field(default=None)

    @property
    def parsed(self) -> bool:
        return self.workflow is not None

    @property
    def cost_usd(self) -> float:
        return self.execution.cost_usd if self.execution else 0.0

    @property
    def latency_s(self) -> float:
        return self.execution.latency_s if self.execution else 0.0

    def to_row(self, *, run_id: str) -> dict:
        """Flatten for ``rollouts.jsonl``."""
        ex = self.execution
        return {
            "run_id": run_id,
            "arm": self.arm,
            "question_id": self.question_id,
            "rollout_index": self.rollout_index,
            "parsed": self.parsed,
            "parse_reason": self.parse_reason,
            "parse_detail": self.parse_detail,
            "repaired": self.repaired,
            "n_steps": self.workflow.n_steps if self.workflow else 0,
            "workers_used": list(ex.workers_used) if ex else [],
            "failed_steps": list(ex.failed_steps) if ex else [],
            "workflow": self.workflow.to_dict() if self.workflow else None,
            "score": self.score.value,
            "correct": self.score.correct,
            "score_reason": self.score.reason,
            "extracted": self.score.extracted,
            "gold": self.score.gold,
            "advantage": self.advantage,
            "cost_usd": self.cost_usd,
            "latency_s": round(self.latency_s, 4),
            "tokens_in": ex.tokens_in if ex else 0,
            "tokens_out": ex.tokens_out if ex else 0,
            "raw_chars": len(self.raw or ""),
        }


async def one_rollout(
    raw: str,
    question: Question,
    registry: WorkerRegistry,
    *,
    governor: Governor,
    rollout_index: int,
    arm: str = "conductor",
    self_worker: Worker | None = None,
    default_deps: DepDefault = "all",
    max_tokens: int = 1024,
) -> RolloutRecord:
    """Parse one emission, execute it if valid, and score the result."""
    result = parse(raw, known_workers=registry.names(), default_deps=default_deps)
    if not result.ok:
        return RolloutRecord(
            question_id=question.id,
            rollout_index=rollout_index,
            arm=arm,
            raw=raw,
            score=score_failure(result.reason or "parse_failed", question.gold),
            parse_reason=result.reason,
            parse_detail=result.detail,
            repaired=result.repaired,
        )

    workflow: Workflow = result.workflow  # type: ignore[assignment]
    execution = await execute(
        workflow,
        registry,
        question.text,
        governor=governor,
        self_worker=self_worker,
        max_tokens=max_tokens,
    )
    score = (
        score_answer(execution.final_text, question.gold)
        if execution.ok
        else score_failure("final_step_failed", question.gold)
    )
    return RolloutRecord(
        question_id=question.id,
        rollout_index=rollout_index,
        arm=arm,
        raw=raw,
        score=score,
        workflow=workflow,
        repaired=result.repaired,
        execution=execution,
    )


async def rollout_group(
    policy: object,
    registry: WorkerRegistry,
    question: Question,
    *,
    governor: Governor,
    k: int = 4,
    arm: str = "conductor",
    self_worker: Worker | None = None,
    default_deps: DepDefault = "all",
    max_tokens: int = 1024,
) -> list[RolloutRecord]:
    """Sample ``k`` workflows for one question, execute them all, and attach GRPO advantages.

    Rollouts run concurrently on purpose: the concurrency governor is what keeps that safe, and
    serialising here would hide the throttling behaviour phase 3 has to live with.
    """
    raws = await policy.propose(question.text, registry.catalog_text(), k=k)  # type: ignore[attr-defined]
    records = await asyncio.gather(
        *(
            one_rollout(
                raw,
                question,
                registry,
                governor=governor,
                rollout_index=i,
                arm=arm,
                self_worker=self_worker,
                default_deps=default_deps,
                max_tokens=max_tokens,
            )
            for i, raw in enumerate(raws)
        )
    )
    group = list(records)
    for record, advantage in zip(group, group_advantages(group), strict=True):
        record.advantage = advantage
    return group


def group_advantages(records: list[RolloutRecord]) -> list[float]:
    """Group-relative advantages: ``(r_i - mean(r)) / std(r)``.

    A degenerate group (every rollout scored the same) yields all zeros, which is correct: there is
    no preference to learn from k identical outcomes. Watching how often that happens is how you
    find out ``k`` is too small or the slice too easy.
    """
    rewards = [r.score.value for r in records]
    if len(rewards) < 2:
        return [0.0 for _ in rewards]
    mean = statistics.fmean(rewards)
    stdev = statistics.pstdev(rewards)
    if stdev == 0.0:
        return [0.0 for _ in rewards]
    return [(r - mean) / stdev for r in rewards]
