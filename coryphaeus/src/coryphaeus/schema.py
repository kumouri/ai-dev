"""The workflow schema — the only structure the orchestrator will execute.

A conductor emits text; :mod:`coryphaeus.workflow` turns that text into a :class:`Workflow` or a
named failure. This module owns the *shape* and the *rules*, so "is this workflow legal?" has
exactly one answer in exactly one place.

Design note: a step's ``deps`` may reference **strictly earlier** steps only. Cycles are therefore
impossible by construction and index order is always a valid execution order — a validation rule
doing the work a topological sort would otherwise do. See ``docs/adr/0001``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

#: Hard cap on workflow length, following the paper's ≤5 steps.
MAX_STEPS = 5

#: Reserved worker name meaning "the conductor handles this step itself" — the paper's recursion
#: mechanism. The orchestrator resolves it to whichever worker is acting as the conductor.
SELF = "self"


class WorkflowError(ValueError):
    """An un-executable workflow.

    ``reason`` is a stable code, not prose: it is logged per rollout and read as a training metric
    (rising ``unknown_worker`` means the catalog text is failing the policy, rising
    ``too_many_steps`` means the cap is not reaching it, and so on).
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


#: Every failure code this module can produce. Kept explicit so reporting can enumerate them
#: without having seen one occur.
REASONS: tuple[str, ...] = (
    "no_steps",
    "too_many_steps",
    "bad_index",
    "empty_subtask",
    "unknown_worker",
    "bad_dep",
    "forward_dep",
    "self_dep",
    "bad_final",
)


@dataclass(frozen=True, slots=True)
class Step:
    """One routed subtask.

    Args:
        index: 1-based position, and the identifier ``deps`` refer to.
        subtask: the natural-language instruction handed to the worker.
        worker: registry name, or :data:`SELF`.
        deps: indices of earlier steps whose results this step is shown.
    """

    index: int
    subtask: str
    worker: str
    deps: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Workflow:
    """A validated, executable plan."""

    steps: tuple[Step, ...]
    final: int
    raw: str = ""

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def step(self, index: int) -> Step:
        """Return the step with the given 1-based index."""
        return self.steps[index - 1]

    @property
    def final_step(self) -> Step:
        return self.step(self.final)

    def workers_used(self) -> tuple[str, ...]:
        """Distinct worker names, in first-use order."""
        seen: dict[str, None] = {}
        for s in self.steps:
            seen.setdefault(s.worker, None)
        return tuple(seen)

    def to_dict(self) -> dict:
        return {
            "steps": [
                {"index": s.index, "subtask": s.subtask, "worker": s.worker, "deps": list(s.deps)}
                for s in self.steps
            ],
            "final": self.final,
        }


def validate(steps: Sequence[Step], final: int, *, known_workers: Iterable[str]) -> None:
    """Raise :class:`WorkflowError` unless these steps form an executable workflow.

    Args:
        steps: candidate steps, expected to be indexed 1..n in order.
        final: index of the step whose output is the answer.
        known_workers: names the registry can resolve. :data:`SELF` is always allowed.
    """
    if not steps:
        raise WorkflowError("no_steps")
    if len(steps) > MAX_STEPS:
        raise WorkflowError("too_many_steps", f"{len(steps)} > {MAX_STEPS}")

    allowed = set(known_workers) | {SELF}

    for position, step in enumerate(steps, start=1):
        if step.index != position:
            raise WorkflowError(
                "bad_index", f"step at position {position} claims index {step.index}"
            )
        if not step.subtask or not step.subtask.strip():
            raise WorkflowError("empty_subtask", f"step {step.index}")
        if step.worker not in allowed:
            raise WorkflowError("unknown_worker", f"step {step.index}: {step.worker!r}")
        for dep in step.deps:
            if not isinstance(dep, int) or dep < 1 or dep > len(steps):
                raise WorkflowError("bad_dep", f"step {step.index} -> {dep!r}")
            if dep == step.index:
                raise WorkflowError("self_dep", f"step {step.index}")
            if dep > step.index:
                raise WorkflowError("forward_dep", f"step {step.index} -> {dep}")

    if not isinstance(final, int) or final < 1 or final > len(steps):
        raise WorkflowError("bad_final", repr(final))


def build(
    steps: Sequence[Step],
    final: int | None = None,
    *,
    known_workers: Iterable[str],
    raw: str = "",
) -> Workflow:
    """Validate and construct a :class:`Workflow`.

    ``final`` defaults to the last step, which is what a conductor almost always means.
    """
    resolved_final = len(steps) if final is None else final
    validate(steps, resolved_final, known_workers=known_workers)
    return Workflow(steps=tuple(steps), final=resolved_final, raw=raw)
