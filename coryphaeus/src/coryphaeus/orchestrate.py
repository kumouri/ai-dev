"""Execute a workflow.

Steps run in index order, which is a valid topological order because ``deps`` may only point
backwards (see ``schema``). Each step is shown *exactly* the earlier results it declared — that
selectivity is the mechanism being studied, so the orchestrator must not be generous with context.

A failing worker is data, not an exception: the step records the failure, downstream steps see a
short error note, and the rollout is scored on whatever the final step produced. Papering over that
would hide from the policy the cost of routing to an unreliable worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .policy.prompts import WORKER_SYSTEM, render_worker_prompt
from .schema import SELF, Step, Workflow
from .workers.base import Worker, WorkerResult
from .workers.governor import Governor
from .workers.registry import WorkerRegistry


@dataclass(slots=True)
class StepOutcome:
    """One executed step."""

    step: Step
    result: WorkerResult
    prompt_chars: int
    resolved_worker: str

    @property
    def ok(self) -> bool:
        return self.result.ok


@dataclass(slots=True)
class ExecutionResult:
    """What running a workflow produced, in full, including the parts that failed."""

    final_text: str
    outcomes: list[StepOutcome] = field(default_factory=list)
    ok: bool = True
    error: str | None = None

    @property
    def n_steps(self) -> int:
        return len(self.outcomes)

    @property
    def cost_usd(self) -> float:
        return sum(o.result.cost_usd for o in self.outcomes)

    @property
    def latency_s(self) -> float:
        """Summed worker latency. Steps are sequential, so this is also the critical path."""
        return sum(o.result.latency_s for o in self.outcomes)

    @property
    def tokens_in(self) -> int:
        return sum(o.result.tokens_in for o in self.outcomes)

    @property
    def tokens_out(self) -> int:
        return sum(o.result.tokens_out for o in self.outcomes)

    @property
    def workers_used(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for o in self.outcomes:
            seen.setdefault(o.resolved_worker, None)
        return tuple(seen)

    @property
    def failed_steps(self) -> tuple[int, ...]:
        return tuple(o.step.index for o in self.outcomes if not o.ok)


def _error_note(result: WorkerResult) -> str:
    return f"[worker {result.worker} failed: {result.error or 'unknown error'}]"


async def execute(
    workflow: Workflow,
    registry: WorkerRegistry,
    question: str,
    *,
    governor: Governor,
    self_worker: Worker | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> ExecutionResult:
    """Run ``workflow`` and return the final step's text plus full per-step accounting.

    Args:
        self_worker: resolves :data:`~coryphaeus.schema.SELF`. Usually the conductor's own worker.
            If a workflow self-assigns and this is ``None``, those steps fail with
            ``self_unavailable`` rather than being silently rerouted — a substituted worker would
            corrupt the very attribution the telemetry exists to capture.
        temperature: low by default. Workers are being measured, not sampled for diversity; that job
            belongs to the policy.
    """
    outcomes: list[StepOutcome] = []
    texts: dict[int, str] = {}

    for step in workflow.steps:
        is_final = step.index == workflow.final
        priors = [(dep, texts.get(dep, "")) for dep in step.deps]
        prompt = render_worker_prompt(question, step.subtask, priors, is_final=is_final)

        if step.worker == SELF:
            worker = self_worker
            resolved = self_worker.spec.name if self_worker is not None else SELF
        else:
            worker = registry.get(step.worker)
            resolved = step.worker

        if worker is None:
            result = WorkerResult(
                worker=SELF,
                text="",
                ok=False,
                error="self_unavailable: self-assigned step, but no conductor worker was provided",
            )
        else:
            result = await governor.invoke(
                worker,
                prompt,
                system=WORKER_SYSTEM,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        texts[step.index] = result.text if result.ok else _error_note(result)
        outcomes.append(
            StepOutcome(
                step=step,
                result=result,
                prompt_chars=len(prompt),
                resolved_worker=resolved,
            )
        )

    final_text = texts.get(workflow.final, "")
    final_outcome = next((o for o in outcomes if o.step.index == workflow.final), None)
    ok = bool(final_outcome and final_outcome.ok)
    return ExecutionResult(
        final_text=final_text,
        outcomes=outcomes,
        ok=ok,
        error=None if ok else "final step failed",
    )


async def run_solo(
    worker: Worker,
    question: str,
    *,
    governor: Governor,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> ExecutionResult:
    """Ask one worker directly — the control arm.

    Shaped as an :class:`ExecutionResult` with a single synthetic step so the report can compare
    arms on identical fields (cost, latency, tokens) without special-casing.
    """
    from .policy.prompts import SOLO_SYSTEM, SOLO_TEMPLATE

    prompt = SOLO_TEMPLATE.format(question=question)
    result = await governor.invoke(
        worker, prompt, system=SOLO_SYSTEM, max_tokens=max_tokens, temperature=temperature
    )
    step = Step(index=1, subtask="(solo: answer directly)", worker=worker.spec.name, deps=())
    outcome = StepOutcome(
        step=step, result=result, prompt_chars=len(prompt), resolved_worker=worker.spec.name
    )
    return ExecutionResult(
        final_text=result.text,
        outcomes=[outcome],
        ok=result.ok,
        error=None if result.ok else (result.error or "solo worker failed"),
    )
