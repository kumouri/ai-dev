"""The worker contract.

A worker is anything that turns a prompt into text and reports what that cost. Providers differ in
billing and throttling, so the *spec* carries the facts the registry and governor need; the
*result* carries everything telemetry wants, including for local and flat-rate workers where the
marginal dollar cost is zero. A number you did not record is a number you cannot report, and the
result being replicated is a cost result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class WorkerBusy(Exception):
    """The provider said "not now" — the governor backs this worker off and retries.

    Raised for **transient** conditions only: HTTP 429 (rate/concurrency limit) and 502/503/504
    (gateway and capacity errors, e.g. Featherless's ``capacity_exhausted``).

    Why the distinction matters more than it looks: under GRPO, a rollout that fails is scored zero,
    and zero is how the policy learns "routing there was a bad choice". If a provider hiccup were
    reported as an ordinary failure, infrastructure noise would be laundered into a routing lesson
    and the policy would learn to avoid a perfectly good worker. Permanent failures (403 on a gated
    model, 404 on a bad id) are *not* this — those are real, and retrying them just wastes time.
    """


#: Statuses the governor should back off and retry rather than treat as an outcome.
TRANSIENT_STATUSES = frozenset({429, 502, 503, 504})

#: Provider error codes that mean "try again" **despite arriving with a 4xx status**. Featherless
#: returns ``400 completion_error`` for a transient generation failure (observed live 2026-07-29: a
#: model that answered correctly moments earlier). Status alone is not enough to classify these, and
#: getting it wrong scores a working worker as a bad routing choice.
TRANSIENT_ERROR_CODES = frozenset({"completion_error", "capacity_exhausted", "server_error"})


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Static facts about a worker.

    Args:
        name: registry key — the string the conductor writes in a workflow. Short and mnemonic.
        model: provider-side model identifier.
        provider: ``"fake"`` | ``"ollama"`` | ``"featherless"`` | ``"openrouter"``. Also the
            governor's pool key.
        params_b: parameter count in billions, shown in the catalog and used to derive units.
        units: concurrency units this worker consumes from its provider's budget.
        tags: capability hints shown to the conductor (e.g. ``("math", "fast")``).
        usd_per_mtok_in / usd_per_mtok_out: for cost accounting. Zero for local and flat-rate
            workers — cost is then reported as tokens and latency, which is the honest currency.
    """

    name: str
    model: str
    provider: str
    params_b: float | None = None
    units: int = 1
    tags: tuple[str, ...] = ()
    usd_per_mtok_in: float = 0.0
    usd_per_mtok_out: float = 0.0

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.usd_per_mtok_in + tokens_out * self.usd_per_mtok_out) / 1_000_000.0

    def catalog_line(self) -> str:
        """One line of the menu the conductor reads. Terse on purpose — it is prompt budget."""
        size = f"{self.params_b:g}B" if self.params_b else "unknown size"
        tags = f"; {', '.join(self.tags)}" if self.tags else ""
        return f"- {self.name} ({size}{tags})"


@dataclass(frozen=True, slots=True)
class WorkerResult:
    """What a single worker invocation produced and cost."""

    worker: str
    text: str
    ok: bool = True
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    attempts: int = 1
    error: str | None = None
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Worker(Protocol):
    """Anything that can answer a prompt."""

    spec: WorkerSpec

    async def invoke(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> WorkerResult:
        """Answer ``prompt``.

        Implementations should return a ``WorkerResult`` with ``ok=False`` and ``error`` set for
        ordinary failures rather than raising — a failing worker is data, not an exception. The one
        exception is :class:`WorkerBusy`, which the governor must see in order to back off.
        """
        ...


class timed:
    """Tiny context manager for wall-clock timing; ``with timed() as t: ...`` then ``t.elapsed``."""

    __slots__ = ("_start", "elapsed")

    def __enter__(self) -> timed:
        self._start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start


def units_for_params(params_b: float | None) -> int:
    """**Fallback only.** Estimate concurrency units from parameter count.

    Prefer the provider's own number: Featherless reports ``concurrency_cost`` per model on
    ``/v1/models``, and a catalogue snapshot (``manifests/featherless_pool.json``, refreshed by
    ``scripts/featherless_catalog.py``) carries it. Use this function only for a model absent from
    the snapshot.

    Why it is not good enough on its own: the real distribution has **four** tiers, not two, and the
    24–32B band costs **2** — so this heuristic over-reserves on exactly the mid-size workers a
    router most wants, halving effective concurrency for nothing. ``Qwen2.5-32B-Instruct`` is the
    pinned counter-example in ``tests/test_pool_sampling.py``.

    Unknown size still estimates high: guessing cheap is the mistake that produces 429s.
    """
    if params_b is None:
        return 4
    if params_b < 16:
        return 1
    if params_b < 40:
        return 2
    return 4
