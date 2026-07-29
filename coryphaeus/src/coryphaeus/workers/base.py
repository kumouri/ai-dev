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
    """Provider signalled over-capacity (HTTP 429). The governor backs this worker off."""


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Static facts about a worker.

    Args:
        name: registry key — the string the conductor writes in a workflow. Short and mnemonic.
        model: provider-side model identifier.
        provider: ``"fake"`` | ``"ollama"`` | ``"featherless"``. Also the governor's pool key.
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
    """Featherless's published concurrency pricing: under 16B costs 1 unit, larger costs 4.

    Unknown size is treated as large, because guessing cheap is the mistake that produces 429s.
    """
    if params_b is None:
        return 4
    return 1 if params_b < 16 else 4
