"""Concurrency governance — the reason a flat-rate worker pool is usable for GRPO at all.

Providers throttle differently and the difference is load-bearing:

* **Featherless** bills concurrency in *units*: a sub-16B model costs 1, a 70B+ model costs 4, and a
  Premium account holds 4 units total before HTTP 429. One 70B call saturates the account.
* **Ollama** is local — the limit is VRAM, not a quota. A model larger than VRAM spills to system
  RAM and crawls, so it is given a unit cost that keeps it effectively serial.

Two implementation notes that are easy to get wrong:

1. ``asyncio.Semaphore`` cannot acquire N permits atomically. Looping single acquires lets two
   concurrent 4-unit requests each hold 2 and deadlock, so this uses an ``asyncio.Condition``.
2. A 429 must back off **that worker**, never the batch. One oversubscribed large worker stalling a
   slice of small calls that would have fitted is the failure mode this exists to prevent.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import random
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field

from .base import Worker, WorkerBusy, WorkerResult


class UnitPool:
    """An N-unit counting semaphore with atomic multi-unit acquisition."""

    def __init__(self, budget: int, name: str = "pool") -> None:
        if budget < 1:
            raise ValueError(f"{name}: budget must be >= 1, got {budget}")
        self.name = name
        self.budget = budget
        self._available = budget
        self._condition = asyncio.Condition()

    @property
    def available(self) -> int:
        return self._available

    def clamp(self, units: int) -> int:
        """Never ask for more than the budget — that would wait forever, not throttle."""
        return max(1, min(units, self.budget))

    async def acquire(self, units: int) -> int:
        want = self.clamp(units)
        async with self._condition:
            await self._condition.wait_for(lambda: self._available >= want)
            self._available -= want
        return want

    async def release(self, units: int) -> None:
        async with self._condition:
            self._available = min(self.budget, self._available + units)
            self._condition.notify_all()

    @contextlib.asynccontextmanager
    async def slot(self, units: int) -> AsyncIterator[int]:
        held = await self.acquire(units)
        try:
            yield held
        finally:
            await self.release(held)


@dataclass
class _BackoffState:
    strikes: int = 0
    ready_at: float = 0.0


@dataclass
class Governor:
    """Owns concurrency budgets and the retry policy for every worker call.

    Args:
        budgets: provider name → unit budget.
        max_retries: attempts after the first, on :class:`WorkerBusy` only.
        base_delay / max_delay: exponential backoff bounds, in seconds.
        jitter: fraction of the delay added at random, to de-synchronise a burst of retries.
        rng: injectable for deterministic tests.
        sleeper: injectable ``async def (seconds)`` for tests that must not actually wait.
    """

    budgets: dict[str, int]
    max_retries: int = 3
    base_delay: float = 0.5
    max_delay: float = 30.0
    jitter: float = 0.25
    rng: random.Random = field(default_factory=random.Random)
    sleeper: object | None = None
    _pools: dict[str, UnitPool] = field(default_factory=dict, init=False, repr=False)
    _backoff: dict[str, _BackoffState] = field(default_factory=dict, init=False, repr=False)
    busy_events: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        for provider, budget in self.budgets.items():
            self._pools[provider] = UnitPool(budget, name=provider)

    def pool(self, provider: str) -> UnitPool:
        """Pool for ``provider``, created with a budget of 1 if unregistered.

        Defaulting to 1 rather than unlimited is deliberate: an unknown provider is throttled until
        somebody states its budget.
        """
        if provider not in self._pools:
            self._pools[provider] = UnitPool(1, name=provider)
        return self._pools[provider]

    async def _sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if self.sleeper is not None:
            await self.sleeper(seconds)  # type: ignore[operator]
        else:
            await asyncio.sleep(seconds)

    def delay_for(self, worker_name: str) -> float:
        """Current backoff delay for a worker, from its strike count."""
        state = self._backoff.get(worker_name)
        if state is None or state.strikes == 0:
            return 0.0
        raw = min(self.max_delay, self.base_delay * (2 ** (state.strikes - 1)))
        return raw * (1.0 + self.jitter * self.rng.random())

    def note_busy(self, worker_name: str) -> float:
        """Record a 429 and return the delay to wait before retrying that worker."""
        state = self._backoff.setdefault(worker_name, _BackoffState())
        state.strikes += 1
        self.busy_events += 1
        return self.delay_for(worker_name)

    def note_ok(self, worker_name: str) -> None:
        """A success clears the worker's strikes — throttling is a moment, not a verdict."""
        self._backoff.pop(worker_name, None)

    def strikes(self, worker_name: str) -> int:
        state = self._backoff.get(worker_name)
        return state.strikes if state else 0

    async def invoke(
        self,
        worker: Worker,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> WorkerResult:
        """Invoke ``worker`` under its provider's unit budget, retrying only on 429."""
        spec = worker.spec
        pool = self.pool(spec.provider)
        attempts = 0
        delay = 0.0
        last_error = "unknown"

        while attempts <= self.max_retries:
            attempts += 1
            async with pool.slot(spec.units):
                try:
                    result = await worker.invoke(
                        prompt, system=system, max_tokens=max_tokens, temperature=temperature
                    )
                except WorkerBusy as exc:
                    last_error = f"busy: {exc}" if str(exc) else "busy"
                    delay = self.note_busy(spec.name)
                else:
                    if result.ok:
                        self.note_ok(spec.name)
                    return dataclasses.replace(result, attempts=attempts)
            # Sleep outside the slot so a backing-off worker does not hold units it cannot use.
            await self._sleep(delay)

        return WorkerResult(
            worker=spec.name,
            text="",
            ok=False,
            attempts=attempts,
            error=f"exhausted retries after {attempts} attempts ({last_error})",
        )


def governor_for(specs: Iterable[object], **kwargs: object) -> Governor:
    """Build a governor sized for the providers actually present.

    Budgets come from :mod:`coryphaeus.config` so they are environment-driven, not hardcoded.
    """
    from ..config import settings

    cfg = settings()
    providers = {getattr(spec, "provider", "unknown") for spec in specs}
    budgets: dict[str, int] = {}
    for provider in providers:
        if provider == "featherless":
            budgets[provider] = cfg.featherless_unit_budget
        elif provider == "ollama":
            budgets[provider] = cfg.ollama_unit_budget
        elif provider == "openrouter":
            budgets[provider] = cfg.openrouter_unit_budget
        else:
            # Fake and any future in-process provider: parallelism is free.
            budgets[provider] = 16
    return Governor(budgets=budgets, **kwargs)  # type: ignore[arg-type]
