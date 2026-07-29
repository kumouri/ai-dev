"""Concurrency budgets and 429 backoff — behaviour a live provider cannot test deterministically."""

from __future__ import annotations

import asyncio

import pytest

from coryphaeus.workers import Governor, UnitPool, WorkerBusy, WorkerResult, fake_spec
from coryphaeus.workers.fake import FakeWorker


async def _no_sleep(_seconds: float) -> None:
    return None


def _governor(budget: int = 4, **kwargs) -> Governor:
    return Governor(
        budgets={"fake": budget}, sleeper=_no_sleep, base_delay=0.0, jitter=0.0, **kwargs
    )


async def test_unit_pool_grants_and_returns_units():
    pool = UnitPool(4, "test")
    async with pool.slot(3):
        assert pool.available == 1
    assert pool.available == 4


async def test_unit_pool_acquires_multiple_units_atomically():
    """Two concurrent 4-unit requests on a 4-unit budget must serialise, not deadlock at 2 each."""
    pool = UnitPool(4, "test")
    order: list[str] = []

    async def big(tag: str) -> None:
        async with pool.slot(4):
            order.append(f"{tag}-in")
            await asyncio.sleep(0.01)
            order.append(f"{tag}-out")

    await asyncio.wait_for(asyncio.gather(big("a"), big("b")), timeout=2.0)
    # Strict alternation proves neither call held a partial allocation.
    assert order in (["a-in", "a-out", "b-in", "b-out"], ["b-in", "b-out", "a-in", "a-out"])


async def test_oversized_request_is_clamped_rather_than_hanging():
    """A 4-unit worker against a 2-unit budget must run, not wait forever."""
    pool = UnitPool(2, "small")
    async with pool.slot(4):
        assert pool.available == 0


async def test_budget_limits_real_concurrency():
    worker = FakeWorker(fake_spec("w", units=1), answers={"q": "1"}, latency_s=0.02)
    gov = _governor(budget=2)
    await asyncio.gather(*(gov.invoke(worker, "q") for _ in range(6)))
    assert worker.calls == 6
    assert worker.max_concurrent <= 2


async def test_large_worker_saturates_its_provider():
    """A 4-unit worker on a 4-unit budget is effectively serial — the Featherless 70B case."""
    worker = FakeWorker(fake_spec("big", params_b=70, units=4), answers={"q": "1"}, latency_s=0.02)
    gov = _governor(budget=4)
    await asyncio.gather(*(gov.invoke(worker, "q") for _ in range(4)))
    assert worker.max_concurrent == 1


async def test_busy_is_retried_and_then_succeeds():
    worker = FakeWorker(fake_spec("w"), answers={"q": "1"}, busy_every=2)
    gov = _governor()
    result = await gov.invoke(worker, "q")  # call 1 ok
    assert result.ok and result.attempts == 1
    result = await gov.invoke(worker, "q")  # call 2 busy, call 3 ok
    assert result.ok
    assert result.attempts == 2
    assert gov.busy_events == 1


async def test_retries_are_bounded_and_report_exhaustion():
    worker = FakeWorker(fake_spec("w"), answers={"q": "1"}, busy_every=1)  # always busy
    gov = _governor(max_retries=2)
    result = await gov.invoke(worker, "q")
    assert not result.ok
    assert result.attempts == 3  # first try + 2 retries
    assert "exhausted retries" in (result.error or "")


async def test_backoff_is_per_worker_not_global():
    """One throttled worker must not delay another — the whole point of per-worker backoff."""
    busy = FakeWorker(fake_spec("busy"), answers={"q": "1"}, busy_every=1)
    calm = FakeWorker(fake_spec("calm"), answers={"q": "1"})
    gov = _governor(max_retries=1)
    await gov.invoke(busy, "q")
    assert gov.strikes("busy") == 2
    assert gov.strikes("calm") == 0
    result = await gov.invoke(calm, "q")
    assert result.ok and result.attempts == 1


async def test_backoff_grows_then_clears_on_success():
    gov = Governor(budgets={"fake": 4}, sleeper=_no_sleep, base_delay=1.0, jitter=0.0)
    assert gov.note_busy("w") == 1.0
    assert gov.note_busy("w") == 2.0
    assert gov.note_busy("w") == 4.0
    gov.note_ok("w")
    assert gov.strikes("w") == 0
    assert gov.delay_for("w") == 0.0


async def test_backoff_is_capped():
    gov = Governor(
        budgets={"fake": 4}, sleeper=_no_sleep, base_delay=1.0, max_delay=5.0, jitter=0.0
    )
    for _ in range(10):
        gov.note_busy("w")
    assert gov.delay_for("w") == 5.0


async def test_unregistered_provider_defaults_to_serial():
    """An unknown provider is throttled to 1 until somebody states its budget."""
    gov = _governor()
    assert gov.pool("mystery").budget == 1


async def test_ordinary_failures_are_returned_not_retried():
    class Broken:
        spec = fake_spec("broken")

        async def invoke(self, prompt, *, system=None, max_tokens=1024, temperature=0.7):
            return WorkerResult(worker="broken", text="", ok=False, error="http 500")

    gov = _governor()
    result = await gov.invoke(Broken(), "q")
    assert not result.ok
    assert result.attempts == 1  # a 500 is not a 429; retrying it is not the governor's business
    assert result.error == "http 500"


async def test_busy_exception_reaches_the_governor_from_a_worker():
    worker = FakeWorker(fake_spec("w"), answers={}, busy_every=1)
    with pytest.raises(WorkerBusy):
        await worker.invoke("q")
