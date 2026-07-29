"""Shared fixtures. Everything here is offline — no network, no key, no GPU."""

from __future__ import annotations

import json

import pytest

from coryphaeus.datasets.loaders import load_fixture
from coryphaeus.workers import Governor, WorkerRegistry, make_fake_pool


@pytest.fixture
def questions():
    return load_fixture()


@pytest.fixture
def answers(questions):
    """Question text → gold answer, the mapping a FakeWorker matches against a prompt."""
    return {q.text: q.gold for q in questions}


@pytest.fixture
def pool(answers):
    return make_fake_pool(answers)


@pytest.fixture
def registry(pool):
    return WorkerRegistry(pool)


@pytest.fixture
def governor(registry):
    """A governor with generous budgets and no real sleeping."""

    async def no_sleep(_seconds: float) -> None:
        return None

    return Governor(budgets={"fake": 16}, sleeper=no_sleep, base_delay=0.0, jitter=0.0)


def _emit(steps: list[dict], final: int | None = None, *, fenced: bool = True) -> str:
    """Build a conductor emission the way a model would — prose around a fenced JSON block."""
    payload: dict = {"steps": steps}
    if final is not None:
        payload["final"] = final
    body = json.dumps(payload)
    return f"Here is my plan.\n```json\n{body}\n```" if fenced else body


@pytest.fixture
def emit():
    return _emit
