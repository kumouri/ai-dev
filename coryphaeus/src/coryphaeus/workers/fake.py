"""A deterministic, offline worker pool — a first-class module, not a test fixture.

Ships in ``src/`` on purpose (see ``docs/adr/0001``). It makes the whole loop — parse, execute,
score, group, advantages — runnable with no network, no key, no GPU, and it is the only way to test
governor and failure-path behaviour deterministically. It also lets "does routing beat the best
single worker?" be checked against a pool whose right answer is *known*, before that metric is
trusted on real workers.

Competence is configured, not simulated: each worker answers correctly on the fraction of questions
its ``skill`` implies, chosen by a stable hash so the same worker always fails the same questions.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
from collections.abc import Callable, Iterable, Mapping

from .base import WorkerBusy, WorkerResult, WorkerSpec

#: Marker a fake worker emits so the reward's answer extraction has something realistic to find.
ANSWER_TEMPLATE = "Reasoning omitted (fake worker).\nThe answer is \\boxed{{{answer}}}"


def _unit_hash(*parts: str) -> float:
    """A stable float in [0, 1) from the given strings — same inputs, same value, every run."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


class FakeWorker:
    """A worker whose competence, latency and failure rate are configured.

    Args:
        spec: as for any worker; ``provider`` is forced to ``"fake"``.
        answers: question-key → correct answer. The key is matched as a substring of the prompt, so
            a subtask that quotes the question still resolves. Absent key → the worker says so.
        skill: probability of returning the correct answer when it knows one.
        wrong_answer: what it returns when its dice come up short.
        latency_s: simulated wall-clock, awaited so concurrency tests are meaningful.
        busy_every: raise :class:`WorkerBusy` on every Nth call (0 disables) — for governor tests.
        seed: shifts the competence hash without changing its determinism.
    """

    def __init__(
        self,
        spec: WorkerSpec,
        *,
        answers: Mapping[str, str] | None = None,
        skill: float = 1.0,
        wrong_answer: str | Callable[[str], str] = "0",
        latency_s: float = 0.0,
        busy_every: int = 0,
        seed: int = 0,
    ) -> None:
        # A fake worker must never be mistaken for a real provider in telemetry or the governor.
        self.spec = spec if spec.provider == "fake" else dataclasses.replace(spec, provider="fake")
        self.answers = dict(answers or {})
        self.skill = skill
        self.wrong_answer = wrong_answer
        self.latency_s = latency_s
        self.busy_every = busy_every
        self.seed = seed
        self.calls = 0
        self.prompts: list[str] = []
        self.max_concurrent = 0
        self._in_flight = 0

    def _lookup(self, prompt: str) -> tuple[str | None, str | None]:
        """Return (question key, gold answer) for the first known question in ``prompt``."""
        for key, answer in self.answers.items():
            if key and key in prompt:
                return key, answer
        return None, None

    def knows(self, prompt: str) -> bool:
        return self._lookup(prompt)[0] is not None

    def would_be_correct(self, question_key: str) -> bool:
        """Whether this worker gets that question right — deterministic, inspectable by tests."""
        return _unit_hash(self.spec.name, str(self.seed), question_key) < self.skill

    async def invoke(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> WorkerResult:
        self.calls += 1
        self.prompts.append(prompt)
        if self.busy_every and self.calls % self.busy_every == 0:
            raise WorkerBusy(f"{self.spec.name} simulated 429 on call {self.calls}")

        self._in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self._in_flight)
        try:
            if self.latency_s:
                await asyncio.sleep(self.latency_s)
        finally:
            self._in_flight -= 1

        key, gold = self._lookup(prompt)
        if key is None:
            text = "I do not have that question."
        elif self.would_be_correct(key):
            text = ANSWER_TEMPLATE.format(answer=gold)
        else:
            wrong = self.wrong_answer(key) if callable(self.wrong_answer) else self.wrong_answer
            text = ANSWER_TEMPLATE.format(answer=wrong)

        tokens_in = max(1, len(prompt) // 4)
        tokens_out = max(1, len(text) // 4)
        return WorkerResult(
            worker=self.spec.name,
            text=text,
            ok=True,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=self.latency_s,
            cost_usd=self.spec.cost(tokens_in, tokens_out),
        )


def fake_spec(
    name: str,
    *,
    params_b: float = 4.0,
    units: int = 1,
    tags: Iterable[str] = (),
) -> WorkerSpec:
    return WorkerSpec(
        name=name,
        model=f"fake/{name}",
        provider="fake",
        params_b=params_b,
        units=units,
        tags=tuple(tags),
    )


def make_fake_pool(
    answers: Mapping[str, str],
    *,
    seed: int = 0,
) -> list[FakeWorker]:
    """A deliberately heterogeneous three-worker pool.

    Competence is uncorrelated across the three (different hash inputs), so a perfect router beats
    every one of them and a random router beats none — exactly the property needed to sanity check
    the baseline metric before pointing it at real models.
    """
    return [
        FakeWorker(
            fake_spec("tiny", params_b=2.0, tags=("fast", "cheap")),
            answers=answers,
            skill=0.35,
            latency_s=0.0,
            seed=seed,
        ),
        FakeWorker(
            fake_spec("mid", params_b=9.0, tags=("balanced",)),
            answers=answers,
            skill=0.55,
            latency_s=0.0,
            seed=seed + 1,
        ),
        FakeWorker(
            fake_spec("big", params_b=27.0, units=4, tags=("slow", "strong", "math")),
            answers=answers,
            skill=0.7,
            latency_s=0.0,
            seed=seed + 2,
        ),
    ]
