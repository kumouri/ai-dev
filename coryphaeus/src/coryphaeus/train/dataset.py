"""Build the training rows a GRPO trainer consumes.

One row per question, carrying the rendered conductor prompt plus everything the reward function
needs to score what comes back: the question, the gold answer, and **which sub-pool that prompt
advertised**.

That last column is the point. The prompt embeds a worker catalogue, so a randomised pool *is* a
different prompt — and scoring has to honour the same pool the conductor was shown, or routing to a
worker it was never offered would be silently accepted. Pool composition therefore travels with the
row rather than being global state.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..datasets.loaders import Question
from ..policy.prompts import PROMPT_VERSION, render_conductor_prompt
from ..pools import sample_pool
from ..schema import MAX_STEPS
from ..workers.registry import WorkerRegistry


@dataclass(frozen=True, slots=True)
class TrainRow:
    """One training example. ``to_dict`` is the shape a HF ``Dataset`` is built from."""

    prompt: str
    question: str
    question_id: str
    gold: str
    pool: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "question": self.question,
            "question_id": self.question_id,
            "gold": self.gold,
            "pool": list(self.pool),
        }


def build_row(question: Question, registry: WorkerRegistry, pool: Sequence[str]) -> TrainRow:
    """Render one row against a specific sub-pool."""
    view = registry.subset(pool)
    catalog = view.catalog_text()
    example_worker = view.names()[0]
    return TrainRow(
        prompt=render_conductor_prompt(
            question.text,
            catalog,
            max_steps=MAX_STEPS,
            example_worker=example_worker,
        ),
        question=question.text,
        question_id=question.id,
        gold=question.gold,
        pool=tuple(view.names()),
    )


def build_rows(
    questions: Iterable[Question],
    registry: WorkerRegistry,
    *,
    pools: Sequence[Sequence[str]] | None = None,
    seed: int = 0,
    pool_size: int | tuple[int, int] = (2, 4),
) -> list[TrainRow]:
    """Render training rows, one per question.

    Args:
        pools: fixed compositions to cycle through — pass a :class:`~coryphaeus.pools.PoolSplit`
            side to keep train and eval pools apart. When ``None``, a fresh pool is sampled per
            question from ``registry``.
        seed: makes both the sampling and the cycling reproducible. A run's pool assignment can then
            be reconstructed from the seed alone, which matters when a result needs explaining.

    Cycling rather than sampling from ``pools`` is deliberate: it guarantees every composition is
    seen a similar number of times, where sampling would leave coverage to luck on a small set.
    """
    rng = random.Random(seed)
    rows: list[TrainRow] = []
    for index, question in enumerate(questions):
        if pools:
            pool = pools[index % len(pools)]
        else:
            pool = sample_pool(registry, rng, size=pool_size)
        rows.append(build_row(question, registry, pool))
    return rows


def to_hf_dataset(rows: Sequence[TrainRow]):
    """Wrap rows in a ``datasets.Dataset``. Requires the ``data`` extra."""
    try:
        from datasets import Dataset  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("needs the 'data' extra: uv sync --extra data") from exc
    return Dataset.from_list([row.to_dict() for row in rows])


def describe(rows: Sequence[TrainRow]) -> dict:
    """Run metadata worth recording: what the policy was actually shown."""
    pool_counts: dict[tuple[str, ...], int] = {}
    for row in rows:
        pool_counts[row.pool] = pool_counts.get(row.pool, 0) + 1
    prompt_chars = [len(row.prompt) for row in rows]
    return {
        "rows": len(rows),
        "prompt_version": PROMPT_VERSION,
        "distinct_pools": len(pool_counts),
        "pool_counts": {"+".join(pool): count for pool, count in sorted(pool_counts.items())},
        "prompt_chars_min": min(prompt_chars) if prompt_chars else 0,
        "prompt_chars_max": max(prompt_chars) if prompt_chars else 0,
    }
