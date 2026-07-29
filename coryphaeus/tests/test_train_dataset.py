"""Training rows — offline.

The property that matters: the prompt a row carries and the pool it records must agree. If they can
drift, the conductor is shown one catalogue and scored against another, and the resulting
`unknown_worker` failures look like a policy problem rather than a harness bug.
"""

from __future__ import annotations

import pytest

from coryphaeus.policy.prompts import PROMPT_VERSION
from coryphaeus.train.dataset import build_row, build_rows, describe
from coryphaeus.workers import WorkerRegistry, fake_spec
from coryphaeus.workers.fake import FakeWorker


@pytest.fixture
def registry6():
    return WorkerRegistry(
        FakeWorker(fake_spec(name), answers={}, seed=i)
        for i, name in enumerate(("a", "b", "c", "d", "e", "f"))
    )


def test_row_prompt_advertises_only_the_sub_pool(registry6, questions):
    row = build_row(questions[0], registry6, ("b", "d"))
    assert "- b (" in row.prompt
    assert "- d (" in row.prompt
    for absent in ("- a (", "- c (", "- e (", "- f ("):
        assert absent not in row.prompt


def test_row_records_the_pool_it_advertised(registry6, questions):
    row = build_row(questions[0], registry6, ("b", "d"))
    assert row.pool == ("b", "d")


def test_row_carries_everything_the_reward_needs(registry6, questions):
    q = questions[0]
    row = build_row(q, registry6, ("a", "b"))
    data = row.to_dict()
    assert set(data) == {"prompt", "question", "question_id", "gold", "pool"}
    assert data["gold"] == q.gold
    assert data["question_id"] == q.id
    assert data["question"] == q.text
    assert data["pool"] == ["a", "b"]


def test_prompt_contains_the_question_and_the_step_cap(registry6, questions):
    row = build_row(questions[0], registry6, ("a", "b"))
    assert questions[0].text in row.prompt
    assert "At most 5 steps" in row.prompt


def test_prompt_example_names_a_worker_from_this_pool(registry6, questions):
    """A worked example naming an unavailable worker would teach exactly the wrong thing."""
    row = build_row(questions[0], registry6, ("d", "e"))
    example_lines = [ln for ln in row.prompt.splitlines() if '"worker"' in ln]
    assert example_lines
    assert all(('"d"' in ln or '"e"' in ln) for ln in example_lines)


def test_fixed_pools_are_cycled_so_coverage_is_even(registry6, questions):
    pools = [("a", "b"), ("c", "d"), ("e", "f")]
    rows = build_rows(questions[:6], registry6, pools=pools)
    assert [r.pool for r in rows] == [
        ("a", "b"),
        ("c", "d"),
        ("e", "f"),
        ("a", "b"),
        ("c", "d"),
        ("e", "f"),
    ]


def test_sampling_is_reproducible_from_the_seed(registry6, questions):
    a = build_rows(questions[:5], registry6, seed=42)
    b = build_rows(questions[:5], registry6, seed=42)
    assert [r.pool for r in a] == [r.pool for r in b]


def test_different_seeds_give_different_pool_assignments(registry6, questions):
    a = build_rows(questions[:8], registry6, seed=1)
    b = build_rows(questions[:8], registry6, seed=2)
    assert [r.pool for r in a] != [r.pool for r in b]


def test_sampled_pools_respect_the_size_range(registry6, questions):
    rows = build_rows(questions, registry6, seed=7, pool_size=(2, 3))
    assert all(2 <= len(r.pool) <= 3 for r in rows)


def test_one_row_per_question(registry6, questions):
    rows = build_rows(questions, registry6, seed=0)
    assert len(rows) == len(questions)
    assert [r.question_id for r in rows] == [q.id for q in questions]


def test_describe_records_what_the_policy_was_shown(registry6, questions):
    rows = build_rows(questions, registry6, seed=3)
    info = describe(rows)
    assert info["rows"] == len(questions)
    assert info["prompt_version"] == PROMPT_VERSION
    assert info["distinct_pools"] >= 2
    assert sum(info["pool_counts"].values()) == len(questions)
    assert info["prompt_chars_min"] > 0


def test_describe_of_nothing_does_not_divide_by_zero():
    info = describe([])
    assert info["rows"] == 0
    assert info["prompt_chars_min"] == 0


def test_an_unknown_worker_in_a_pool_is_rejected(registry6, questions):
    """A typo in a composition must fail here, not silently narrow the catalogue."""
    with pytest.raises(KeyError):
        build_row(questions[0], registry6, ("a", "nope"))
