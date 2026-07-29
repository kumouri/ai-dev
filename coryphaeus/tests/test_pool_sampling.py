"""Subset views and randomised pool composition — all offline."""

from __future__ import annotations

import random

import pytest

from coryphaeus.pools import PoolSplit, sample_pool, sample_pool_split
from coryphaeus.workers import WorkerRegistry, fake_spec, units_for_params
from coryphaeus.workers.fake import FakeWorker
from coryphaeus.workflow import parse


@pytest.fixture
def big_registry():
    """Six fake workers — enough to draw many distinct sub-pools from."""
    return WorkerRegistry(
        FakeWorker(fake_spec(name), answers={"q": "1"}, seed=i)
        for i, name in enumerate(("a", "b", "c", "d", "e", "f"))
    )


def test_subset_shares_worker_objects(big_registry):
    """Governor backoff state and call counters are per-worker; a subset must not fork them."""
    view = big_registry.subset(("a", "c"))
    assert view.names() == ("a", "c")
    assert view.get("a") is big_registry.get("a")


def test_subset_preserves_the_order_given(big_registry):
    """Order is part of the composition: it is the order the catalog lists workers in."""
    assert big_registry.subset(("c", "a", "b")).names() == ("c", "a", "b")


def test_subset_of_an_unknown_name_raises(big_registry):
    """A sample referencing a worker we do not have is a bug, not something to filter away."""
    with pytest.raises(KeyError):
        big_registry.subset(("a", "nope"))


def test_catalog_text_tracks_the_subset(big_registry):
    full = big_registry.catalog_text()
    view = big_registry.subset(("a", "b")).catalog_text()
    assert len(view.splitlines()) == 2
    assert len(full.splitlines()) == 6
    assert "- f " not in view


def test_out_of_pool_routing_is_already_a_named_failure(big_registry):
    """No new validation needed: parse() checks against whatever pool it is handed."""
    view = big_registry.subset(("a", "b"))
    raw = '```json\n{"steps": [{"subtask": "solve", "worker": "f"}]}\n```'
    assert parse(raw, known_workers=view.names()).reason == "unknown_worker"
    assert parse(raw, known_workers=big_registry.names()).ok  # 'f' is fine in the full pool


def test_sampling_is_deterministic_per_seed(big_registry):
    a = sample_pool(big_registry, random.Random(7))
    b = sample_pool(big_registry, random.Random(7))
    assert a == b


def test_different_seeds_give_different_compositions(big_registry):
    draws = {sample_pool(big_registry, random.Random(s)) for s in range(20)}
    assert len(draws) > 1


def test_sample_respects_an_exact_size(big_registry):
    for _ in range(10):
        assert len(sample_pool(big_registry, random.Random(1), size=3)) == 3


def test_sample_respects_a_range(big_registry):
    sizes = {len(sample_pool(big_registry, random.Random(s), size=(2, 4))) for s in range(30)}
    assert sizes <= {2, 3, 4}
    assert len(sizes) > 1


def test_sample_never_returns_a_single_worker_when_two_are_available(big_registry):
    """A one-worker pool has nothing to route between, so it is not a meaningful composition."""
    for s in range(30):
        assert len(sample_pool(big_registry, random.Random(s), size=1)) >= 2


def test_sample_of_a_one_worker_registry_is_allowed():
    solo = WorkerRegistry([FakeWorker(fake_spec("only"), answers={})])
    assert sample_pool(solo, random.Random(0)) == ("only",)


def test_sample_of_an_empty_registry_raises():
    with pytest.raises(ValueError, match="empty registry"):
        sample_pool(WorkerRegistry(), random.Random(0))


def test_sample_clamps_to_the_registry_size(big_registry):
    assert len(sample_pool(big_registry, random.Random(0), size=99)) == 6


def test_split_holds_train_and_eval_compositions_apart(big_registry):
    split = sample_pool_split(big_registry, seed=3, n_train=6, n_eval=3)
    assert len(split.train) == 6
    assert len(split.evaluation) == 3
    assert not set(split.train) & set(split.evaluation)
    assert len(set(split.train)) == 6  # distinct within the train side too


def test_split_is_reproducible(big_registry):
    a = sample_pool_split(big_registry, seed=11)
    b = sample_pool_split(big_registry, seed=11)
    assert a == b


def test_split_refuses_when_the_registry_is_too_small():
    """Silently returning fewer compositions would make a leak look like a pass."""
    small = WorkerRegistry(FakeWorker(fake_spec(n), answers={}) for n in ("a", "b", "c"))
    with pytest.raises(ValueError, match="distinct compositions"):
        sample_pool_split(small, n_train=20, n_eval=10, size=3)


def test_pool_split_rejects_a_leaking_split():
    with pytest.raises(ValueError, match="leak"):
        PoolSplit(train=(("a", "b"),), evaluation=(("a", "b"),))


@pytest.mark.parametrize(
    ("params_b", "expected"),
    [
        (7.0, 1),
        (14.0, 1),
        (32.0, 2),  # the pinned counter-example: the guess used to say 4
        (27.0, 2),
        (70.0, 4),
        (72.0, 4),
        (None, 4),  # unknown estimates high; guessing cheap is what causes 429s
    ],
)
def test_units_fallback_has_four_tiers_not_two(params_b, expected):
    """`units_for_params` is a fallback, but a wrong fallback halves concurrency in the 24-32B band.

    The provider's own `concurrency_cost` remains the source of truth via the pinned manifest.
    """
    assert units_for_params(params_b) == expected
