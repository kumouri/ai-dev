"""The oracle ceiling — how much routing could possibly be worth in a given pool.

This exists because phase 0's negative result turned out to be mostly a *pool* property: 34% of
questions were solved by no worker, so a perfect router capped only 7.3 points above the best single
worker. A routing experiment on a pool with no headroom cannot succeed, and the failure reads as the
conductor's fault. Measuring the ceiling first is what stops that misattribution.
"""

from __future__ import annotations

from coryphaeus.reporting import oracle_ceiling, render_oracle


def row(arm: str, qid: str, correct: bool, routed: str | None = None) -> dict:
    return {
        "arm": arm,
        "question_id": qid,
        "correct": correct,
        "parse_reason": None,
        "score_reason": "ok",
        "workers_used": [routed] if routed else [],
        "n_steps": 1,
    }


def test_ceiling_and_headroom_on_a_complementary_pool():
    """Two workers each solving a disjoint half: ceiling 100%, best solo 50%, headroom +50%."""
    rows = []
    for i in range(10):
        rows.append(row("solo:a", f"q{i}", i % 2 == 0))
        rows.append(row("solo:b", f"q{i}", i % 2 == 1))
    view = oracle_ceiling(rows)
    assert view is not None
    assert view.n_questions == 10
    assert view.solved_by_someone == 10
    assert view.ceiling == 1.0
    assert view.best_solo_accuracy == 0.5
    assert view.headroom == 0.5


def test_a_correlated_pool_has_almost_no_headroom():
    """The phase-0 situation: one worker dominates, so routing can barely help."""
    rows = []
    for i in range(10):
        strong = i < 6
        rows.append(row("solo:big", f"q{i}", strong))
        rows.append(row("solo:small", f"q{i}", i < 4))  # a strict subset
    view = oracle_ceiling(rows)
    assert view is not None
    assert view.best_solo_accuracy == 0.6
    assert view.ceiling == 0.6
    assert view.headroom == 0.0
    assert "Small" in render_oracle(view)
    assert "more about the pool" in render_oracle(view)


def test_unsolvable_questions_lower_the_ceiling():
    rows = []
    for i in range(10):
        rows.append(row("solo:a", f"q{i}", i < 3))
        rows.append(row("solo:b", f"q{i}", 3 <= i < 5))
    view = oracle_ceiling(rows)
    assert view is not None
    assert view.solved_by_someone == 5
    assert view.unsolvable == 5
    assert view.ceiling == 0.5


def test_routing_quality_counts_only_solvable_questions():
    """Crediting or blaming a router for an unsolvable question measures nothing."""
    rows = [
        row("solo:a", "q1", True),
        row("solo:b", "q1", False),
        row("solo:a", "q2", False),
        row("solo:b", "q2", False),  # nobody solved q2
        row("conductor:c", "q1", True, routed="a"),
        row("conductor:c", "q2", False, routed="b"),
    ]
    view = oracle_ceiling(rows)
    assert view is not None
    hit, miss = view.routing["conductor:c"]
    assert (hit, miss) == (1, 0)  # q2 excluded entirely


def test_a_router_choosing_the_wrong_worker_is_counted():
    rows = [
        row("solo:a", "q1", True),
        row("solo:b", "q1", False),
        row("conductor:c", "q1", False, routed="b"),
    ]
    view = oracle_ceiling(rows)
    assert view is not None
    assert view.routing["conductor:c"] == (0, 1)


def test_only_the_shared_question_set_is_used():
    """An early-stopped run leaves arms at unequal n; comparing across slices is the trap."""
    rows = [
        row("solo:a", "q1", True),
        row("solo:a", "q2", True),
        row("solo:a", "q3", True),  # 'a' saw an extra question
        row("solo:b", "q1", False),
        row("solo:b", "q2", False),
    ]
    view = oracle_ceiling(rows)
    assert view is not None
    assert view.n_questions == 2  # q3 dropped


def test_one_solo_arm_has_no_ceiling_to_speak_of():
    rows = [row("solo:a", "q1", True)]
    assert oracle_ceiling(rows) is None
    assert "two or more solo arms" in render_oracle(None)


def test_no_shared_questions_returns_nothing():
    rows = [row("solo:a", "q1", True), row("solo:b", "q2", True)]
    assert oracle_ceiling(rows) is None


def test_empty_input_is_not_a_crash():
    assert oracle_ceiling([]) is None


def test_render_states_the_headroom_explicitly():
    rows = []
    for i in range(10):
        rows.append(row("solo:a", f"q{i}", i % 2 == 0))
        rows.append(row("solo:b", f"q{i}", i % 2 == 1))
    text = render_oracle(oracle_ceiling(rows))
    assert "ROUTING HEADROOM" in text
    assert "+50.0%" in text
    assert "perfect router      : 100.0%" in text
