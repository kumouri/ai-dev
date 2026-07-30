"""The r5 guardrails: the zero-variance gate's arithmetic and the deadline decision.

Both exist because r4 demonstrated their absence: 56% of its GPU time bought zero gradient, and it
was ~26 minutes from busting a 6-hour deadline nothing enforced when a CUDA fault got there first.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from coryphaeus.train.grpo import deadline_reached

# The gate lives in scripts/, which is not a package — load it the way Python actually can.
_spec = importlib.util.spec_from_file_location(
    "gate_zero_std",
    Path(__file__).resolve().parents[1] / "scripts" / "gate_zero_std.py",
)
_gate = importlib.util.module_from_spec(_spec)
sys.modules["gate_zero_std"] = _gate
_spec.loader.exec_module(_gate)
zero_variance_fraction = _gate.zero_variance_fraction


def rollout(qid: str, score: float) -> dict:
    return {"question_id": qid, "score": score}


def test_unanimous_groups_are_counted_as_dead():
    rows = [rollout("q1", 1.0)] * 4 + [rollout("q2", 0.0)] * 4
    assert zero_variance_fraction(rows) == (2, 2)


def test_mixed_groups_are_alive():
    rows = [rollout("q1", 1.0), rollout("q1", 0.0), rollout("q1", 1.0), rollout("q1", 1.0)]
    assert zero_variance_fraction(rows) == (0, 1)


def test_incomplete_groups_do_not_count_either_way():
    """A 3-rollout group at crash time must not sway the verdict."""
    rows = [rollout("q1", 1.0)] * 3 + [rollout("q2", 1.0), rollout("q2", 0.0)] * 2
    dead, total = zero_variance_fraction(rows)
    assert total == 1  # only q2 is complete
    assert dead == 0


def test_only_the_first_four_rollouts_form_the_group():
    """Repeated questions (2-epoch sets) append more rollouts; the group is the first k."""
    rows = [rollout("q1", 1.0)] * 4 + [rollout("q1", 0.0)] * 4
    assert zero_variance_fraction(rows) == (1, 1)


def test_the_r4_scenario_fails_a_20_percent_gate():
    """22 unanimous-correct + 5 unanimous-wrong + 33 mixed = 45% dead — the measured r4 outcome."""
    rows = []
    for i in range(22):
        rows += [rollout(f"easy{i}", 1.0)] * 4
    for i in range(5):
        rows += [rollout(f"hard{i}", 0.0)] * 4
    for i in range(33):
        rows += [rollout(f"mix{i}", 1.0), rollout(f"mix{i}", 0.0)] * 2
    dead, total = zero_variance_fraction(rows)
    assert (dead, total) == (27, 60)
    assert dead / total > 0.20  # r4 would not have shipped


def test_empty_telemetry_is_no_verdict_material():
    assert zero_variance_fraction([]) == (0, 0)


# --- the deadline decision -----------------------------------------------------------------------


def test_deadline_off_never_fires():
    assert deadline_reached(0.0, 1e9, 0.0) is False


def test_deadline_fires_exactly_at_budget():
    start = 1000.0
    assert deadline_reached(start, start + 6 * 3600.0, 6.0) is True
    assert deadline_reached(start, start + 6 * 3600.0 - 1, 6.0) is False


def test_deadline_handles_fractional_hours():
    start = 0.0
    assert deadline_reached(start, 0.5 * 3600.0, 0.5) is True
    assert deadline_reached(start, 0.4 * 3600.0, 0.5) is False
