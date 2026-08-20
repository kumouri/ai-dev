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


def rollout(qid: str, score: float, parsed: bool = True) -> dict:
    return {"question_id": qid, "score": score, "parsed": parsed}


def test_unanimous_groups_are_counted_as_dead():
    rows = [rollout("q1", 1.0)] * 4 + [rollout("q2", 0.0)] * 4
    assert zero_variance_fraction(rows) == (2, 0, 2)


def test_mixed_groups_are_alive():
    rows = [rollout("q1", 1.0), rollout("q1", 0.0), rollout("q1", 1.0), rollout("q1", 1.0)]
    assert zero_variance_fraction(rows) == (0, 0, 1)


def test_incomplete_groups_do_not_count_either_way():
    """A 3-rollout group at crash time must not sway the verdict."""
    rows = [rollout("q1", 1.0)] * 3 + [rollout("q2", 1.0), rollout("q2", 0.0)] * 2
    dead, storms, total = zero_variance_fraction(rows)
    assert total == 1  # only q2 is complete
    assert (dead, storms) == (0, 0)


def test_only_the_first_four_rollouts_form_the_group():
    """Repeated questions (2-epoch sets) append more rollouts; the group is the first k."""
    rows = [rollout("q1", 1.0)] * 4 + [rollout("q1", 0.0)] * 4
    assert zero_variance_fraction(rows) == (1, 0, 1)


def test_the_r4_scenario_fails_a_20_percent_gate():
    """22 unanimous-correct + 5 unanimous-wrong + 33 mixed = 45% dead — the measured r4 outcome.
    r4's unanimous-wrong groups were PARSED (its parse rate was far above the storm line), so
    the parse-aware accounting changes nothing about the r4 verdict."""
    rows = []
    for i in range(22):
        rows += [rollout(f"easy{i}", 1.0)] * 4
    for i in range(5):
        rows += [rollout(f"hard{i}", 0.0)] * 4
    for i in range(33):
        rows += [rollout(f"mix{i}", 1.0), rollout(f"mix{i}", 0.0)] * 2
    dead, storms, total = zero_variance_fraction(rows)
    assert (dead, storms, total) == (27, 0, 60)
    assert dead / total > 0.20  # r4 would not have shipped


def test_empty_telemetry_is_no_verdict_material():
    assert zero_variance_fraction([]) == (0, 0, 0)


def test_a_parse_storm_group_is_policy_noise_not_question_deadness():
    """Zero-variance at score 0 with <=1/4 parsed says the POLICY failed to emit workflows, not
    that the question is dead — the 2026-08-01/20 probes' unanimous-wrong groups were all this
    shape (0/4 or 1/4 parsed, json_decode / too_many_steps / bad_final)."""
    rows = [rollout("q1", 0.0, parsed=False)] * 4
    rows += [rollout("q2", 0.0, parsed=False)] * 3 + [rollout("q2", 0.0, parsed=True)]
    assert zero_variance_fraction(rows) == (0, 2, 2)


def test_an_engaged_zero_score_group_still_counts_dead():
    """Two of four parsed and all four at 0 = the policy genuinely engaged and the question
    bought no gradient anyway — that IS question signal, and it stays counted."""
    rows = [
        rollout("q1", 0.0, parsed=True),
        rollout("q1", 0.0, parsed=True),
        rollout("q1", 0.0, parsed=False),
        rollout("q1", 0.0, parsed=False),
    ]
    assert zero_variance_fraction(rows) == (1, 0, 1)


def test_unanimous_correct_is_never_a_storm():
    """The r4 disease — too-easy questions — is parsed by construction and must stay counted
    no matter what the parse accounting does."""
    rows = [rollout("q1", 1.0, parsed=True)] * 4
    assert zero_variance_fraction(rows) == (1, 0, 1)


def test_rows_without_a_parsed_field_reproduce_the_old_accounting():
    """Pre-2026-08 telemetry has no ``parsed`` key; missing counts as parsed, so old runs read
    exactly as they always did."""
    rows = [{"question_id": "q1", "score": 0.0}] * 4
    assert zero_variance_fraction(rows) == (1, 0, 1)


def test_the_0820_probe_scenario_passes_a_20_percent_gate():
    """The measured 2026-08-20 probe: 15 mixed + 3 parse-storms + 2 unanimous-correct read 25%
    under the old accounting (FAIL); the question signal is 2/20 = 10% (PASS). This is the
    regression test for the night the gate mistook policy infancy for a mis-calibrated set."""
    rows = []
    for i in range(15):
        rows += [rollout(f"mix{i}", 1.0), rollout(f"mix{i}", 0.0)] * 2
    for i in range(3):
        rows += [rollout(f"storm{i}", 0.0, parsed=False)] * 4
    for i in range(2):
        rows += [rollout(f"easy{i}", 1.0)] * 4
    dead, storms, total = zero_variance_fraction(rows)
    assert (dead, storms, total) == (2, 3, 20)
    assert dead / total < 0.20  # ships
    assert (dead + storms) / total >= 0.20  # the old accounting would have refused it


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
