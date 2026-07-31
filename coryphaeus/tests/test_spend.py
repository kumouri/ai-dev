"""The token-spend ledger — the guard between auto-top-up and a runaway per-token bill.

The mechanism (append-only, over-count-only) is tested to death in test_cloud_budget.py; these
tests pin what is *different* about the token ledger: its own env override, its own file, and the
telemetry-sum settle path.
"""

import json
from pathlib import Path

import pytest

from coryphaeus.cloud.budget import BudgetExceeded, BudgetLedger
from coryphaeus.spend import (
    EST_TOKENS_IN,
    TOKEN_LEDGER_FILENAME,
    reserve_worst_case,
    sum_cost_usd,
    token_ledger,
)
from coryphaeus.workers.featherless import featherless_spec
from coryphaeus.workers.openrouter import openrouter_spec


class _Holder:
    def __init__(self, spec):
        self.spec = spec


class _StubRegistry:
    """Just enough registry for reserve_worst_case: get(name).spec.cost()."""

    def __init__(self, *specs):
        self._workers = {s.name: _Holder(s) for s in specs}

    def get(self, name):
        return self._workers[name]


def write_jsonl(path: Path, rows: list) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# --- the factory ---------------------------------------------------------------------------------


def test_token_ledger_appends_conventional_filename_to_a_directory(tmp_path):
    ledger = token_ledger(tmp_path)
    assert ledger.path == tmp_path / TOKEN_LEDGER_FILENAME


def test_token_ledger_accepts_an_explicit_file(tmp_path):
    ledger = token_ledger(tmp_path / "custom.jsonl")
    assert ledger.path == tmp_path / "custom.jsonl"


def test_token_ledger_reads_its_own_env_var_not_the_cloud_one(tmp_path, monkeypatch):
    monkeypatch.setenv("CORYPHAEUS_TOKEN_BUDGET_USD", "12.5")
    monkeypatch.setenv("CORYPHAEUS_CLOUD_BUDGET_USD", "999")
    assert token_ledger(tmp_path).ceiling_usd == 12.5


def test_cloud_ledger_still_reads_the_cloud_env_var(tmp_path, monkeypatch):
    """Regression for the env_var parameter: the default must not have moved."""
    monkeypatch.setenv("CORYPHAEUS_CLOUD_BUDGET_USD", "77")
    monkeypatch.delenv("CORYPHAEUS_TOKEN_BUDGET_USD", raising=False)
    assert BudgetLedger(tmp_path / "budget.jsonl").ceiling_usd == 77.0


def test_refusal_names_the_token_env_var(tmp_path, monkeypatch):
    """The refusal must tell the operator which knob raises *this* ceiling — a token-budget
    refusal that says CORYPHAEUS_CLOUD_BUDGET_USD sends them to the wrong bill."""
    monkeypatch.delenv("CORYPHAEUS_TOKEN_BUDGET_USD", raising=False)
    ledger = token_ledger(tmp_path, ceiling_usd=1.0)
    with pytest.raises(BudgetExceeded, match="CORYPHAEUS_TOKEN_BUDGET_USD"):
        ledger.reserve("r1", 2.0)


def test_token_and_cloud_ledgers_are_separate_files(tmp_path):
    """One month of GPU rental must not eat the token ceiling, and vice versa."""
    cloud = BudgetLedger(tmp_path / "budget.jsonl", ceiling_usd=50.0)
    tokens = token_ledger(tmp_path, ceiling_usd=50.0)
    cloud.reserve("gpu-1", 49.0)
    tokens.reserve("cal-1", 49.0)  # would raise if the cloud reservation were visible here
    assert cloud.month_spend().total_usd == 49.0
    assert tokens.month_spend().total_usd == 49.0


# --- the shared reservation helper ---------------------------------------------------------------


def test_reserve_worst_case_prices_only_the_paid_workers(tmp_path):
    """Free workers may appear in the call plan freely — their cost is zero, and pricing them
    would inflate the reservation for nothing."""
    registry = _StubRegistry(
        featherless_spec("free", "x/free"),
        openrouter_spec("paid", "x/paid", usd_per_mtok_in=1.0, usd_per_mtok_out=1.0),
    )
    ledger = token_ledger(tmp_path)
    got_ledger, reservation_id = reserve_worst_case(
        registry, {"free": 100, "paid": 10}, 1024, "unit-test", ledger=ledger
    )
    assert got_ledger is ledger and reservation_id is not None
    expected = round(10 * (EST_TOKENS_IN * 1.0 + 1024 * 1.0) / 1_000_000, 4)
    assert ledger.month_spend().reserved_usd == expected


def test_reserve_worst_case_is_a_no_op_for_flat_rate_pools(tmp_path):
    registry = _StubRegistry(featherless_spec("free", "x/free"))
    ledger = token_ledger(tmp_path)
    assert reserve_worst_case(registry, {"free": 500}, 1024, "x", ledger=ledger) == (None, None)
    assert ledger.rows() == []


# --- the settle source ---------------------------------------------------------------------------


def test_sum_cost_usd_totals_across_files(tmp_path):
    a = write_jsonl(tmp_path / "a.jsonl", [{"cost_usd": 0.001}, {"cost_usd": 0.002}])
    b = write_jsonl(tmp_path / "b.jsonl", [{"cost_usd": 0.0005}])
    assert sum_cost_usd([a, b]) == 0.0035


def test_sum_cost_usd_treats_missing_and_null_cost_as_zero(tmp_path):
    """Local and flat-rate workers have no receipt; that is not a cost and not an error."""
    path = write_jsonl(
        tmp_path / "mixed.jsonl",
        [{"cost_usd": 0.01}, {"worker": "q9"}, {"cost_usd": None}],
    )
    assert sum_cost_usd([path]) == 0.01


def test_sum_cost_usd_skips_blank_lines_and_missing_files(tmp_path):
    path = tmp_path / "gappy.jsonl"
    path.write_text('{"cost_usd": 0.5}\n\n{"cost_usd": 0.25}\n', encoding="utf-8")
    assert sum_cost_usd([path, tmp_path / "never-written.jsonl"]) == 0.75


def test_reserve_then_settle_from_telemetry_round_trip(tmp_path):
    """The intended calling convention end to end: worst-case reserve, settle with the sum of
    the run's own receipts, month reflects the actual."""
    telemetry = write_jsonl(
        tmp_path / "rollouts.jsonl",
        [{"cost_usd": 0.004}, {"cost_usd": 0.006}, {"worker": "local"}],
    )
    ledger = token_ledger(tmp_path, ceiling_usd=50.0)
    ledger.reserve("calibrate-500", 2.0, note="500q x 6 samples, worst case")
    ledger.settle("calibrate-500", sum_cost_usd([telemetry]), note="from rollouts.jsonl")
    spend = ledger.month_spend()
    assert spend.reserved_usd == 0.0
    assert spend.settled_usd == 0.01
