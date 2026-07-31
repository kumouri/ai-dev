"""The budget ledger — the component that guards actual dollars, tested to its failure modes.

The design property under test throughout: the ledger can only ever OVER-count after a crash,
never under-count. Reservations are written before provisioning at worst-case estimates and only
replaced by settle; a launcher that dies between the two leaves its estimate standing. Vigilance
is the cheap failure; silent spend is the expensive one.
"""

from __future__ import annotations

import json

import pytest

from coryphaeus.cloud import BudgetExceeded, BudgetLedger


@pytest.fixture
def ledger(tmp_path):
    return BudgetLedger(tmp_path / "spend.jsonl", ceiling_usd=50.0)


def test_reserve_then_settle_moves_estimate_to_actual(ledger):
    ledger.reserve("run-1", 4.00, note="validate run, 8h cap x 0.34 + volume")
    spend = ledger.month_spend()
    assert spend.reserved_usd == 4.00
    assert spend.settled_usd == 0.0

    ledger.settle("run-1", 1.10, note="terminated after 3.2h")
    spend = ledger.month_spend()
    assert spend.reserved_usd == 0.0
    assert spend.settled_usd == 1.10


def test_a_crashed_launcher_leaves_the_reservation_counting(ledger):
    """The whole point: no settle ever arrives, so the worst-case estimate keeps counting."""
    ledger.reserve("run-crash", 5.00)
    assert ledger.month_spend().total_usd == 5.00  # never silently zero


def test_ceiling_refusal_is_a_complete_decision(ledger):
    ledger.reserve("run-1", 30.00)
    ledger.settle("run-1", 18.00)
    ledger.reserve("run-2", 20.00)
    with pytest.raises(BudgetExceeded) as exc:
        ledger.reserve("run-3", 15.00)
    message = str(exc.value)
    # "Budget exceeded" alone sends a human digging; the message must BE the decision.
    assert "$18.00 settled" in message
    assert "$20.00 reserved" in message
    assert "$50.00" in message
    assert "resets" in message
    assert "CORYPHAEUS_CLOUD_BUDGET_USD" in message


def test_exactly_at_the_ceiling_is_allowed(ledger):
    ledger.reserve("run-1", 50.00)  # <= ceiling: allowed; the gate is on EXCEEDING


def test_one_dollar_past_the_ceiling_is_not(ledger):
    ledger.reserve("run-1", 45.00)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("run-2", 6.00)


def test_settle_is_idempotent_because_crash_handlers_also_settle(ledger):
    """The launcher settles from its normal exit AND its finally block; both may run."""
    ledger.reserve("run-1", 4.00)
    ledger.settle("run-1", 2.00)
    ledger.settle("run-1", 99.00)  # the duplicate must be a no-op, not a second charge
    assert ledger.month_spend().settled_usd == 2.00


def test_env_var_overrides_the_default_ceiling(tmp_path, monkeypatch):
    monkeypatch.setenv("CORYPHAEUS_CLOUD_BUDGET_USD", "10")
    ledger = BudgetLedger(tmp_path / "spend.jsonl")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("run-1", 11.00)


def test_explicit_ceiling_beats_the_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("CORYPHAEUS_CLOUD_BUDGET_USD", "10")
    ledger = BudgetLedger(tmp_path / "spend.jsonl", ceiling_usd=100.0)
    ledger.reserve("run-1", 60.00)  # explicit constructor arg wins


def test_the_ledger_is_receipts_not_a_counter(ledger):
    """Append-only: settle does not rewrite the reserve row; both survive as evidence."""
    ledger.reserve("run-1", 4.00)
    ledger.settle("run-1", 1.10)
    rows = [json.loads(line) for line in ledger.path.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in rows] == ["reserve", "settle"]
    assert rows[0]["estimated_usd"] == 4.00  # the original estimate is preserved, not overwritten


def test_missing_file_means_zero_spend(tmp_path):
    ledger = BudgetLedger(tmp_path / "never-written.jsonl")
    spend = ledger.month_spend()
    assert spend.total_usd == 0.0
    assert spend.remaining_usd == 50.0


def test_multiple_concurrent_reservations_accumulate(ledger):
    """Four parallel seeds at $1.36/hr was the researched use case — they must all count."""
    for seed in range(4):
        ledger.reserve(f"seed-{seed}", 11.00)
    assert ledger.month_spend().reserved_usd == 44.00
    with pytest.raises(BudgetExceeded):
        ledger.reserve("seed-5", 11.00)
