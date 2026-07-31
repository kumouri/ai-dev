"""The spend ledger — a hard monthly gate on provisioning, and the receipts behind it.

Same philosophy as the concurrency governor: the refusal must be complete. "Budget exceeded" alone
sends a human digging; "spent $47.10 of $50.00 this month, resets 2026-08-01" is a decision.

The ledger is an append-only JSONL (one row per provision/settle event) plus derived monthly
rollups. Append-only because billing disputes are settled by receipts, not by a mutable counter —
and because a crashed launcher must never be able to *lose* a spend record. Rows are written
BEFORE provisioning (a reservation at the estimated cost) and settled AFTER termination with the
actual; an unsettled reservation counts at its estimate, so a crash between the two can only
over-count, never under-count. Fail-closed, in money's favor.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_CEILING_USD = 50.0


class BudgetExceeded(RuntimeError):
    """Provisioning would pass the monthly ceiling. Message carries spent/ceiling/reset."""


@dataclass(frozen=True, slots=True)
class MonthlySpend:
    month: str  # "2026-07"
    settled_usd: float
    reserved_usd: float
    ceiling_usd: float

    @property
    def total_usd(self) -> float:
        return self.settled_usd + self.reserved_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.ceiling_usd - self.total_usd)


def _now() -> datetime:
    return datetime.now(UTC)


def _month_of(stamp: str) -> str:
    return stamp[:7]


class BudgetLedger:
    """Append-only spend journal with a monthly ceiling.

    Args:
        path: the JSONL file. Lives under the run/telemetry tree by default — it is a *record*,
            not config, and it must survive any individual run directory being cleaned up.
        ceiling_usd: monthly cap. Overridable via ``CORYPHAEUS_CLOUD_BUDGET_USD`` so a public
            clone's default is conservative and a power user raises it in env, not code.
    """

    def __init__(self, path: Path, *, ceiling_usd: float | None = None) -> None:
        self.path = path
        env_ceiling = os.environ.get("CORYPHAEUS_CLOUD_BUDGET_USD")
        self.ceiling_usd = (
            ceiling_usd
            if ceiling_usd is not None
            else float(env_ceiling)
            if env_ceiling
            else DEFAULT_CEILING_USD
        )

    # --- reading --------------------------------------------------------------------------------

    def rows(self) -> list[dict]:
        if not self.path.is_file():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def month_spend(self, month: str | None = None) -> MonthlySpend:
        month = month or _month_of(_now().isoformat())
        settled = 0.0
        reserved: dict[str, float] = {}
        for row in self.rows():
            if _month_of(row["at"]) != month:
                continue
            if row["event"] == "reserve":
                reserved[row["reservation_id"]] = float(row["estimated_usd"])
            elif row["event"] == "settle":
                reserved.pop(row["reservation_id"], None)
                settled += float(row["actual_usd"])
        return MonthlySpend(
            month=month,
            settled_usd=round(settled, 4),
            reserved_usd=round(sum(reserved.values()), 4),
            ceiling_usd=self.ceiling_usd,
        )

    # --- writing --------------------------------------------------------------------------------

    def _append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def reserve(self, reservation_id: str, estimated_usd: float, *, note: str = "") -> None:
        """Record intent to spend BEFORE provisioning. Raises BudgetExceeded past the ceiling.

        The estimate should be the run's worst case (max runtime x price + volume), not its hope.
        An unsettled reservation keeps counting at this estimate, so a launcher that crashes
        without settling leaves the ledger over-counting — the failure mode that costs vigilance,
        never money.
        """
        spend = self.month_spend()
        if spend.total_usd + estimated_usd > self.ceiling_usd:
            resets = (_now().replace(day=28) + timedelta(days=4)).replace(day=1)
            raise BudgetExceeded(
                f"provisioning ~${estimated_usd:.2f} would pass the monthly ceiling: "
                f"${spend.settled_usd:.2f} settled + ${spend.reserved_usd:.2f} reserved of "
                f"${self.ceiling_usd:.2f} for {spend.month}; resets {resets:%Y-%m-%d}. Raise "
                f"CORYPHAEUS_CLOUD_BUDGET_USD deliberately if this is intended."
            )
        self._append(
            {
                "event": "reserve",
                "reservation_id": reservation_id,
                "estimated_usd": round(estimated_usd, 4),
                "note": note,
                "at": _now().isoformat(),
            }
        )

    def settle(self, reservation_id: str, actual_usd: float, *, note: str = "") -> None:
        """Replace a reservation with what the run actually cost. Idempotent per reservation —
        the launcher settles from its normal exit AND its crash handler, whichever runs last wins
        nothing (the first settle already removed the reservation; a second is a no-op row)."""
        already = any(
            row["event"] == "settle" and row["reservation_id"] == reservation_id
            for row in self.rows()
        )
        if already:
            return
        self._append(
            {
                "event": "settle",
                "reservation_id": reservation_id,
                "actual_usd": round(actual_usd, 4),
                "note": note,
                "at": _now().isoformat(),
            }
        )
