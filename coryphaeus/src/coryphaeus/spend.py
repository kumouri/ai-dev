"""Token-spend ledger — the cloud ledger's discipline, pointed at per-token worker providers.

Featherless is flat-rate: the account's concurrency units meter throughput, and money is settled
by the subscription. OpenRouter is per-token **with auto-top-up enabled**, which means nothing
upstream ever refuses runaway spend — a retry loop or a runaway calibration would be billed
politely and indefinitely. This ledger is that refusal. Same append-only reserve/settle journal,
same over-count-only crash behaviour (see ``coryphaeus.cloud.budget``), separate file and separate
ceiling: GPU rental and token spend are different bills, and folding them together would let one
quietly starve the other's month.

Scripts reserve a worst-case estimate before a paid pool run and settle with the actual afterwards.
The actual comes from telemetry: every OpenRouter rollout row carries ``cost_usd`` computed from
the provider's returned usage and the manifest's pinned prices, so the settle is a sum over the
run's own receipts rather than a second guess.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .cloud.budget import BudgetLedger

TOKEN_BUDGET_ENV = "CORYPHAEUS_TOKEN_BUDGET_USD"

#: Filename convention, kept beside the runs it paid for — a sibling of the cloud ledger's
#: ``budget.jsonl``, not inside any single run directory.
TOKEN_LEDGER_FILENAME = "token-budget.jsonl"


def token_ledger(path: Path, *, ceiling_usd: float | None = None) -> BudgetLedger:
    """The token-spend ledger: a ``BudgetLedger`` under ``CORYPHAEUS_TOKEN_BUDGET_USD``.

    ``path`` may be the ledger file itself or a directory (the conventional filename is appended).
    """
    if path.suffix != ".jsonl":
        path = path / TOKEN_LEDGER_FILENAME
    return BudgetLedger(path, ceiling_usd=ceiling_usd, env_var=TOKEN_BUDGET_ENV)


def sum_cost_usd(paths: Iterable[Path]) -> float:
    """Total ``cost_usd`` across JSONL telemetry files: the settle amount for a paid run.

    Rows without a ``cost_usd`` (local and flat-rate workers) count as zero — absence of a receipt
    is not a cost. Files that do not exist contribute nothing: a run that produced no telemetry
    spent nothing, and raising here would block the settle that records exactly that.
    """
    total = 0.0
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                total += float(row.get("cost_usd") or 0.0)
    return round(total, 6)
