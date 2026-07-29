"""Turn rollout rows into the comparison this phase exists to produce.

The headline is one number: does the conductor beat the **best single worker**? Everything else here
supports reading that honestly — parse-failure rate (a conductor that cannot emit valid workflows is
not routing), cost and latency (the paper's result is about efficiency, not only accuracy), and the
count of degenerate rollout groups (no advantage signal to train on).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean

SOLO_PREFIX = "solo:"
CONDUCTOR_PREFIX = "conductor:"


@dataclass(slots=True)
class ArmSummary:
    """Aggregate results for one experimental arm."""

    arm: str
    n: int = 0
    correct: int = 0
    parse_failures: int = 0
    no_answer: int = 0
    failed_final: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    steps: list[int] = field(default_factory=list)
    worker_counts: dict[str, int] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def parse_failure_rate(self) -> float:
        return self.parse_failures / self.n if self.n else 0.0

    @property
    def mean_latency(self) -> float:
        return self.latency_s / self.n if self.n else 0.0

    @property
    def mean_steps(self) -> float:
        return fmean(self.steps) if self.steps else 0.0

    @property
    def mean_tokens(self) -> float:
        return (self.tokens_in + self.tokens_out) / self.n if self.n else 0.0

    @property
    def is_solo(self) -> bool:
        return self.arm.startswith(SOLO_PREFIX)

    @property
    def label(self) -> str:
        for prefix in (SOLO_PREFIX, CONDUCTOR_PREFIX):
            if self.arm.startswith(prefix):
                return self.arm[len(prefix) :]
        return self.arm

    def top_workers(self, limit: int = 3) -> str:
        """Which workers this arm actually routed to — the routing behaviour in one cell."""
        if not self.worker_counts:
            return "-"
        ranked = sorted(self.worker_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return ", ".join(f"{name}×{count}" for name, count in ranked[:limit])


def summarize(rollout_rows: list[dict]) -> list[ArmSummary]:
    """Aggregate ``rollouts.jsonl`` rows per arm, solo arms first and best-accuracy first."""
    arms: dict[str, ArmSummary] = {}
    for row in rollout_rows:
        arm = row.get("arm", "unknown")
        summary = arms.setdefault(arm, ArmSummary(arm=arm))
        summary.n += 1
        summary.correct += 1 if row.get("correct") else 0
        # A ``parse_reason`` is only ever set on an actual parse failure. Testing ``parsed`` instead
        # would count every solo row, which has no workflow to parse in the first place.
        if row.get("parse_reason"):
            summary.parse_failures += 1
        reason = row.get("score_reason")
        if reason == "no_answer":
            summary.no_answer += 1
        elif reason == "final_step_failed":
            summary.failed_final += 1
        summary.cost_usd += float(row.get("cost_usd") or 0.0)
        summary.latency_s += float(row.get("latency_s") or 0.0)
        summary.tokens_in += int(row.get("tokens_in") or 0)
        summary.tokens_out += int(row.get("tokens_out") or 0)
        if row.get("n_steps"):
            summary.steps.append(int(row["n_steps"]))
        for worker in row.get("workers_used") or []:
            summary.worker_counts[worker] = summary.worker_counts.get(worker, 0) + 1

    return sorted(arms.values(), key=lambda s: (not s.is_solo, -s.accuracy, s.arm))


def best_solo(summaries: list[ArmSummary]) -> ArmSummary | None:
    """The best single worker — the bar the conductor has to clear."""
    solos = [s for s in summaries if s.is_solo and s.n]
    return max(solos, key=lambda s: s.accuracy) if solos else None


_COLUMNS = (
    ("arm", 22),
    ("n", 5),
    ("acc", 7),
    ("parse!", 7),
    ("steps", 6),
    ("tok/q", 8),
    ("lat/q", 7),
    ("cost", 9),
    ("routed to", 26),
)


def render_table(summaries: list[ArmSummary]) -> str:
    header = "  ".join(name.ljust(width) for name, width in _COLUMNS)
    rule = "-" * len(header)
    lines = [header, rule]
    for s in summaries:
        cells = [
            (("" if s.is_solo else "→ ") + s.label)[: _COLUMNS[0][1]].ljust(_COLUMNS[0][1]),
            str(s.n).ljust(_COLUMNS[1][1]),
            f"{s.accuracy:.1%}".ljust(_COLUMNS[2][1]),
            (f"{s.parse_failure_rate:.1%}" if not s.is_solo else "-").ljust(_COLUMNS[3][1]),
            (f"{s.mean_steps:.2f}" if not s.is_solo else "-").ljust(_COLUMNS[4][1]),
            f"{s.mean_tokens:.0f}".ljust(_COLUMNS[5][1]),
            f"{s.mean_latency:.2f}s".ljust(_COLUMNS[6][1]),
            f"${s.cost_usd:.4f}".ljust(_COLUMNS[7][1]),
            s.top_workers()[: _COLUMNS[8][1]].ljust(_COLUMNS[8][1]),
        ]
        lines.append("  ".join(cells))
    return "\n".join(lines)


def render_verdict(summaries: list[ArmSummary]) -> str:
    """The phase-0 question, answered in plain words rather than left to the reader."""
    bar = best_solo(summaries)
    conductors = [s for s in summaries if not s.is_solo and s.n]
    if bar is None:
        return "No solo arm ran, so there is no bar to clear."
    if not conductors:
        return f"No conductor arm ran. Best single worker: {bar.label} at {bar.accuracy:.1%}."

    lines = [f"Bar to clear — best single worker: {bar.label} at {bar.accuracy:.1%} ({bar.n} q)."]
    for s in conductors:
        delta = s.accuracy - bar.accuracy
        sign = "+" if delta >= 0 else ""
        verdict = "BEATS" if delta > 0 else ("ties" if delta == 0 else "loses to")
        lines.append(
            f"  {s.label}: {s.accuracy:.1%} ({sign}{delta:.1%}) — {verdict} the best single worker."
        )
        if s.parse_failure_rate > 0:
            lines.append(
                f"    {s.parse_failures}/{s.n} rollouts never parsed "
                f"({s.parse_failure_rate:.1%}) — scored zero, as designed."
            )
    return "\n".join(lines)


def render_parse_failures(rollout_rows: list[dict], limit: int = 8) -> str:
    """Failure reasons, most common first. These are the training metrics phase 3 will watch."""
    counts: dict[str, int] = {}
    for row in rollout_rows:
        reason = row.get("parse_reason")
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return "No parse failures."
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return "\n".join(f"  {reason}: {count}" for reason, count in ranked)
