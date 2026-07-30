"""The go/no-go gate between a probe run and a full run.

r4 spent 56% of its GPU time on zero-variance groups. A ~20-step probe measures that fraction on
the *current* question set with the *current* policy; the full run ships only if it clears the
threshold. The probe's checkpoints resume into the full run, so a passed gate costs nothing.

Exit 0 = ship it. Exit 1 = the set is still mis-calibrated; fix the filter, not the trainer.

    uv run python coryphaeus/scripts/gate_zero_std.py --label grpo15b-r5 --threshold 0.2
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

from coryphaeus.config import settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="run label; the newest matching run is read")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument(
        "--min-groups",
        type=int,
        default=12,
        help="refuse to rule on fewer complete groups than this — a verdict from n=3 is noise "
        "wearing a badge",
    )
    return parser.parse_args(argv)


def zero_variance_fraction(rollout_rows: list[dict]) -> tuple[int, int]:
    """(zero-variance groups, complete groups) over rollout telemetry."""
    by_question: dict[str, list[float]] = collections.defaultdict(list)
    for row in rollout_rows:
        by_question[row["question_id"]].append(row["score"])
    complete = {q: s[:4] for q, s in by_question.items() if len(s) >= 4}
    dead = sum(1 for scores in complete.values() if len(set(scores)) == 1)
    return dead, len(complete)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs = sorted(
        p for p in settings().runs_dir.glob(f"*-{args.label}") if (p / "rollouts.jsonl").is_file()
    )
    if not runs:
        print(f"no runs matching *-{args.label} with telemetry", file=sys.stderr)
        return 2
    run_dir = runs[-1]
    rows = [
        json.loads(line)
        for line in (run_dir / "rollouts.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    dead, total = zero_variance_fraction(rows)
    if total < args.min_groups:
        print(
            f"only {total} complete groups (< {args.min_groups}) in {run_dir.name} — no verdict",
            file=sys.stderr,
        )
        return 2

    fraction = dead / total
    verdict = "PASS" if fraction < args.threshold else "FAIL"
    print(
        f"{run_dir.name}: {dead}/{total} zero-variance groups = {fraction:.0%} "
        f"(threshold {args.threshold:.0%}) -> {verdict}"
    )
    if verdict == "FAIL":
        print("the question set is still mis-calibrated — fix the filter, not the trainer.")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
