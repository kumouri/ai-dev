"""The go/no-go gate between a probe run and a full run.

r4 spent 56% of its GPU time on zero-variance groups. A ~20-step probe measures that fraction on
the *current* question set with the *current* policy; the full run ships only if it clears the
threshold. The probe's checkpoints resume into the full run, so a passed gate costs nothing.

The fraction counts QUESTION signal only. A zero-variance group in which at most one of the k
rollouts even parsed is a *parse-storm* — the policy failing to emit a valid workflow k times —
and says nothing about the question. At probe time the policy parse-fails about half its rollouts
(48% and 52% measured, 2026-08-01 and 2026-08-20), so iid chance alone manufactures ~24%
zero-at-zero groups on a perfectly calibrated set; counting those as "mis-calibrated set" is how
two sound gates FAILed in a row on the OR-math set. Storm groups are excluded from the fraction
but always reported beside it, never hidden. Unanimous-CORRECT groups are parsed by construction,
so the r4 disease — too-easy questions — stays fully counted.

Exit 0 = ship it. Exit 1 = the set is mis-calibrated; fix the filter, not the trainer.

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


def zero_variance_fraction(rollout_rows: list[dict]) -> tuple[int, int, int]:
    """(question-dead groups, parse-storm groups, complete groups) over rollout telemetry.

    A group is *question-dead* when all k scores are identical AND at least two of its rollouts
    parsed — the policy genuinely engaged the question and it still bought no gradient. A
    zero-variance group with <=1 parsed is a *parse-storm*: policy noise, counted separately.
    Rows without a ``parsed`` field (pre-2026-08 telemetry) count as parsed, which reproduces
    the old accounting exactly.
    """
    by_question: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rollout_rows:
        by_question[row["question_id"]].append(row)
    dead = storms = total = 0
    for rows in by_question.values():
        group = rows[:4]
        if len(group) < 4:
            continue
        total += 1
        if len({row["score"] for row in group}) > 1:
            continue
        parsed = sum(1 for row in group if row.get("parsed", True))
        if parsed >= 2:
            dead += 1
        else:
            storms += 1
    return dead, storms, total


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
    dead, storms, total = zero_variance_fraction(rows)
    if total < args.min_groups:
        print(
            f"only {total} complete groups (< {args.min_groups}) in {run_dir.name} — no verdict",
            file=sys.stderr,
        )
        return 2

    fraction = dead / total
    verdict = "PASS" if fraction < args.threshold else "FAIL"
    print(
        f"{run_dir.name}: {dead}/{total} question-dead zero-variance groups = {fraction:.0%} "
        f"(threshold {args.threshold:.0%}; {storms} parse-storm groups reported as policy noise) "
        f"-> {verdict}"
    )
    if verdict == "FAIL":
        print("the question set is still mis-calibrated — fix the filter, not the trainer.")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
