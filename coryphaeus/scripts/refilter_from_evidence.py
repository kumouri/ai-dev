"""Refilter a question set using measured training evidence only — zero probes, zero cost.

Every training run measures its questions for free: a complete group is either mixed (carries
gradient) or unanimous (dead weight). This folds all of that evidence back into the set:

  - measured mixed in ANY run       -> keep (empirically contested under the real policy)
  - measured ONLY unanimous         -> drop (proven dead at current ability)
  - never measured                  -> keep (unproven, not guilty)

Gate fails stop being an hour of re-probing and become a minutes-long fold of what the failed
probe just taught us. Each gate round strictly shrinks the dead weight.

    uv run python coryphaeus/scripts/refilter_from_evidence.py \
        --in coryphaeus/runs/calibrated-gsm8k-v2.jsonl \
        --evidence coryphaeus/runs/*grpo15b-r4 coryphaeus/runs/*grpo15b-r5 \
        --out coryphaeus/runs/calibrated-gsm8k-v3.jsonl
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--evidence", nargs="+", required=True, help="run dirs (globs ok)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=4, help="group size")
    parser.add_argument(
        "--recency",
        action="store_true",
        help="classify each question by its LATEST run's groups only. Under drift, old "
        "contestedness is history, not prediction: a question mixed at step 0 and unanimous at "
        "step 15 is a question the policy outgrew. The default (any-run-mixed wins) kept 3 of "
        "r5's 5 dead questions on the strength of stale r4 evidence — measured 2026-07-30.",
    )
    return parser.parse_args(argv)


def classify(
    evidence_dirs: list[str], k: int, *, recency: bool = False
) -> tuple[set[str], set[str]]:
    """(mixed, measured-but-only-unanimous) question ids across the evidence runs.

    ``recency=True`` classifies each question by the LATEST run that measured it — under policy
    drift, old contestedness is history, not prediction. Run order comes from the timestamped run
    directory names.
    """
    run_dirs: list[Path] = []
    for pattern in evidence_dirs:
        run_dirs += [Path(p) for p in glob.glob(pattern)]
    run_dirs = sorted(run_dirs, key=lambda p: p.name)  # timestamps sort chronologically

    #: qid -> list of (run_index, group)
    groups: dict[str, list[tuple[int, list[float]]]] = collections.defaultdict(list)
    for run_index, run in enumerate(run_dirs):
        path = run / "rollouts.jsonl"
        if not path.is_file():
            continue
        per_question: dict[str, list[float]] = collections.defaultdict(list)
        for line in path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                per_question[row["question_id"]].append(row["score"])
        for qid, s in per_question.items():
            for i in range(0, len(s) - k + 1, k):
                groups[qid].append((run_index, s[i : i + k]))

    mixed: set[str] = set()
    only_unanimous: set[str] = set()
    for qid, entries in groups.items():
        if recency:
            latest = max(run_index for run_index, _ in entries)
            considered = [g for run_index, g in entries if run_index == latest]
        else:
            considered = [g for _, g in entries]
        if any(len(set(g)) > 1 for g in considered):
            mixed.add(qid)
        else:
            only_unanimous.add(qid)
    return mixed, only_unanimous


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = [json.loads(line) for line in Path(args.infile).open(encoding="utf-8") if line.strip()]
    mixed, dead = classify(args.evidence, args.k, recency=args.recency)

    kept, dropped = [], []
    for row in rows:
        qid = row["id"]
        if qid in mixed:
            kept.append(dict(row, evidence="measured-mixed"))
        elif qid in dead:
            dropped.append(qid)
        else:
            kept.append(row)  # unmeasured: unproven, not guilty

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"in {len(rows)} -> kept {len(kept)} "
        f"({sum(1 for r in kept if r.get('evidence') == 'measured-mixed')} measured-mixed, "
        f"{len(kept) - sum(1 for r in kept if r.get('evidence') == 'measured-mixed')} unmeasured); "
        f"dropped {len(dropped)} measured-dead"
    )
    print(f"wrote {out}")
    if len(kept) < 30:
        print(
            f"WARNING: only {len(kept)} questions left — the corpus may be exhausted for this "
            "pool; consider MATH-tier sources.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
