"""Filter training questions to the ones where routing actually matters.

In r3, 40% of optimizer steps had zero reward variance — all k rollouts scored identically, so the
GRPO advantage was zero and the step taught nothing. The cause is question difficulty, not the
policy: a question every worker solves (or none solves) has no routing decision in it, and the
group reaches unanimity no matter what the conductor does.

So, the same lesson phase 0's oracle ceiling taught for evaluation, applied to training data:
**probe each candidate with a weak worker and a strong worker, and keep the disagreements** — the
questions where who-you-ask changes the outcome. Those are exactly where routing has gradient.

The probe results are kept per-worker in the output, because (worker, question, outcome) rows are
world-model training data — this pass pre-builds the phase-5 dataset while it filters.

    uv run python coryphaeus/scripts/calibrate_questions.py --limit 800 \
        --out coryphaeus/runs/calibrated-gsm8k.jsonl

    # calibrate the OpenRouter seats only — the served-system comparison arm
    uv run python coryphaeus/scripts/calibrate_questions.py --providers openrouter \
        --weak or-llama31-8b --strong or-qwen3-32b --out coryphaeus/runs/calibrated-or.jsonl

~2 probes/question. Featherless seats are flat-rate; OpenRouter seats are per-token, so paid
probes run inside a token-ledger reservation (worst case reserved up front, actual settled after
— see coryphaeus/spend.py). Expect roughly half an hour for 800 questions at 4 concurrency units.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from coryphaeus.cloud.budget import BudgetExceeded
from coryphaeus.datasets.loaders import Question, load_math_train, load_questions
from coryphaeus.orchestrate import run_solo
from coryphaeus.pools import build_remote_registry
from coryphaeus.reward import score_answer
from coryphaeus.spend import reserve_worst_case
from coryphaeus.workers import WorkerRegistry
from coryphaeus.workers.featherless import MissingApiKey as FeatherlessMissingKey
from coryphaeus.workers.openrouter import MissingApiKey as OpenRouterMissingKey


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=800, help="candidate questions to probe")
    parser.add_argument(
        "--dataset",
        default="gsm8k",
        help="question source: gsm8k, math-train, or a JSONL path (see datasets/loaders.py)",
    )
    parser.add_argument(
        "--levels",
        default="",
        help="math-train only: difficulty levels to draw from, e.g. '3,4,5'. GSM8K taught us the "
        "contested middle is the scarce resource; MATH's level field lets the probe start there.",
    )
    parser.add_argument("--weak", default="qwen25-14b", help="the pool's weakest worker")
    parser.add_argument("--strong", default="qwen25-32b", help="a strong worker")
    parser.add_argument("--out", default="coryphaeus/runs/calibrated-gsm8k.jsonl")
    parser.add_argument("--concurrency", type=int, default=4, help="questions in flight")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--keep-hard",
        type=float,
        default=0.15,
        help="fraction of both-wrong questions to keep anyway (headroom for a smarter router "
        "than two probes can see; 0 disables)",
    )
    parser.add_argument(
        "--refine",
        default="",
        metavar="V1_FILE",
        help="v2 mode: refine an existing calibrated file instead of probing GSM8K fresh. "
        "Uses training telemetry (--evidence-run) as ground truth where it exists, and "
        "multi-sample probes (--samples) where it does not. See docs/r4-filter-analysis.md — "
        "the v1 single-sample probe misclassified 22/22 of the too-easy questions.",
    )
    parser.add_argument(
        "--evidence-run",
        default="",
        metavar="RUN_DIR",
        help="training run dir whose rollouts.jsonl provides measured group outcomes",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=6,
        help="probes per unattempted question in --refine mode; 6 de-quantizes what 1 could not",
    )
    # Default keeps 1-4 successes of 6: the hard-contested shell (1/6) stays IN, the near-easy
    # shell (5/6) stays OUT — r4's dead steps were the too-easy side. The margins matter: the v2
    # run shipped with (0.17, 0.83), which silently excluded BOTH boundary shells because
    # 1/6 = 0.1667 < 0.17 and 5/6 = 0.8333 > 0.83 — a fencepost the docstring contradicted.
    parser.add_argument("--band", type=float, nargs=2, default=(0.15, 0.70), metavar=("LO", "HI"))
    parser.add_argument(
        "--providers",
        default="",
        help="comma-separated provider filter (featherless, openrouter). Default: every provider "
        "in the manifest — calibrate the pool the run will actually use; filter for "
        "single-provider comparison arms.",
    )
    return parser.parse_args(argv)


def build_probe_registry(args: argparse.Namespace) -> WorkerRegistry | None:
    """The probe pool, or None (with the reason printed) when a needed key is absent."""
    providers = tuple(p.strip() for p in args.providers.split(",") if p.strip()) or None
    try:
        return build_remote_registry(max_units=2, providers=providers)
    except (FeatherlessMissingKey, OpenRouterMissingKey) as exc:
        print(f"{exc} (the probes are remote)", file=sys.stderr)
        return None


async def refine(args: argparse.Namespace) -> int:
    """v2: keep measured-mixed, drop measured-unanimous, multi-sample the rest into a band."""

    v1 = [json.loads(line) for line in Path(args.refine).open(encoding="utf-8") if line.strip()]

    measured_mixed: set[str] = set()
    measured_unanimous: set[str] = set()
    if args.evidence_run:
        rollout_path = Path(args.evidence_run) / "rollouts.jsonl"
        by_question: dict[str, list[float]] = {}
        for line in rollout_path.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                by_question.setdefault(row["question_id"], []).append(row["score"])
        for qid, scores in by_question.items():
            if len(scores) >= 4:
                (measured_mixed if len(set(scores[:4])) > 1 else measured_unanimous).add(qid)
        print(
            f"evidence from {args.evidence_run}: {len(measured_mixed)} measured-mixed kept, "
            f"{len(measured_unanimous)} measured-unanimous dropped"
        )

    todo = [r for r in v1 if r["id"] not in measured_mixed and r["id"] not in measured_unanimous]
    print(f"re-probing {len(todo)} unattempted questions with n={args.samples} on {args.weak}")

    registry = build_probe_registry(args)
    if registry is None:
        return 2
    if args.weak not in registry:
        print(
            f"--weak {args.weak!r} is not in the probe pool ({', '.join(registry.names())})",
            file=sys.stderr,
        )
        return 2
    try:
        ledger, reservation_id = reserve_worst_case(
            registry, {args.weak: len(todo) * args.samples}, args.max_tokens, Path(args.out).stem
        )
    except BudgetExceeded as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    governor = registry.governor()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    done = 0
    spent = 0.0

    async def pass_rate(row: dict) -> tuple[dict, float]:
        nonlocal done, spent
        async with semaphore:
            question = Question(id=row["id"], text=row["question"], gold=row["answer"])
            hits = 0
            for _ in range(args.samples):
                execution = await run_solo(
                    registry.get(args.weak),
                    question.text,
                    governor=governor,
                    max_tokens=args.max_tokens,
                )
                spent += execution.cost_usd
                if score_answer(execution.final_text, question.gold).correct:
                    hits += 1
            done += 1
            if done % 20 == 0:
                print(f"  re-probed {done}/{len(todo)}", flush=True)
            return row, hits / args.samples

    try:
        results = await asyncio.gather(*(pass_rate(r) for r in todo))
    finally:
        if ledger is not None and reservation_id is not None:
            ledger.settle(reservation_id, spent, note="summed probe cost_usd")
            print(f"token ledger: settled {reservation_id} at ${spent:.4f}")
    lo, hi = args.band
    banded = [row for row, p in results if lo <= p <= hi]

    kept_rows = [dict(r, evidence="measured-mixed") for r in v1 if r["id"] in measured_mixed]
    kept_rows += [
        dict(row, evidence="banded", weak_pass_rate=p) for row, p in results if lo <= p <= hi
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in kept_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Persist EVERY probed rate, kept or not — refilters must never cost a re-probe again. (The v2
    # pass discarded the rates of excluded questions, so correcting its band means re-paying ~55
    # minutes of probes. Once was enough.)
    rates_path = out.with_suffix(".rates.jsonl")
    with rates_path.open("w", encoding="utf-8") as fh:
        for row, p in results:
            fh.write(json.dumps({"id": row["id"], "weak_pass_rate": p}) + "\n")

    rates = sorted(p for _, p in results)
    print(f"\npass-rate distribution over re-probed (n={len(rates)}; all rates -> {rates_path}):")
    # Bin by success COUNT: with n samples the reachable rates are exactly k/n, and float-range
    # bins leave holes at those exact values — the v2 histogram summed to 103 of 153 this way.
    for k in range(args.samples + 1):
        count = sum(1 for p in rates if abs(p - k / args.samples) < 1e-9)
        print(f"  {k}/{args.samples}: {'#' * count} {count}")
    print(
        f"\nkept {len(kept_rows)} = {len(measured_mixed)} measured-mixed + {len(banded)} banded "
        f"[{lo:.2f},{hi:.2f}]; wrote {out}"
    )
    print(f"transient backoffs: {governor.busy_events}")
    return 0


async def probe(args: argparse.Namespace) -> int:
    registry = build_probe_registry(args)
    if registry is None:
        return 2
    for name in (args.weak, args.strong):
        if name not in registry:
            print(
                f"{name!r} is not in the probe pool ({', '.join(registry.names())})",
                file=sys.stderr,
            )
            return 2

    if args.dataset.strip().lower() in {"math-train", "math_train"}:
        levels = tuple(int(x) for x in args.levels.split(",") if x.strip()) if args.levels else ()
        questions = load_math_train(limit=args.limit, levels=levels)
    else:
        questions = load_questions(args.dataset, limit=args.limit)
        if args.levels:
            print("--levels only applies to math-train; ignoring", file=sys.stderr)
    try:
        ledger, reservation_id = reserve_worst_case(
            registry,
            {args.weak: len(questions), args.strong: len(questions)},
            args.max_tokens,
            Path(args.out).stem,
        )
    except BudgetExceeded as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3
    governor = registry.governor()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    done = 0
    spent = 0.0

    async def one(question: Question) -> dict:
        nonlocal done, spent
        async with semaphore:
            outcomes: dict[str, bool] = {}
            for name in (args.weak, args.strong):
                execution = await run_solo(
                    registry.get(name), question.text, governor=governor, max_tokens=args.max_tokens
                )
                spent += execution.cost_usd
                score = score_answer(execution.final_text, question.gold)
                outcomes[name] = bool(score.correct)
            done += 1
            if done % 25 == 0:
                print(f"  probed {done}/{len(questions)}", flush=True)
            return {
                "id": question.id,
                "question": question.text,
                "answer": question.gold,
                "probes": outcomes,
            }

    try:
        rows = await asyncio.gather(*(one(q) for q in questions))
    finally:
        if ledger is not None and reservation_id is not None:
            ledger.settle(reservation_id, spent, note="summed probe cost_usd")
            print(f"token ledger: settled {reservation_id} at ${spent:.4f}")

    # Classify. Two probes give a coarse but honest signal:
    #   disagree        -> routing changes the outcome here: KEEP (the signal we exist to find)
    #   both correct    -> any routing wins; a unanimous all-1.0 group in waiting: DROP
    #   both wrong      -> probably unroutable, but two probes cannot see a third worker's win,
    #                      so keep a small deterministic fraction as headroom.
    disagree = [r for r in rows if len(set(r["probes"].values())) == 2]
    both_right = [r for r in rows if all(r["probes"].values())]
    both_wrong = [r for r in rows if not any(r["probes"].values())]
    hard_kept = both_wrong[:: max(1, round(1 / args.keep_hard))] if args.keep_hard > 0 else []

    kept = disagree + hard_kept
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    print(
        f"\nprobed {total}: disagree {len(disagree)} ({len(disagree) / total:.0%})  "
        f"both-correct {len(both_right)} ({len(both_right) / total:.0%})  "
        f"both-wrong {len(both_wrong)} ({len(both_wrong) / total:.0%})"
    )
    print(f"kept {len(kept)} ({len(disagree)} discriminative + {len(hard_kept)} hard)")
    print(f"wrote {out}")
    print(f"429/transient backoffs during probing: {governor.busy_events}")
    if len(kept) < 200:
        print(
            f"\nWARNING: {len(kept)} kept questions < 200 planned steps — raise --limit or "
            "accept repeats.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    _args = parse_args()
    raise SystemExit(asyncio.run(refine(_args) if _args.refine else probe(_args)))
