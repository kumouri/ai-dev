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

Flat-rate remote calls; ~2 probes/question. Expect roughly half an hour for 800 questions at
4 concurrency units.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from coryphaeus.config import settings
from coryphaeus.datasets.loaders import Question, load_gsm8k
from coryphaeus.orchestrate import run_solo
from coryphaeus.pools import build_remote_registry
from coryphaeus.reward import score_answer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=800, help="GSM8K train candidates to probe")
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
    return parser.parse_args(argv)


async def probe(args: argparse.Namespace) -> int:
    cfg = settings()
    if not cfg.has_featherless:
        print("FEATHERLESS_API_KEY is not set — the probes are remote.", file=sys.stderr)
        return 2

    registry = build_remote_registry(max_units=2)
    for name in (args.weak, args.strong):
        if name not in registry:
            print(
                f"{name!r} is not in the pinned pool ({', '.join(registry.names())})",
                file=sys.stderr,
            )
            return 2

    questions = load_gsm8k(split="train", limit=args.limit)
    governor = registry.governor()
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    done = 0

    async def one(question: Question) -> dict:
        nonlocal done
        async with semaphore:
            outcomes: dict[str, bool] = {}
            for name in (args.weak, args.strong):
                execution = await run_solo(
                    registry.get(name), question.text, governor=governor, max_tokens=args.max_tokens
                )
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

    rows = await asyncio.gather(*(one(q) for q in questions))

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
    raise SystemExit(asyncio.run(probe(parse_args())))
