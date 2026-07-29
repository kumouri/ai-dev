"""The phase-0 experiment: does a prompted conductor beat the best single worker?

Runs one solo arm per worker plus one conductor arm per requested conductor, over the same question
slice, writing full telemetry and printing the comparison.

    # offline, no network, no key — proves the loop end to end
    uv run python coryphaeus/scripts/run_baseline.py --fake --limit 12

    # local models on the bundled fixture
    uv run python coryphaeus/scripts/run_baseline.py --local-only --limit 5 --conductor q4

    # the real slice
    uv run python coryphaeus/scripts/run_baseline.py --local-only --dataset gsm8k --limit 100 \
        --conductor q4,q27
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from coryphaeus.datasets.loaders import Question, load_questions
from coryphaeus.orchestrate import run_solo
from coryphaeus.policy import PromptedConductor
from coryphaeus.pools import build_local_registry, build_remote_registry
from coryphaeus.reporting import render_parse_failures, render_table, render_verdict, summarize
from coryphaeus.reward import score_answer
from coryphaeus.rollout import RolloutRecord, rollout_group
from coryphaeus.telemetry import RunWriter, step_rows
from coryphaeus.workers import Governor, WorkerRegistry, make_fake_pool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", default="fixture", help="fixture | gsm8k | math500 | path.jsonl"
    )
    parser.add_argument("--limit", type=int, default=20, help="questions to run")
    parser.add_argument("--local-only", action="store_true", help="local Ollama pool")
    parser.add_argument("--remote", action="store_true", help="include the Featherless pool")
    parser.add_argument("--fake", action="store_true", help="offline fake pool (no network)")
    parser.add_argument("--models", default="", help="limit the pool to these worker names")
    parser.add_argument(
        "--conductor",
        default="",
        help="comma-separated worker names to use as conductors (empty = solo arms only)",
    )
    parser.add_argument("--k", type=int, default=1, help="rollouts per question per conductor")
    parser.add_argument("--no-solo", action="store_true", help="skip the solo arms")
    parser.add_argument("--concurrency", type=int, default=2, help="questions in flight")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8, help="conductor sampling temp")
    parser.add_argument("--label", default="baseline", help="run directory suffix")
    return parser.parse_args(argv)


def build_registry(args: argparse.Namespace, questions: list[Question]) -> WorkerRegistry:
    models = tuple(m.strip() for m in args.models.split(",") if m.strip()) or None
    if args.fake:
        return WorkerRegistry(make_fake_pool({q.text: q.gold for q in questions}))
    registry = WorkerRegistry()
    if args.local_only or not args.remote:
        for worker in build_local_registry(models):
            registry.add(worker)
    if args.remote:
        for worker in build_remote_registry(models):
            registry.add(worker)
    return registry


async def solo_arm(
    worker_name: str,
    question: Question,
    registry: WorkerRegistry,
    governor: Governor,
    *,
    max_tokens: int,
) -> tuple[str, RolloutRecord]:
    worker = registry.get(worker_name)
    execution = await run_solo(worker, question.text, governor=governor, max_tokens=max_tokens)
    score = score_answer(execution.final_text, question.gold)
    record = RolloutRecord(
        question_id=question.id,
        rollout_index=0,
        arm=f"solo:{worker_name}",
        raw="",
        score=score,
        execution=execution,
    )
    return record.arm, record


async def run(args: argparse.Namespace) -> int:
    questions = load_questions(args.dataset, limit=args.limit)
    if not questions:
        print("no questions loaded", file=sys.stderr)
        return 2

    registry = build_registry(args, questions)
    if not len(registry):
        print("no workers configured — try --fake, --local-only, or --remote", file=sys.stderr)
        return 2

    conductor_names = tuple(c.strip() for c in args.conductor.split(",") if c.strip())
    for name in conductor_names:
        if name not in registry:
            print(
                f"conductor {name!r} is not in the pool ({', '.join(registry.names())})",
                file=sys.stderr,
            )
            return 2
    if args.no_solo and not conductor_names:
        print("nothing to run: --no-solo with no --conductor", file=sys.stderr)
        return 2

    governor = registry.governor()
    conductors = {
        name: PromptedConductor(
            registry.get(name), governor, temperature=args.temperature, max_tokens=768
        )
        for name in conductor_names
    }

    print(f"dataset      : {args.dataset} ({len(questions)} questions)")
    print(f"pool         : {', '.join(registry.names())}")
    print(f"conductors   : {', '.join(conductor_names) or '(none)'}  k={args.k}")
    print(f"solo arms    : {'skipped' if args.no_solo else 'yes'}")
    print(f"catalog:\n{registry.catalog_text()}\n")

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    started = time.perf_counter()
    rollout_rows: list[dict] = []

    with RunWriter.create(args.label) as writer:
        run_dir = writer.run_dir
        run_id = run_dir.name
        # Meta first, so an interrupted run still says what it was trying to do.
        writer.write_meta(
            run_id=run_id,
            status="running",
            dataset=args.dataset,
            n_questions=len(questions),
            pool=list(registry.names()),
            catalog=registry.catalog_text(),
            conductors={name: p.describe() for name, p in conductors.items()},
            k=args.k,
            solo_arms=not args.no_solo,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
        )
        print(f"run dir   : {run_dir}\n")

        def record_now(record: RolloutRecord) -> None:
            """Persist a rollout the moment it finishes.

            Buffering to the end would mean a run that dies at question 95 leaves nothing at all.
            These runs are long enough that partial results are worth more than tidy code.
            """
            row = record.to_row(run_id=run_id)
            rollout_rows.append(row)
            writer.write_rollout(row)
            if record.execution is not None:
                for step_row in step_rows(
                    run_id=run_id,
                    question_id=record.question_id,
                    arm=record.arm,
                    rollout_index=record.rollout_index,
                    execution=record.execution,
                ):
                    writer.write_step(step_row)

        # Sweep arm-by-arm, not question-by-question. A local pool's total footprint usually exceeds
        # VRAM, so round-robining models per question makes Ollama evict and reload on nearly every
        # call — minutes of load time per question. Finishing one worker's whole slice before moving
        # on keeps it resident: nothing about the science changes, everything about the clock does.
        async def solo_sweep(name: str) -> None:
            done = 0

            async def one(question: Question) -> None:
                nonlocal done
                async with semaphore:
                    _arm, record = await solo_arm(
                        name, question, registry, governor, max_tokens=args.max_tokens
                    )
                    record_now(record)
                    done += 1
                    hits = sum(
                        1 for r in rollout_rows if r["arm"] == f"solo:{name}" and r["correct"]
                    )
                    print(f"  solo:{name} [{done}/{len(questions)}] correct={hits}", flush=True)

            await asyncio.gather(*(one(q) for q in questions))

        async def conductor_sweep(name: str, policy: PromptedConductor) -> None:
            done = 0
            arm = f"conductor:{name}"

            async def one(question: Question) -> None:
                nonlocal done
                async with semaphore:
                    group = await rollout_group(
                        policy,
                        registry,
                        question,
                        governor=governor,
                        k=args.k,
                        arm=arm,
                        self_worker=policy.worker,
                        max_tokens=args.max_tokens,
                    )
                    for record in group:
                        record_now(record)
                    done += 1
                    rows = [r for r in rollout_rows if r["arm"] == arm]
                    hits = sum(1 for r in rows if r["correct"])
                    bad = sum(1 for r in rows if r["parse_reason"])
                    print(
                        f"  {arm} [{done}/{len(questions)}] correct={hits} unparsed={bad}",
                        flush=True,
                    )

            await asyncio.gather(*(one(q) for q in questions))

        if not args.no_solo:
            for name in registry.names():
                await solo_sweep(name)
        for name, policy in conductors.items():
            await conductor_sweep(name, policy)

        elapsed = time.perf_counter() - started
        writer.write_meta(
            run_id=run_id,
            status="finished",
            dataset=args.dataset,
            n_questions=len(questions),
            pool=list(registry.names()),
            catalog=registry.catalog_text(),
            conductors={name: p.describe() for name, p in conductors.items()},
            k=args.k,
            solo_arms=not args.no_solo,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            concurrency=args.concurrency,
            elapsed_s=round(elapsed, 2),
            busy_events=governor.busy_events,
        )

    summaries = summarize(rollout_rows)
    print(f"\n{render_table(summaries)}\n")
    print(render_verdict(summaries))
    print("\nparse failures by reason:")
    print(render_parse_failures(rollout_rows))
    print(f"\nwall clock: {elapsed:.1f}s   429s backed off: {governor.busy_events}")
    print(f"run dir   : {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(parse_args())))
