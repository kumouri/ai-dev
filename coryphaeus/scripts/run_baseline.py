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

from coryphaeus.cloud.budget import BudgetExceeded
from coryphaeus.datasets.loaders import Question, load_questions
from coryphaeus.orchestrate import run_solo
from coryphaeus.policy import PromptedConductor
from coryphaeus.pools import build_local_registry, build_remote_registry
from coryphaeus.reporting import render_parse_failures, render_table, render_verdict, summarize
from coryphaeus.reward import score_answer
from coryphaeus.rollout import RolloutRecord, rollout_group
from coryphaeus.spend import EST_TOKENS_IN, reserve_worst_case
from coryphaeus.telemetry import RunWriter, step_rows
from coryphaeus.workers import Governor, WorkerRegistry, make_fake_pool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", default="fixture", help="fixture | gsm8k | math500 | path.jsonl"
    )
    parser.add_argument("--limit", type=int, default=20, help="questions to run")
    parser.add_argument("--local-only", action="store_true", help="local Ollama pool")
    parser.add_argument("--remote", action="store_true", help="include the pinned remote pool")
    parser.add_argument("--fake", action="store_true", help="offline fake pool (no network)")
    parser.add_argument("--models", default="", help="limit the pool to these worker names")
    parser.add_argument(
        "--providers",
        default="",
        help="with --remote: comma-separated provider filter (featherless, openrouter). "
        "Default: every provider in the manifest. Paid seats run inside a token-ledger "
        "reservation either way.",
    )
    parser.add_argument(
        "--conductor",
        default="",
        help="comma-separated worker names to use as conductors (empty = solo arms only)",
    )
    parser.add_argument("--k", type=int, default=1, help="rollouts per question per conductor")
    parser.add_argument("--no-solo", action="store_true", help="skip the solo arms")
    parser.add_argument("--concurrency", type=int, default=2, help="questions in flight")
    parser.add_argument(
        "--chunk",
        type=int,
        default=25,
        help="questions per block; every block boundary yields a complete cross-arm comparison "
        "(0 = one block, fewest model reloads but no interim result)",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8, help="conductor sampling temp")
    parser.add_argument("--label", default="baseline", help="run directory suffix")
    return parser.parse_args(argv)


def build_registry(args: argparse.Namespace, questions: list[Question]) -> WorkerRegistry:
    models = tuple(m.strip() for m in args.models.split(",") if m.strip()) or None
    providers = tuple(p.strip() for p in args.providers.split(",") if p.strip()) or None
    if args.fake:
        return WorkerRegistry(make_fake_pool({q.text: q.gold for q in questions}))
    registry = WorkerRegistry()
    if args.local_only or not args.remote:
        for worker in build_local_registry(models):
            registry.add(worker)
    if args.remote:
        for worker in build_remote_registry(models, providers=providers):
            registry.add(worker)
    return registry


def worst_case_calls(
    args: argparse.Namespace,
    registry: WorkerRegistry,
    conductor_names: tuple[str, ...],
    n_questions: int,
) -> dict[str, int]:
    """Upper-bound call counts per worker, for the token reservation.

    Solo arms are exact (one call per question per worker). Conductor arms cannot know their
    routing in advance, so every workflow call is charged to the *priciest* seat in the pool and
    every workflow is assumed to use the maximum five steps — a deliberate over-estimate; the
    settle reports what actually happened.
    """
    calls: dict[str, int] = {}
    if not args.no_solo:
        for name in registry.names():
            calls[name] = calls.get(name, 0) + n_questions
    if conductor_names:
        priciest = max(
            registry.names(),
            key=lambda n: registry.get(n).spec.cost(EST_TOKENS_IN, args.max_tokens),
        )
        for name in conductor_names:
            calls[name] = calls.get(name, 0) + n_questions * args.k
            calls[priciest] = calls.get(priciest, 0) + n_questions * args.k * 5
    return calls


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

    try:
        ledger, reservation_id = reserve_worst_case(
            registry,
            worst_case_calls(args, registry, conductor_names, len(questions)),
            args.max_tokens,
            f"baseline-{args.label}",
        )
    except BudgetExceeded as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3

    governor = registry.governor()

    print(f"dataset      : {args.dataset} ({len(questions)} questions)")
    print(f"pool         : {', '.join(registry.names())}")
    print(f"conductors   : {', '.join(conductor_names) or '(none)'}  k={args.k}")
    print(f"solo arms    : {'skipped' if args.no_solo else 'yes'}")
    print(f"catalog:\n{registry.catalog_text()}\n")

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    started = time.perf_counter()
    # Shared with _run_arms so the settle sees whatever finished, even on a crash mid-arm —
    # rows land here the moment each rollout completes.
    rollout_rows: list[dict] = []

    try:
        return await _run_arms(args, questions, registry, conductor_names, governor,
                               semaphore, started, rollout_rows)  # fmt: skip
    finally:
        if ledger is not None and reservation_id is not None:
            spent = round(sum(float(r.get("cost_usd") or 0.0) for r in rollout_rows), 6)
            ledger.settle(reservation_id, spent, note="summed rollout cost_usd")
            print(f"token ledger: settled {reservation_id} at ${spent:.4f}")


async def _run_arms(
    args: argparse.Namespace,
    questions: list[Question],
    registry: WorkerRegistry,
    conductor_names: tuple[str, ...],
    governor: Governor,
    semaphore: asyncio.Semaphore,
    started: float,
    rollout_rows: list[dict],
) -> int:
    conductors = {
        name: PromptedConductor(
            registry.get(name), governor, temperature=args.temperature, max_tokens=768
        )
        for name in conductor_names
    }

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

        # Sweep arm-by-arm *within a block of questions*, which resolves a real tension:
        #
        #   - Question-by-question round-robin thrashes a local pool. Total footprint usually
        #     exceeds VRAM, so Ollama evicts and reloads on nearly every call.
        #   - Whole-slice-per-arm avoids that, but a run interrupted halfway leaves ONE complete
        #     arm and nothing to compare it against, which is worth nothing at all.
        #
        # Blocking gives both: model loads are bounded to (arms x blocks), and every block boundary
        # has a complete comparison across all arms. Set --chunk 0 for one block per arm.
        async def solo_sweep(name: str, batch: list[Question]) -> None:
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
                    print(f"  solo:{name} [{done}/{len(batch)}] correct so far={hits}", flush=True)

            await asyncio.gather(*(one(q) for q in batch))

        async def conductor_sweep(
            name: str, policy: PromptedConductor, batch: list[Question]
        ) -> None:
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
                        f"  {arm} [{done}/{len(batch)}] correct so far={hits} unparsed={bad}",
                        flush=True,
                    )

            await asyncio.gather(*(one(q) for q in batch))

        size = args.chunk if args.chunk > 0 else len(questions)
        blocks = [questions[i : i + size] for i in range(0, len(questions), size)]
        for block_index, batch in enumerate(blocks, start=1):
            if len(blocks) > 1:
                print(f"\n--- block {block_index}/{len(blocks)} ({len(batch)} questions) ---")
            if not args.no_solo:
                for name in registry.names():
                    await solo_sweep(name, batch)
            for name, policy in conductors.items():
                await conductor_sweep(name, policy, batch)
            if len(blocks) > 1 and block_index < len(blocks):
                # An interim comparison at every boundary, so a run that has to be cut short has
                # already answered the question on however many questions it got through.
                print(f"\n{render_table(summarize(rollout_rows))}")
                print(f"{render_verdict(summarize(rollout_rows))}\n")

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
