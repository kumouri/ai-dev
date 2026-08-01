"""Evaluate a trained (or base) conductor checkpoint in the same harness as every other arm.

Two phases, deliberately separate:

**A — GPU, local:** load the checkpoint, render the *same* conversational prompt training used
(``train.dataset.build_row``: ``CONDUCTOR_SYSTEM`` + ``include_self=False`` + the catalog), and
generate one emission per question. Generation is free and private; nothing remote happens here.

**B — network, ledger-bracketed:** replay each emission through the exact scoring path the
baselines used (``rollout.one_rollout``): parse → execute against the advertised pool → score.
Same parser, same governor, same reward — so the resulting number sits honestly in the same
table as the solo arms and the prompted conductors.

The catalog is the FULL selected pool — a composition the policy never saw during training
(training cycled 2–3-worker sub-pools), so a win here is a pool-generalization claim, not a
memorized menu.

    # the trained arm
    uv run python coryphaeus/scripts/eval_trained.py \
        --checkpoint /path/to/checkpoint --dataset math500 --limit 400 \
        --providers openrouter --label eval-or-trained

    # the untrained control: same weights family, same prompt, same pool
    uv run python coryphaeus/scripts/eval_trained.py \
        --checkpoint Qwen/Qwen2.5-1.5B-Instruct --arm base-1.5b \
        --dataset math500 --limit 400 --providers openrouter --label eval-or-base

Needs the train extra (torch + transformers) and a CUDA device for phase A; phase B needs the
selected providers' API keys. Run it where the GPU lives.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from coryphaeus.cloud.budget import BudgetExceeded
from coryphaeus.datasets.loaders import Question, load_questions
from coryphaeus.pools import build_remote_registry
from coryphaeus.reporting import render_parse_failures, render_table, summarize
from coryphaeus.rollout import one_rollout
from coryphaeus.spend import EST_TOKENS_IN, reserve_worst_case
from coryphaeus.telemetry import RunWriter, step_rows
from coryphaeus.train.dataset import build_row
from coryphaeus.workers import WorkerRegistry
from coryphaeus.workers.featherless import MissingApiKey as FeatherlessMissingKey
from coryphaeus.workers.openrouter import MissingApiKey as OpenRouterMissingKey


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="checkpoint directory, or a HF model id for the untrained control arm",
    )
    parser.add_argument(
        "--arm",
        default="",
        help="arm label in the results (default: trained:<checkpoint dir name>)",
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        help="tokenizer source when the checkpoint dir ships without one (a TRL save sometimes "
        "does); the base model id is the honest choice — the tokenizer is identical by "
        "construction. Default: the checkpoint itself.",
    )
    parser.add_argument("--dataset", default="math500", help="fixture | gsm8k | math500 | path")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument(
        "--providers",
        default="",
        help="comma-separated worker-provider filter for the pool (featherless, openrouter)",
    )
    parser.add_argument("--max-units", type=int, default=2, help="drop workers above this cost")
    parser.add_argument("--temperature", type=float, default=0.8, help="matches the prompted arm")
    parser.add_argument("--gen-max-tokens", type=int, default=768, help="workflow budget, phase A")
    parser.add_argument("--gen-batch", type=int, default=8, help="questions per generate() call")
    parser.add_argument("--max-tokens", type=int, default=1024, help="worker budget, phase B")
    parser.add_argument("--concurrency", type=int, default=8, help="questions in flight, phase B")
    parser.add_argument("--label", default="eval-trained", help="run directory suffix")
    return parser.parse_args(argv)


def generate_emissions(
    args: argparse.Namespace, questions: list[Question], registry: WorkerRegistry
) -> list[str]:
    """Phase A: one emission per question, batched, on the local GPU.

    Imports live here so the offline world (tests, --help, machines without torch) never pays
    for them.
    """
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    print(f"loading {args.checkpoint} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.checkpoint)
    # Decoder-only batch generation pads on the LEFT; right padding would put the answer's first
    # tokens behind pad positions and shift every completion.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    pool = tuple(registry.names())
    rows = [build_row(q, registry, pool) for q in questions]
    prompts = [
        tokenizer.apply_chat_template(
            [dict(m) for m in row.messages], tokenize=False, add_generation_prompt=True
        )
        for row in rows
    ]

    emissions: list[str] = []
    started = time.perf_counter()
    for start in range(0, len(prompts), args.gen_batch):
        batch = prompts[start : start + args.gen_batch]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=args.gen_max_tokens,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
        # Slice off the prompt: with left padding every row's completion starts at the same
        # column, which is the whole reason for left padding.
        completions = output[:, encoded["input_ids"].shape[1] :]
        emissions.extend(tokenizer.batch_decode(completions, skip_special_tokens=True))
        done = min(start + args.gen_batch, len(prompts))
        rate = done / max(time.perf_counter() - started, 1e-9)
        print(f"  generated {done}/{len(prompts)} ({rate:.1f} q/s)", flush=True)

    del model
    torch.cuda.empty_cache()
    return emissions


async def execute_and_score(
    args: argparse.Namespace,
    arm: str,
    questions: list[Question],
    emissions: list[str],
    registry: WorkerRegistry,
) -> int:
    """Phase B: the baselines' exact scoring path, ledger-bracketed."""
    governor = registry.governor()
    priciest = max(
        registry.names(),
        key=lambda n: registry.get(n).spec.cost(EST_TOKENS_IN, args.max_tokens),
    )
    try:
        ledger, reservation_id = reserve_worst_case(
            registry,
            # Every workflow charged the max five steps at the priciest seat — the same
            # over-estimate run_baseline uses for conductor arms; the settle reports reality.
            {priciest: len(questions) * 5},
            args.max_tokens,
            f"eval-{args.label}",
        )
    except BudgetExceeded as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    rollout_rows: list[dict] = []
    done = 0

    try:
        with RunWriter.create(args.label) as writer:
            run_dir = writer.run_dir
            run_id = run_dir.name
            writer.write_meta(
                run_id=run_id,
                status="running",
                dataset=args.dataset,
                n_questions=len(questions),
                pool=list(registry.names()),
                catalog=registry.catalog_text(),
                conductors={
                    arm: {
                        "kind": "trained-checkpoint",
                        "checkpoint": str(args.checkpoint),
                        "temperature": args.temperature,
                        "gen_max_tokens": args.gen_max_tokens,
                    }
                },
                k=1,
                solo_arms=False,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                concurrency=args.concurrency,
            )
            print(f"run dir   : {run_dir}\n")

            async def one(question: Question, raw: str) -> None:
                nonlocal done
                async with semaphore:
                    record = await one_rollout(
                        raw,
                        question,
                        registry,
                        governor=governor,
                        rollout_index=0,
                        arm=arm,
                        self_worker=None,
                        max_tokens=args.max_tokens,
                    )
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
                    done += 1
                    if done % 25 == 0:
                        hits = sum(1 for r in rollout_rows if r["correct"])
                        bad = sum(1 for r in rollout_rows if r["parse_reason"])
                        print(
                            f"  {arm} [{done}/{len(questions)}] correct={hits} unparsed={bad}",
                            flush=True,
                        )

            await asyncio.gather(
                *(one(q, raw) for q, raw in zip(questions, emissions, strict=True))
            )
            writer.write_meta(run_id=run_id, status="finished", busy_events=governor.busy_events)
    finally:
        if ledger is not None and reservation_id is not None:
            spent = round(sum(float(r.get("cost_usd") or 0.0) for r in rollout_rows), 6)
            ledger.settle(reservation_id, spent, note="summed rollout cost_usd")
            print(f"token ledger: settled {reservation_id} at ${spent:.4f}")

    print(f"\n{render_table(summarize(rollout_rows))}")
    print("\nparse failures by reason:")
    print(render_parse_failures(rollout_rows))
    print(f"\ntransient backoffs: {governor.busy_events}")
    print(f"run dir   : {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    arm = args.arm or f"trained:{Path(args.checkpoint).name}"
    providers = tuple(p.strip() for p in args.providers.split(",") if p.strip()) or None

    try:
        registry = build_remote_registry(max_units=args.max_units, providers=providers)
    except (FeatherlessMissingKey, OpenRouterMissingKey) as exc:
        print(f"{exc} (phase B executes workflows remotely)", file=sys.stderr)
        return 2
    if len(registry) < 2:
        print("the pool has fewer than 2 workers — nothing to route between", file=sys.stderr)
        return 2

    questions = load_questions(args.dataset, limit=args.limit)
    if not questions:
        print("no questions loaded", file=sys.stderr)
        return 2

    print(f"arm       : {arm}")
    print(f"dataset   : {args.dataset} ({len(questions)} questions)")
    print(f"pool      : {', '.join(registry.names())}")
    print(f"catalog:\n{registry.catalog_text()}\n")

    emissions = generate_emissions(args, questions, registry)
    parseable_preview = sum(1 for e in emissions if "{" in e)
    print(f"\nphase A done: {len(emissions)} emissions ({parseable_preview} contain JSON-ish)\n")
    print(json.dumps({"sample_emission": emissions[0][:400]}, ensure_ascii=False))

    return asyncio.run(execute_and_score(args, arm, questions, emissions, registry))


if __name__ == "__main__":
    raise SystemExit(main())
