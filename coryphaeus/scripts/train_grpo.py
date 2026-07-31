"""Train the conductor with GRPO.

Workers live on the **remote** pool: the policy needs the GPU, and a resident local pool would
contend with it for the same 24 GB. Cost-1 and cost-2 workers only by default, so several rollouts
genuinely run at once — a single high-cost worker consumes the whole concurrency budget and
serializes every other rollout in the batch.

    # prove the loop: small policy, tiny slice, a handful of steps
    uv run python coryphaeus/scripts/train_grpo.py --smoke

    # the real run
    uv run python coryphaeus/scripts/train_grpo.py --limit 1000 --k 4

Run it inside the Linux environment (see docs/ROADMAP.md phase 2) — the train extra is Linux-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coryphaeus.config import settings
from coryphaeus.datasets.loaders import load_gsm8k, load_jsonl
from coryphaeus.pools import build_remote_registry, sample_pool_split
from coryphaeus.telemetry import RunWriter, step_rows
from coryphaeus.train.bridge import RewardBridge, make_reward_fn
from coryphaeus.train.dataset import build_rows, describe, to_hf_dataset
from coryphaeus.train.grpo import (
    DEFAULT_POLICY,
    SMOKE_POLICY,
    TrainSettings,
    build_trainer,
    describe_environment,
)
from coryphaeus.workers.featherless import MissingApiKey as FeatherlessMissingKey
from coryphaeus.workers.openrouter import MissingApiKey as OpenRouterMissingKey


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="", help=f"policy (default {DEFAULT_POLICY})")
    parser.add_argument("--smoke", action="store_true", help=f"tiny run on {SMOKE_POLICY}")
    parser.add_argument("--limit", type=int, default=1000, help="GSM8K train questions")
    parser.add_argument("--k", type=int, default=4, help="rollouts per question (num_generations)")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--max-units", type=int, default=2, help="drop workers above this unit cost"
    )
    parser.add_argument("--pool-size", type=int, nargs=2, default=(2, 3), metavar=("MIN", "MAX"))
    parser.add_argument("--n-pools", type=int, default=6, help="distinct training compositions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="grpo")
    parser.add_argument("--dry-run", action="store_true", help="build everything, train nothing")
    parser.add_argument(
        "--questions-file",
        default="",
        help="JSONL of calibrated questions (scripts/calibrate_questions.py). Without it, raw "
        "GSM8K — where ~40%% of groups reached unanimity in r3 and taught nothing.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="",
        help="where checkpoints go. Point at native ext4 when training in WSL — 6-9 GB "
        "checkpoints over drvfs/9P are their own slow-motion incident.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default="",
        metavar="CHECKPOINT",
        help="resume from a checkpoint dir, or bare --resume for the newest one in "
        "--checkpoint-dir. A fault should cost one save interval, not the run.",
    )
    parser.add_argument(
        "--deadline-hours",
        type=float,
        default=0.0,
        help="wall-clock budget; past it the trainer saves and stops at the next step boundary. "
        "Step count is the ambition, the deadline is the promise. 0 = off.",
    )
    parser.add_argument(
        "--providers",
        default="",
        help="comma-separated provider filter for the worker pool (featherless, openrouter). "
        "Default: every provider in the manifest. The launcher passes the providers whose keys "
        "it actually ships — the pool a box may use is exactly the pool it can authenticate to.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = settings()
    providers = tuple(p.strip() for p in args.providers.split(",") if p.strip()) or None

    env = describe_environment()
    print("environment:", json.dumps(env, indent=2))
    if not env.get("cuda_available"):
        print("\nCUDA is not available — refusing to start a training run on CPU.", file=sys.stderr)
        return 2

    try:
        registry = build_remote_registry(max_units=args.max_units, providers=providers)
    except (FeatherlessMissingKey, OpenRouterMissingKey) as exc:
        print(
            f"{exc}\nTraining routes workers remotely so the GPU is free for the policy — a "
            "local pool would contend for the same VRAM. If this box should not have that "
            "provider at all, pass --providers with the ones it has keys for.",
            file=sys.stderr,
        )
        return 2
    if len(registry) < 2:
        print(
            f"the remote pool has {len(registry)} worker(s) at or below {args.max_units} units — "
            "routing needs at least two to choose between. Re-pin with "
            "scripts/featherless_catalog.py.",
            file=sys.stderr,
        )
        return 2

    n_train_pools = min(args.n_pools, 6)
    split = sample_pool_split(
        registry,
        seed=args.seed,
        n_train=n_train_pools,
        n_eval=max(1, n_train_pools // 2),
        size=tuple(args.pool_size),
    )

    limit = 32 if args.smoke else args.limit
    if args.questions_file:
        questions = load_jsonl(args.questions_file, limit=limit, source="calibrated")
        print(f"calibrated questions: {len(questions)} from {args.questions_file}")
    else:
        questions = load_gsm8k(split="train", limit=limit)
    rows = build_rows(questions, registry, pools=split.train, seed=args.seed)

    print(f"\npool         : {', '.join(registry.names())}")
    print(f"dataset      : {json.dumps(describe(rows), indent=2)}")
    print(f"eval pools   : {[list(p) for p in split.evaluation]} (held out)")

    # Checkpoints live under their own subtree, not alongside experiment run records — otherwise
    # `report.py --latest` finds a checkpoint directory instead of the last run.
    checkpoint_root = (
        Path(args.checkpoint_dir) if args.checkpoint_dir else Path(cfg.runs_dir) / "checkpoints"
    )
    train_settings = TrainSettings(
        model=args.model or (SMOKE_POLICY if args.smoke else DEFAULT_POLICY),
        output_dir=checkpoint_root / f"{args.label}-{args.model or 'default'}".replace("/", "_"),
        num_generations=args.k,
        max_steps=5 if args.smoke else args.max_steps,
    )

    resume_from: str | None = None
    if args.resume:
        if args.resume != "latest":
            resume_from = args.resume
        else:
            candidates = sorted(
                train_settings.output_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
            )
            if not candidates:
                print(
                    f"--resume: no checkpoint-* under {train_settings.output_dir}", file=sys.stderr
                )
                return 2
            resume_from = str(candidates[-1])
        print(f"resuming from {resume_from}")

    # A dry run deliberately scores malformed completions, so its record would otherwise read as a
    # training run that got everything wrong. Name it for what it is.
    label = f"{args.label}-dryrun" if args.dry_run else args.label
    with RunWriter.create(label) as writer:
        run_id = writer.run_dir.name
        writer.write_meta(
            run_id=run_id,
            status="running",
            kind="grpo",
            environment=env,
            model=train_settings.model,
            pool=list(registry.names()),
            train_pools=[list(p) for p in split.train],
            eval_pools=[list(p) for p in split.evaluation],
            dataset=describe(rows),
            num_generations=args.k,
            seed=args.seed,
        )

        def record(records) -> None:
            """Training rollouts feed the same telemetry schema as evaluation runs.

            That is what makes the world-model dataset grow during training for free, rather than
            needing its own collection pass later.
            """
            for r in records:
                writer.write_rollout(r.to_row(run_id=run_id))
                if r.execution is not None:
                    for row in step_rows(
                        run_id=run_id,
                        question_id=r.question_id,
                        arm="train",
                        rollout_index=r.rollout_index,
                        execution=r.execution,
                    ):
                        writer.write_step(row)

        governor = registry.governor()
        with RewardBridge(registry, governor, max_tokens=1024) as bridge:
            reward_fn = make_reward_fn(bridge, on_records=record)

            if args.dry_run:
                print("\n--- dry run: everything except trainer.train() ---")
                sample = rows[: args.k]
                scores = reward_fn(
                    completions=["no plan here"] * len(sample),
                    question=[r.question for r in sample],
                    gold=[r.gold for r in sample],
                    question_id=[r.question_id for r in sample],
                    pool=[list(r.pool) for r in sample],
                )
                print(f"scores for deliberately-malformed completions: {scores}")
                print("(all zero is correct — an unparseable workflow earns nothing)")
                # Build the trainer too. This is what catches a TRL API change: constructing the
                # config and trainer is where a renamed parameter surfaces, and finding that here
                # beats finding it an hour into a real run.
                _trainer, fit = build_trainer(train_settings, to_hf_dataset(rows), reward_fn)
                print(f"\nGRPOTrainer constructed for {train_settings.model}")
                print(f"config settings not supported by this TRL: {list(fit.dropped) or 'none'}")
                return 0

            trainer, fit = build_trainer(train_settings, to_hf_dataset(rows), reward_fn)
            if fit.dropped:
                print(f"\nNOT applied by this TRL build: {', '.join(fit.dropped)}")
            if args.deadline_hours > 0:
                from coryphaeus.train.grpo import make_deadline_callback

                trainer.add_callback(make_deadline_callback(args.deadline_hours))
                print(f"deadline: {args.deadline_hours:.1f}h — will stop at a checkpoint")
            print(f"\ntraining {train_settings.model} — output {train_settings.output_dir}\n")
            trainer.train(resume_from_checkpoint=resume_from)
            trainer.save_model(str(train_settings.output_dir))

        writer.write_meta(
            run_id=run_id,
            status="finished",
            kind="grpo",
            environment=env,
            model=train_settings.model,
            pool=list(registry.names()),
            batches_scored=bridge.batches,
            rollouts_scored=bridge.rollouts,
            busy_events=governor.busy_events,
        )
        print(f"\nrun dir: {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
