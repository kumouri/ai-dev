"""Ping every worker in a pool once and report what came back.

Run this before any experiment. It catches the boring failures — a model that is not pulled, a base
URL pointing nowhere, a missing key, a wrong model id — while they are still one line of output
rather than a run that dies at question 40.

    uv run python coryphaeus/scripts/smoke_workers.py --local
    uv run python coryphaeus/scripts/smoke_workers.py --featherless
    uv run python coryphaeus/scripts/smoke_workers.py --fake      # no network at all
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from coryphaeus.config import settings
from coryphaeus.pools import build_local_registry, build_remote_registry
from coryphaeus.reward import extract_answer
from coryphaeus.workers import WorkerRegistry, make_fake_pool

PROBE = "What is 17 + 25? Reply with just the number in the form \\boxed{answer}."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--local", action="store_true", help="probe the local Ollama pool")
    parser.add_argument("--featherless", action="store_true", help="probe the remote pool")
    parser.add_argument("--fake", action="store_true", help="probe the offline fake pool")
    parser.add_argument("--models", default="", help="comma-separated worker names to limit to")
    # Generous on purpose: a tight budget on a reasoning model yields blank content, which reads as
    # a broken worker. See the note in workers/ollama.py.
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args(argv)


def build(args: argparse.Namespace) -> WorkerRegistry:
    models = tuple(m.strip() for m in args.models.split(",") if m.strip()) or None
    if args.fake:
        return WorkerRegistry(make_fake_pool({PROBE: "42"}))
    registry = WorkerRegistry()
    if args.local:
        for worker in build_local_registry(models):
            registry.add(worker)
    if args.featherless:
        for worker in build_remote_registry(models):
            registry.add(worker)
    return registry


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (args.local or args.featherless or args.fake):
        print("pick at least one pool: --local, --featherless, or --fake", file=sys.stderr)
        return 2

    cfg = settings()
    if args.local:
        print(f"ollama base url: {cfg.ollama_base_url}  (unit budget {cfg.ollama_unit_budget})")
    if args.featherless:
        if not cfg.has_featherless:
            print(
                "FEATHERLESS_API_KEY is not set — see .env.example. Skipping the remote pool.",
                file=sys.stderr,
            )
            args.featherless = False
        else:
            print(
                f"featherless base url: {cfg.featherless_base_url}  "
                f"(unit budget {cfg.featherless_unit_budget})"
            )

    registry = build(args)
    if not len(registry):
        print("no workers configured for the selected pool(s).", file=sys.stderr)
        if args.featherless:
            print(
                "REMOTE_POOL in coryphaeus/pools.py is empty by default: model ids differ per "
                "account, and a wrong id is a 404 discovered mid-run. Add entries from your "
                "provider's catalogue.",
                file=sys.stderr,
            )
        if args.models:
            print(f"--models {args.models!r} may not match any worker name.", file=sys.stderr)
        return 2

    governor = registry.governor()
    print(f"\nprobing {len(registry)} worker(s): {PROBE}\n")
    print(f"{'worker':14} {'units':6} {'ok':4} {'latency':9} {'tokens':13} answer / error")
    print("-" * 92)

    failures = 0
    # Sequential on purpose: a smoke test should isolate which worker is broken, not race them.
    for worker in registry:
        result = await governor.invoke(worker, PROBE, max_tokens=args.max_tokens, temperature=0.0)
        answer = extract_answer(result.text) if result.ok else None
        if not result.ok:
            failures += 1
            detail = result.error or "unknown error"
        elif answer is not None:
            detail = answer
        elif (
            result.meta.get("finish_reason") == "length"
            or result.meta.get("done_reason") == "length"
        ):
            # A verbose model cut off mid-explanation. Say so: a bare "None" reads as "refused to
            # answer", which sends you debugging the model instead of raising --max-tokens.
            detail = f"(truncated at {args.max_tokens} tokens — no answer yet; raise --max-tokens)"
            failures += 1
        else:
            detail = "(responded, but no answer could be extracted)"
            failures += 1
        print(
            f"{worker.spec.name:14} {worker.spec.units:<6} {'yes' if result.ok else 'NO':4} "
            f"{result.latency_s:>7.2f}s  {result.tokens_in:>5}/{result.tokens_out:<6} "
            f"{str(detail)[:56]}"
        )

    print(f"\n{len(registry) - failures}/{len(registry)} usable.")
    if governor.busy_events:
        print(
            f"{governor.busy_events} transient failure(s) backed off and retried "
            "(rate limit, capacity, or a failed generation) — not counted against the workers."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
