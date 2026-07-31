"""Ping every worker in a pool once and report what came back.

Run this before any experiment. It catches the boring failures — a model that is not pulled, a base
URL pointing nowhere, a missing key, a wrong model id — while they are still one line of output
rather than a run that dies at question 40.

    uv run python coryphaeus/scripts/smoke_workers.py --local
    uv run python coryphaeus/scripts/smoke_workers.py --featherless
    uv run python coryphaeus/scripts/smoke_workers.py --openrouter  # pennies; prints the receipt
    uv run python coryphaeus/scripts/smoke_workers.py --fake      # no network at all

For OpenRouter seats each row also shows WHO actually served the call and what it cost — the pin
is enforced request-side, but the smoke is where a wrong pin, a quantization surprise, or a
think-leak shows up as one line instead of a poisoned calibration.
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
    parser.add_argument(
        "--featherless", action="store_true", help="probe the manifest's featherless seats"
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="probe the manifest's openrouter seats (paid per token — a probe costs ~$0.001)",
    )
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
    remote_providers = tuple(
        name
        for name, wanted in (("featherless", args.featherless), ("openrouter", args.openrouter))
        if wanted
    )
    if remote_providers:
        for worker in build_remote_registry(models, providers=remote_providers):
            registry.add(worker)
    return registry


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (args.local or args.featherless or args.openrouter or args.fake):
        print(
            "pick at least one pool: --local, --featherless, --openrouter, or --fake",
            file=sys.stderr,
        )
        return 2

    cfg = settings()
    if args.local:
        print(f"ollama base url: {cfg.ollama_base_url}  (unit budget {cfg.ollama_unit_budget})")
    if args.featherless:
        if not cfg.has_featherless:
            print(
                "FEATHERLESS_API_KEY is not set — see .env.example. Skipping those seats.",
                file=sys.stderr,
            )
            args.featherless = False
        else:
            print(
                f"featherless base url: {cfg.featherless_base_url}  "
                f"(unit budget {cfg.featherless_unit_budget})"
            )
    if args.openrouter:
        if not cfg.has_openrouter:
            print(
                "OPENROUTER_API_KEY is not set — see .env.example. Skipping those seats.",
                file=sys.stderr,
            )
            args.openrouter = False
        else:
            print(
                f"openrouter base url: {cfg.openrouter_base_url}  "
                f"(unit budget {cfg.openrouter_unit_budget})"
            )

    registry = build(args)
    if not len(registry):
        print("no workers configured for the selected pool(s).", file=sys.stderr)
        if args.featherless or args.openrouter:
            print(
                "the pinned manifest (src/coryphaeus/manifests/worker_pool.json) has no seats for "
                "the selected provider(s) — pin some, or check the provider filter.",
                file=sys.stderr,
            )
        if args.models:
            print(f"--models {args.models!r} may not match any worker name.", file=sys.stderr)
        return 2

    governor = registry.governor()
    print(f"\nprobing {len(registry)} worker(s): {PROBE}\n")
    print(f"{'worker':20} {'units':6} {'ok':4} {'latency':9} {'tokens':13} answer / error")
    print("-" * 98)

    failures = 0
    total_cost = 0.0
    # Sequential on purpose: a smoke test should isolate which worker is broken, not race them.
    for worker in registry:
        result = await governor.invoke(worker, PROBE, max_tokens=args.max_tokens, temperature=0.0)
        answer = extract_answer(result.text) if result.ok else None
        total_cost += result.cost_usd
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
        if worker.spec.provider == "openrouter":
            # The receipt: who actually served it (None = provenance absent, pin still enforced
            # request-side) and what the call cost. A think-leak here means the reasoning knob is
            # not landing on this endpoint — surface it before it eats a calibration's budgets.
            served = result.meta.get("served_by") or "provenance-absent"
            leak = "  THINK-LEAK" if result.meta.get("think_leak_stripped") else ""
            detail = f"{str(detail)[:34]}  [{served} ${result.cost_usd:.5f}{leak}]"
        print(
            f"{worker.spec.name:20} {worker.spec.units:<6} {'yes' if result.ok else 'NO':4} "
            f"{result.latency_s:>7.2f}s  {result.tokens_in:>5}/{result.tokens_out:<6} "
            f"{str(detail)[:64]}"
        )

    print(f"\n{len(registry) - failures}/{len(registry)} usable.")
    if total_cost:
        print(f"probe spend: ${total_cost:.5f} (per-token seats only)")
    if governor.busy_events:
        print(
            f"{governor.busy_events} transient failure(s) backed off and retried "
            "(rate limit, capacity, or a failed generation) — not counted against the workers."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
