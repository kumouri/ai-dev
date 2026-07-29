"""Inspect the provider catalogue and pin a worker pool from it.

The provider reports each model's real ``concurrency_cost``, which is the number the concurrency
governor needs. Guessing it from parameter count is wrong in the 24–32B band (it costs 2, not 4) and
over-reserving there halves effective concurrency, so the catalogue is the source of truth.

Run this **deliberately**, not at runtime: the catalogue is tens of thousands of models, and a
*pinned* manifest is what keeps runs comparable to each other.

    # look before you pin
    uv run python coryphaeus/scripts/featherless_catalog.py --list --max-cost 2 --grep Instruct
    uv run python coryphaeus/scripts/featherless_catalog.py --summary

    # write the manifest pools.py reads
    uv run python coryphaeus/scripts/featherless_catalog.py --pin \
        Qwen/Qwen2.5-Math-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

from coryphaeus.catalog import manifest_entry
from coryphaeus.config import settings
from coryphaeus.pools import MANIFEST_PATH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", action="store_true", help="cost + model_class distributions")
    parser.add_argument("--list", action="store_true", help="list matching models")
    parser.add_argument("--grep", default="", help="regex against the model id")
    parser.add_argument("--class-grep", default="", help="regex against model_class")
    parser.add_argument("--max-cost", type=int, default=0, help="only models at or below this cost")
    parser.add_argument("--top", type=int, default=40, help="cap on listed rows")
    parser.add_argument(
        "--pin",
        nargs="*",
        metavar="MODEL_ID",
        help="write these model ids to the manifest, with units taken from the catalogue",
    )
    parser.add_argument("--manifest", default=str(MANIFEST_PATH), help="manifest path to write")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


def fetch_catalog(timeout: float) -> list[dict]:
    """GET the provider's model list. Read-only; needs FEATHERLESS_API_KEY."""
    cfg = settings()
    if not cfg.has_featherless:
        raise SystemExit(
            "FEATHERLESS_API_KEY is not set in this process. Put it in .env, or export it — note "
            "that on Windows a newly-set user-level variable is not visible to already-running "
            "processes, so a fresh terminal may be needed."
        )
    url = f"{cfg.featherless_base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {cfg.featherless_api_key}"}
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url, headers=headers)
    if response.status_code >= 400:
        raise SystemExit(
            f"catalogue fetch failed: http {response.status_code}: {response.text[:200]}"
        )
    body = response.json()
    models = body.get("data") if isinstance(body, dict) else body
    if not isinstance(models, list):
        raise SystemExit(f"unexpected catalogue shape: {type(models).__name__}")
    return models


def _available(models: list[dict]) -> list[dict]:
    return [m for m in models if m.get("available_on_current_plan", True)]


def print_summary(models: list[dict]) -> None:
    avail = _available(models)
    print(f"total: {len(models)}   available on this plan: {len(avail)}")
    print("\nconcurrency_cost distribution:")
    costs: dict[int, int] = {}
    for m in avail:
        cost = int(m.get("concurrency_cost") or 0)
        costs[cost] = costs.get(cost, 0) + 1
    for cost in sorted(costs):
        print(f"  cost {cost}: {costs[cost]} models")
    print("\nlargest model_class families:")
    families: dict[str, int] = {}
    for m in avail:
        key = str(m.get("model_class") or "?")
        families[key] = families.get(key, 0) + 1
    for name, count in sorted(families.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {name}: {count}")


def select(models: list[dict], args: argparse.Namespace) -> list[dict]:
    out = _available(models)
    if args.grep:
        pattern = re.compile(args.grep, re.IGNORECASE)
        out = [m for m in out if pattern.search(str(m.get("id", "")))]
    if args.class_grep:
        pattern = re.compile(args.class_grep, re.IGNORECASE)
        out = [m for m in out if pattern.search(str(m.get("model_class", "")))]
    if args.max_cost:
        out = [m for m in out if int(m.get("concurrency_cost") or 99) <= args.max_cost]
    return out


def print_list(models: list[dict], top: int) -> None:
    print(f"{'model id':52} {'class':18} {'units':6} {'ctx':8}")
    print("-" * 88)
    for m in models[:top]:
        print(
            f"{str(m.get('id'))[:52]:52} {str(m.get('model_class'))[:18]:18} "
            f"{str(m.get('concurrency_cost')):6} {str(m.get('context_length')):8}"
        )
    if len(models) > top:
        print(f"... {len(models) - top} more (raise --top)")


def pin(models: list[dict], wanted: list[str], manifest_path: Path) -> int:
    """Write a small, tracked manifest of chosen workers with catalogue-reported units."""
    by_id = {str(m.get("id")): m for m in models}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        print(f"not in the catalogue: {', '.join(missing)}", file=sys.stderr)
        print("a wrong model id is a 404 discovered mid-run — refusing to pin it", file=sys.stderr)
        return 2

    entries = [manifest_entry(by_id[model_id]) for model_id in wanted]
    unsized = [e["name"] for e in entries if e["params_b"] is None]
    if unsized:
        print(
            f"warning: no parameter count derived for {', '.join(unsized)} — their catalog lines "
            "will read 'unknown size', which is a catalog the conductor cannot route on. "
            "Add params_b by hand in the manifest.",
            file=sys.stderr,
        )

    manifest = {
        "pinned_at": datetime.now(UTC).isoformat(),
        "source": "featherless /v1/models (concurrency_cost is the provider's own number)",
        "workers": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"pinned {len(entries)} worker(s) to {manifest_path}")
    for e in entries:
        size = f"{e['params_b']:g}B" if e["params_b"] else "?"
        tags = ", ".join(e["tags"]) or "-"
        print(f"  {e['name']:14} {e['model']:44} {size:>6} units={e['units']}  [{tags}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (args.summary or args.list or args.pin is not None):
        print("nothing to do: pass --summary, --list, or --pin", file=sys.stderr)
        return 2

    models = fetch_catalog(args.timeout)

    if args.summary:
        print_summary(models)
    if args.list:
        print_list(select(models, args), args.top)
    if args.pin is not None:
        if not args.pin:
            print("--pin needs at least one model id", file=sys.stderr)
            return 2
        return pin(models, list(args.pin), Path(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
