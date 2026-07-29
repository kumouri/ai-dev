"""Re-render the comparison for a finished run.

uv run python coryphaeus/scripts/report.py runs/20260729T210000Z-baseline
uv run python coryphaeus/scripts/report.py --latest
uv run python coryphaeus/scripts/report.py --latest --workers   # per-worker step detail
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coryphaeus.config import settings
from coryphaeus.reporting import render_parse_failures, render_table, render_verdict, summarize
from coryphaeus.telemetry import read_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", help="path to a run directory")
    parser.add_argument("--latest", action="store_true", help="use the most recent run")
    parser.add_argument("--workers", action="store_true", help="add per-worker step statistics")
    return parser.parse_args(argv)


def resolve_run_dir(args: argparse.Namespace) -> Path | None:
    if args.run_dir:
        return Path(args.run_dir)
    if args.latest:
        root = settings().runs_dir
        # Only directories that actually hold a run. Checkpoint directories live under the same root
        # and sort after the timestamped ones, so a plain name sort picks the wrong thing.
        candidates = sorted(
            (
                p
                for p in root.glob("*")
                if p.is_dir() and ((p / "rollouts.jsonl").is_file() or (p / "meta.json").is_file())
            ),
            reverse=True,
        )
        return candidates[0] if candidates else None
    return None


def worker_table(step_rows: list[dict]) -> str:
    """Per-worker step statistics — the raw material for the world model, summarised."""
    stats: dict[str, dict[str, float]] = {}
    for row in step_rows:
        worker = row.get("worker", "?")
        entry = stats.setdefault(worker, {"steps": 0, "ok": 0, "latency": 0.0, "tokens": 0})
        entry["steps"] += 1
        entry["ok"] += 1 if row.get("ok") else 0
        entry["latency"] += float(row.get("latency_s") or 0.0)
        entry["tokens"] += int(row.get("tokens_in") or 0) + int(row.get("tokens_out") or 0)

    lines = [f"{'worker':16} {'steps':7} {'ok':8} {'lat/step':10} {'tok/step':9}", "-" * 54]
    for worker, entry in sorted(stats.items(), key=lambda kv: -kv[1]["steps"]):
        steps = entry["steps"]
        lines.append(
            f"{worker:16} {int(steps):<7} {entry['ok'] / steps:>6.1%}  "
            f"{entry['latency'] / steps:>8.2f}s  {entry['tokens'] / steps:>8.0f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = resolve_run_dir(args)
    if run_dir is None:
        print("give a run directory or --latest", file=sys.stderr)
        return 2
    if not run_dir.is_dir():
        print(f"no such run directory: {run_dir}", file=sys.stderr)
        return 2

    meta_path = run_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        status = meta.get("status", "unknown")
        elapsed = meta.get("elapsed_s")
        # An unfinished run has no elapsed time yet. Say "in progress" rather than "Nones", and say
        # it loudly — a partial slice is a legitimate thing to read, but not to quote as final.
        timing = f"elapsed: {elapsed}s" if elapsed is not None else "IN PROGRESS — partial results"
        kind = meta.get("kind", "eval")
        # Training runs carry different metadata than evaluation runs; print what is there rather
        # than a row of Nones.
        dataset = meta.get("dataset")
        n = meta.get("n_questions") or (dataset.get("rows") if isinstance(dataset, dict) else None)
        source = dataset if isinstance(dataset, str) else (meta.get("model") or "—")
        k = meta.get("k") or meta.get("num_generations")
        print(f"run     : {run_dir.name}  [{status}, {kind}]")
        print(f"source  : {source}" + (f" ({n} questions)" if n else ""))
        print(f"pool    : {', '.join(meta.get('pool') or [])}")
        print(f"k       : {k or '—'}   {timing}")
        print()

    rollout_rows = read_jsonl(run_dir / "rollouts.jsonl")
    if not rollout_rows:
        print(f"no rollouts recorded in {run_dir}", file=sys.stderr)
        return 1

    summaries = summarize(rollout_rows)
    print(render_table(summaries))
    print()
    print(render_verdict(summaries))
    print("\nparse failures by reason:")
    print(render_parse_failures(rollout_rows))

    if args.workers:
        step_data = read_jsonl(run_dir / "steps.jsonl")
        if step_data:
            print(f"\nper-worker step detail ({len(step_data)} steps):")
            print(worker_table(step_data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
