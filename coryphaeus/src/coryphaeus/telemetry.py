"""Run artifacts — and the dataset the world model will be trained on.

This is not logging. A ``steps.jsonl`` row is one ``(question, rollout, step, worker, subtask, deps,
tokens, latency, cost, ok, error)`` tuple, and the matching ``rollouts.jsonl`` row has the score.
Together they are exactly the supervised dataset for ``W(subtask, worker) → P(success)`` — which is
why every rollout is recorded from the very first run rather than after a decision to study it.

Cost and latency are recorded even for local and flat-rate workers where marginal dollar cost is
zero. The result being replicated is a cost result; a number not recorded cannot be reported.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import settings

#: Bump on any breaking change to a row's fields. Readers should check it.
SCHEMA_VERSION = 2


def utc_stamp() -> str:
    """Filesystem-safe UTC timestamp, e.g. ``20260729T203115Z``."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


@dataclass(slots=True)
class RunWriter:
    """Append-only writer for one run directory.

    Files: ``meta.json`` (what was run), ``rollouts.jsonl`` (one row per scored rollout),
    ``steps.jsonl`` (one row per executed step).
    """

    run_dir: Path
    _rollouts: object = None
    _steps: object = None

    @classmethod
    def create(cls, label: str = "run", *, root: Path | None = None) -> RunWriter:
        root = root or settings().runs_dir
        run_dir = root / f"{utc_stamp()}-{label}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(run_dir=run_dir)

    def __enter__(self) -> RunWriter:
        self._rollouts = (self.run_dir / "rollouts.jsonl").open("a", encoding="utf-8")
        self._steps = (self.run_dir / "steps.jsonl").open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in (self._rollouts, self._steps):
            if handle is not None:
                handle.close()  # type: ignore[union-attr]
        self._rollouts = self._steps = None

    def write_meta(self, **fields: object) -> None:
        """Write ``meta.json``. Includes environment facts, but nothing identifying a machine."""
        meta = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "platform": platform.system(),
            **fields,
        }
        (self.run_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
        )

    def _append(self, handle: object, row: dict) -> None:
        if handle is None:
            raise RuntimeError("RunWriter used outside its context manager")
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")  # type: ignore[union-attr]
        handle.flush()  # type: ignore[union-attr]
        os.fsync(handle.fileno())  # type: ignore[union-attr]

    def write_rollout(self, row: dict) -> None:
        self._append(self._rollouts, {"schema_version": SCHEMA_VERSION, **row})

    def write_step(self, row: dict) -> None:
        self._append(self._steps, {"schema_version": SCHEMA_VERSION, **row})


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL, skipping blanks. A malformed line raises — silence would hide data loss."""
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: malformed JSONL: {exc}") from exc
    return rows


def step_rows(
    *,
    run_id: str,
    question_id: str,
    arm: str,
    rollout_index: int,
    execution: object,
) -> list[dict]:
    """Flatten an :class:`~coryphaeus.orchestrate.ExecutionResult` into step rows.

    Feature columns for the world model come first (subtask, worker, spec facts); outcome columns
    follow. ``subtask_chars`` is kept alongside the text so a model can be trained on either.
    """
    rows: list[dict] = []
    for outcome in getattr(execution, "outcomes", []):
        step = outcome.step
        result = outcome.result
        rows.append(
            {
                "run_id": run_id,
                "arm": arm,
                "question_id": question_id,
                "rollout_index": rollout_index,
                "step_index": step.index,
                "subtask": step.subtask,
                "subtask_chars": len(step.subtask),
                "deps": list(step.deps),
                "worker_requested": step.worker,
                "worker": outcome.resolved_worker,
                "prompt_chars": outcome.prompt_chars,
                "ok": result.ok,
                "error": result.error,
                "attempts": result.attempts,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "latency_s": round(result.latency_s, 4),
                "cost_usd": result.cost_usd,
                "response_chars": len(result.text or ""),
            }
        )
    return rows
