"""Parse a conductor's raw text emission into a :class:`~coryphaeus.schema.Workflow`.

The parser runs **one deterministic repair pass** over formatting noise (fenced blocks, trailing
commas, smart quotes, stray prose) and then fails with a named reason. It never asks a model to fix
a workflow: a malformed workflow is reward signal, and laundering it would have the harness quietly
doing the policy's job. See ``docs/adr/0001``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .schema import Step, WorkflowError, build

DepDefault = Literal["all", "prev", "none"]

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Outcome of parsing. Exactly one of ``workflow`` / ``reason`` is set."""

    workflow: object | None
    reason: str | None
    raw: str
    detail: str = ""
    repaired: bool = False

    @property
    def ok(self) -> bool:
        return self.workflow is not None


def _candidate_payloads(text: str) -> list[str]:
    """Substrings that might be the JSON object, best guess first.

    Ordered deliberately: a fenced block is the strongest signal, the *last* one strongest of all
    (models often show a draft then a final). Balanced-brace extraction is the fallback for models
    that ignore the fence instruction.
    """
    candidates: list[str] = []
    fenced = _FENCE_RE.findall(text)
    candidates.extend(block.strip() for block in reversed(fenced))

    depth = 0
    start = -1
    spans: list[str] = []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : i + 1])
    candidates.extend(reversed(spans))

    stripped = text.strip()
    if stripped:
        candidates.append(stripped)

    seen: dict[str, None] = {}
    for c in candidates:
        if c:
            seen.setdefault(c, None)
    return list(seen)


def _repair(payload: str) -> str:
    """Deterministic formatting fixes only — never semantic ones."""
    fixed = payload.translate(_SMART_QUOTES)
    fixed = _TRAILING_COMMA_RE.sub(r"\1", fixed)
    # Strip line comments, which small models like to add inside "JSON".
    fixed = re.sub(r"^\s*//.*$", "", fixed, flags=re.MULTILINE)
    return fixed.strip()


def _load(payload: str) -> tuple[dict | None, bool]:
    """Return (parsed object, whether the repair pass was needed)."""
    try:
        obj = json.loads(payload)
        return (obj if isinstance(obj, dict) else None), False
    except json.JSONDecodeError:
        pass
    try:
        obj = json.loads(_repair(payload))
        return (obj if isinstance(obj, dict) else None), True
    except json.JSONDecodeError:
        return None, True


def _coerce_deps(
    value: object, *, position: int, default: DepDefault
) -> tuple[tuple[int, ...] | None, str]:
    """Normalise a step's ``deps`` field. Returns (deps, error-detail)."""
    if value is None:
        if default == "all":
            return tuple(range(1, position)), ""
        if default == "prev":
            return ((position - 1,) if position > 1 else ()), ""
        return (), ""
    if isinstance(value, int) and not isinstance(value, bool):
        return (value,), ""
    if isinstance(value, str):
        found = [int(m) for m in re.findall(r"\d+", value)]
        return tuple(found), ""
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            if isinstance(item, bool):
                return None, f"step {position}: boolean dep"
            if isinstance(item, int):
                out.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                out.append(int(item.strip()))
            else:
                return None, f"step {position}: non-integer dep {item!r}"
        return tuple(out), ""
    return None, f"step {position}: deps has type {type(value).__name__}"


def parse(
    text: str,
    *,
    known_workers: Iterable[str],
    default_deps: DepDefault = "all",
) -> ParseResult:
    """Parse ``text`` into a workflow, or report why it could not be.

    Args:
        text: the policy's raw emission.
        known_workers: worker names the registry can resolve.
        default_deps: what an omitted ``deps`` field means. ``"all"`` (every earlier step) is the
            default because the common last step is an aggregation, and a chain default would
            silently starve it of context. A trained policy's prompt should require deps explicitly.

    Returns:
        A :class:`ParseResult`. Failure reasons come from :data:`coryphaeus.schema.REASONS` plus
        ``json_decode``, ``no_object``, ``steps_not_a_list`` and ``bad_step_object``.
    """
    workers = list(known_workers)

    obj: dict | None = None
    repaired = False
    for payload in _candidate_payloads(text):
        obj, repaired = _load(payload)
        if obj is not None and "steps" in obj:
            break
        obj = None
    if obj is None:
        return ParseResult(None, "json_decode", text, "no JSON object with a 'steps' key", repaired)

    raw_steps = obj.get("steps")
    if not isinstance(raw_steps, list):
        return ParseResult(None, "steps_not_a_list", text, type(raw_steps).__name__, repaired)
    if not raw_steps:
        return ParseResult(None, "no_steps", text, "", repaired)

    steps: list[Step] = []
    for position, raw_step in enumerate(raw_steps, start=1):
        if not isinstance(raw_step, dict):
            return ParseResult(
                None,
                "bad_step_object",
                text,
                f"step {position}: {type(raw_step).__name__}",
                repaired,
            )
        subtask = raw_step.get("subtask") or raw_step.get("task") or raw_step.get("instruction")
        worker = raw_step.get("worker") or raw_step.get("model") or raw_step.get("agent")
        if not isinstance(subtask, str):
            return ParseResult(None, "empty_subtask", text, f"step {position}", repaired)
        if not isinstance(worker, str):
            return ParseResult(None, "unknown_worker", text, f"step {position}: missing", repaired)

        deps_field = raw_step.get("deps", raw_step.get("depends_on", raw_step.get("inputs")))
        deps, detail = _coerce_deps(deps_field, position=position, default=default_deps)
        if deps is None:
            return ParseResult(None, "bad_dep", text, detail, repaired)

        steps.append(
            Step(index=position, subtask=subtask.strip(), worker=worker.strip(), deps=deps)
        )

    final = obj.get("final", obj.get("final_step", obj.get("answer_step")))
    if isinstance(final, str) and final.strip().isdigit():
        final = int(final.strip())
    if final is not None and not isinstance(final, int):
        return ParseResult(None, "bad_final", text, repr(final), repaired)

    try:
        workflow = build(steps, final, known_workers=workers, raw=text)
    except WorkflowError as exc:
        return ParseResult(None, exc.reason, text, exc.detail, repaired)
    return ParseResult(workflow, None, text, "", repaired)
