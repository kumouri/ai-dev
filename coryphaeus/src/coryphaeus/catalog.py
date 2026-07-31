"""Turning a provider catalogue entry into a worker spec.

Pure functions, no network — the fetching lives in ``scripts/featherless_catalog.py``. They are here
rather than in the script because what they produce is the **catalog line the conductor reads**, and
that is the only thing it knows about a worker. Getting a size or a specialist tag wrong is not a
cosmetic bug; it is routing information silently missing.
"""

from __future__ import annotations

import re

#: Below this many billion parameters a worker is "fast"; above SIZE_STRONG it is "strong".
SIZE_FAST = 8.0
SIZE_STRONG = 20.0


def params_from_model_class(model_class: str | None) -> float | None:
    """Read a parameter count out of the provider's ``model_class``.

    The class encodes size with ``b`` acting as the decimal point: ``qwen25-7b`` → 7.0,
    ``qwen2-0b5`` → 0.5, ``qwen3-1b7`` → 1.7, ``llama33-70b`` → 70.0.

    Returns ``None`` when nothing size-like is present, which the caller should treat as a problem
    to fix rather than accept — "unknown size" is a catalog that cannot be routed on.
    """
    if not model_class:
        return None
    match = re.search(r"(\d+)b(\d*)$", model_class.strip().lower())
    if not match:
        return None
    whole, frac = match.group(1), match.group(2)
    return float(f"{whole}.{frac}") if frac else float(whole)


def derive_tags(model_id: str, params_b: float | None) -> tuple[str, ...]:
    """Capability hints for the catalog, derived from provider data rather than invented.

    A specialist tag is the highest-value signal in a mixed pool: a math-tuned 7B is exactly the
    worker that makes routing pay, rather than the policy degenerating into "pick the biggest".

    Deliberately *not* tagged: high concurrency cost. The cost is real — a 4-unit worker serializes
    a 4-unit account — but the reward does not price it yet, and an unpriced tag is prompt budget
    spent on nothing. Add it when the cost term lands.
    """
    tags: list[str] = []
    lowered = model_id.lower()
    if "math" in lowered:
        tags.append("math")
    if "coder" in lowered or "code" in lowered:
        tags.append("code")
    if params_b is not None:
        if params_b < SIZE_FAST:
            tags.append("fast")
        elif params_b <= SIZE_STRONG:
            tags.append("balanced")
        else:
            tags.append("strong")
    return tuple(tags)


def short_name(model_id: str) -> str:
    """A terse registry name. The conductor *writes* this into workflows, so it is prompt budget.

    ``Qwen/Qwen2.5-Math-7B-Instruct`` → ``qwen25-math-7b``.
    """
    tail = model_id.split("/")[-1]
    tail = re.sub(r"-Instruct$", "", tail, flags=re.IGNORECASE)
    tail = tail.replace("Qwen2.5", "qwen25").replace("Qwen3", "qwen3")
    tail = re.sub(r"Llama-3\.3", "llama33", tail, flags=re.IGNORECASE)
    tail = re.sub(r"[^0-9A-Za-z]+", "-", tail).strip("-").lower()
    return tail


def manifest_entry(model: dict) -> dict:
    """Build one manifest row from a catalogue entry.

    ``units`` comes from the provider's own ``concurrency_cost``, never from a size heuristic — the
    24-32B band costs 2, and guessing 4 there halves effective concurrency for nothing.
    """
    model_id = str(model.get("id"))
    model_class = model.get("model_class")
    params_b = params_from_model_class(model_class)
    return {
        "name": short_name(model_id),
        "provider": "featherless",
        "model": model_id,
        "model_class": model_class,
        "params_b": params_b,
        "units": int(model.get("concurrency_cost") or 1),
        "tags": list(derive_tags(model_id, params_b)),
        "context_length": model.get("context_length"),
    }
