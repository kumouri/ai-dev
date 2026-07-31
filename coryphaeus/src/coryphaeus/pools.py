"""Worker pools, and how they are sampled.

Pool *composition* is an experimental variable, not configuration. Two reasons:

1. Routing cannot pay in a pool of near-identical workers, so a pool has to be deliberately spread
   across size, speed and speciality.
2. The paper trains over **randomised** agent pools, which is how its conductor generalises to
   arbitrary worker sets. A conductor evaluated on one fixed pool has learned that pool rather than
   routing — so pool compositions get a train/eval split of their own, alongside the questions.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .workers import (
    FeatherlessWorker,
    OllamaWorker,
    WorkerRegistry,
    featherless_spec,
    ollama_spec,
)

#: Pinned remote pool across providers. Tracked on purpose: pinning is what keeps runs comparable.
#: Featherless seats are refreshed by ``scripts/featherless_catalog.py`` (``units`` from the
#: provider's own ``concurrency_cost``); OpenRouter seats are hand-pinned from live endpoint probes
#: and carry their upstream ``pin``, per-token prices and quantization — a seat change is a
#: different served system, so it must be visible in this file's diff. Note the repo ignores any
#: directory named ``data/``, hence ``manifests/``.
MANIFEST_PATH = Path(__file__).resolve().parent / "manifests" / "worker_pool.json"

#: Local pool. A ``size_gb`` above assumed VRAM makes a worker near-serial (see workers/ollama.py).
LOCAL_POOL: tuple[dict, ...] = (
    {
        "name": "q2",
        "model": "qwen3.5:2b",
        "params_b": 2.0,
        "size_gb": 2.6,
        "tags": ("fast", "cheap"),
    },
    {
        "name": "q4",
        "model": "qwen3.5:4b",
        "params_b": 4.0,
        "size_gb": 3.2,
        "tags": ("fast", "general"),
    },
    {"name": "q9", "model": "qwen3.5:9b", "params_b": 9.0, "size_gb": 6.1, "tags": ("balanced",)},
    {
        "name": "g12",
        "model": "gemma4:12b",
        "params_b": 12.0,
        "size_gb": 7.0,
        "tags": ("balanced", "prose"),
    },
    {
        "name": "q27",
        "model": "qwen3.6:27b",
        "params_b": 27.0,
        "size_gb": 16.2,
        "tags": ("strong", "math", "slow"),
    },
)


@dataclass(frozen=True, slots=True)
class PoolSplit:
    """Train and evaluation pool compositions, held apart.

    Held-out compositions are how we tell routing from pool-memorisation: a conductor that only ever
    saw one catalogue may simply have learned "always pick the third one".
    """

    train: tuple[tuple[str, ...], ...]
    evaluation: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        overlap = set(self.train) & set(self.evaluation)
        if overlap:
            raise ValueError(f"pool compositions leak across the split: {sorted(overlap)}")


def load_manifest(path: Path | None = None) -> list[dict]:
    """Read the pinned remote-pool manifest. Missing file → empty pool, not an error."""
    path = path or MANIFEST_PATH
    if not path.is_file():
        return []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return list(manifest.get("workers") or [])


def build_local_registry(models: tuple[str, ...] | None = None) -> WorkerRegistry:
    """Registry over the local Ollama pool, optionally filtered to ``models`` (by worker name)."""
    registry = WorkerRegistry()
    for entry in LOCAL_POOL:
        if models and entry["name"] not in models:
            continue
        registry.add(OllamaWorker(ollama_spec(**entry)))
    return registry


def _build_remote_worker(entry: dict, units: int):
    """One manifest row → one worker, dispatched on the row's ``provider``."""
    provider = entry.get("provider", "featherless")
    name = entry["name"]
    if provider == "featherless":
        return FeatherlessWorker(
            featherless_spec(
                name,
                entry["model"],
                params_b=entry.get("params_b"),
                units=units,
                tags=tuple(entry.get("tags") or ()),
            )
        )
    if provider == "openrouter":
        # Local import: the adapter needs OPENROUTER_API_KEY machinery that a featherless-only
        # (or offline) caller should never be asked to satisfy.
        from .workers.openrouter import OpenRouterWorker, openrouter_spec

        pin = entry.get("pin")
        if not pin:
            raise ValueError(
                f"{name}: openrouter manifest entries require a 'pin' (upstream provider slug) — "
                "unpinned routing is a different served system per request."
            )
        if entry.get("price_in_per_m") is None or entry.get("price_out_per_m") is None:
            raise ValueError(
                f"{name}: openrouter manifest entries require price_in_per_m/price_out_per_m — "
                "a paid worker with no price would look free in every report."
            )
        return OpenRouterWorker(
            openrouter_spec(
                name,
                entry["model"],
                usd_per_mtok_in=float(entry["price_in_per_m"]),
                usd_per_mtok_out=float(entry["price_out_per_m"]),
                params_b=entry.get("params_b"),
                units=units,
                tags=tuple(entry.get("tags") or ()),
            ),
            pin=pin if isinstance(pin, str) else tuple(pin),
            # JSON null → None → the reasoning field is omitted: for upstreams with no reasoning
            # control, sending it under require_parameters 404s the seat ("No endpoints found").
            think=entry.get("think", False),
            # Seats whose endpoint reasons regardless of the knob need room for burn + answer;
            # a generic cap truncates them into fake incompetence (baseline 2026-07-31).
            max_tokens_floor=int(entry.get("max_tokens_floor") or 0),
        )
    raise ValueError(f"{name}: unknown provider {provider!r} in the worker manifest")


def build_remote_registry(
    models: tuple[str, ...] | None = None,
    *,
    manifest: Path | None = None,
    max_units: int | None = None,
    providers: tuple[str, ...] | None = None,
) -> WorkerRegistry:
    """Registry over the pinned remote pool — every provider in the manifest.

    Requires the API key for each provider that actually appears after filtering
    (``FEATHERLESS_API_KEY``, ``OPENROUTER_API_KEY``).

    Args:
        max_units: drop workers costing more than this many concurrency units. Use it for training:
            Featherless's total budget is small, so a single high-cost worker **serializes every
            other rollout** — cheap in money, expensive in wall-clock-for-everything-else.
            OpenRouter seats cost 1 (no concurrency premium), so they pass this filter; their
            throttle is the governor's account-wide budget. Leave it unset for evaluation, where
            the serializing trade is fine.
        providers: keep only these providers' seats. The point is key hygiene as much as
            selection — a featherless-only caller must not be asked for an OpenRouter key just
            because the manifest gained seats.
    """
    registry = WorkerRegistry()
    for entry in load_manifest(manifest):
        if models and entry["name"] not in models:
            continue
        if providers and entry.get("provider", "featherless") not in providers:
            continue
        units = int(entry.get("units") or 1)
        if max_units is not None and units > max_units:
            continue
        registry.add(_build_remote_worker(entry, units))
    return registry


def build_registry(
    *,
    local: bool = True,
    remote: bool = False,
    models: tuple[str, ...] | None = None,
    max_units: int | None = None,
) -> WorkerRegistry:
    """Combined registry. Names must be unique across providers — the registry enforces it."""
    registry = WorkerRegistry()
    if local:
        for worker in build_local_registry(models):
            registry.add(worker)
    if remote:
        for worker in build_remote_registry(models, max_units=max_units):
            registry.add(worker)
    return registry


def sample_pool(
    registry: WorkerRegistry,
    rng: random.Random,
    *,
    size: int | tuple[int, int] = (2, 4),
) -> tuple[str, ...]:
    """Pick a random sub-pool of worker names.

    Deterministic for a given ``rng`` seed, so a training run is reproducible and a specific
    composition can be replayed from its recorded seed.

    Args:
        size: exact size, or an inclusive ``(min, max)`` range. Clamped to the registry's size; a
            single-worker pool is allowed but pointless (there is nothing to route between), so the
            minimum is 2 wherever the registry can supply it.
    """
    names = list(registry.names())
    if not names:
        raise ValueError("cannot sample from an empty registry")

    if isinstance(size, tuple):
        low, high = size
    else:
        low = high = size
    low = max(1, min(low, len(names)))
    high = max(low, min(high, len(names)))
    if len(names) >= 2:
        low = max(2, low)
        high = max(low, high)

    count = rng.randint(low, high)
    # sample() rather than shuffle-and-slice: order is part of the composition (it is the order the
    # catalog lists workers in, which the conductor sees), so it must vary with the seed too.
    return tuple(rng.sample(names, count))


def sample_pool_split(
    registry: WorkerRegistry,
    *,
    seed: int = 0,
    n_train: int = 8,
    n_eval: int = 4,
    size: int | tuple[int, int] = (2, 4),
) -> PoolSplit:
    """Generate distinct train and evaluation pool compositions.

    Draws until each side has its quota of *distinct* compositions and the two sides do not overlap.
    Gives up rather than looping forever when the registry is too small to supply that many — a
    3-worker registry simply cannot yield 12 distinct sub-pools, and silently returning fewer would
    make a leak look like a pass.
    """
    rng = random.Random(seed)
    wanted = n_train + n_eval
    seen: list[tuple[str, ...]] = []
    attempts = 0
    max_attempts = max(200, wanted * 50)
    while len(seen) < wanted and attempts < max_attempts:
        attempts += 1
        candidate = sample_pool(registry, rng, size=size)
        if candidate not in seen:
            seen.append(candidate)
    if len(seen) < wanted:
        raise ValueError(
            f"registry of {len(registry)} workers yielded only {len(seen)} distinct compositions "
            f"of size {size}; asked for {wanted}. Widen the pool or lower n_train/n_eval."
        )
    return PoolSplit(train=tuple(seen[:n_train]), evaluation=tuple(seen[n_train:wanted]))
