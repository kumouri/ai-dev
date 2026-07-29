"""Ready-made worker pools.

The *composition* of a pool is an experimental variable: routing cannot pay in a pool of
near-identical workers, so these are deliberately spread across size, speed and speciality.

Model names here are defaults, not requirements. ``--models`` on the scripts overrides them, and a
missing model is reported by ``smoke_workers.py`` rather than failing a run halfway through.
"""

from __future__ import annotations

from .workers import (
    FeatherlessWorker,
    OllamaWorker,
    WorkerRegistry,
    featherless_spec,
    ollama_spec,
)

#: Local pool. Sizes and on-disk footprints are what drive the unit costs; ``size_gb`` above the
#: assumed VRAM makes a worker near-serial (see ``workers/ollama.py``).
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

#: Remote pool. Left empty by default: model ids differ per account and a wrong id is a 404 mid-run.
#: Populate from the provider's catalog, or pass ``--models`` to the scripts.
REMOTE_POOL: tuple[dict, ...] = ()


def build_local_registry(models: tuple[str, ...] | None = None) -> WorkerRegistry:
    """Registry over the local Ollama pool, optionally filtered to ``models`` (by worker name)."""
    registry = WorkerRegistry()
    for entry in LOCAL_POOL:
        if models and entry["name"] not in models:
            continue
        registry.add(OllamaWorker(ollama_spec(**entry)))
    return registry


def build_remote_registry(models: tuple[str, ...] | None = None) -> WorkerRegistry:
    """Registry over the remote Featherless pool. Requires ``FEATHERLESS_API_KEY``."""
    registry = WorkerRegistry()
    for entry in REMOTE_POOL:
        if models and entry["name"] not in models:
            continue
        registry.add(FeatherlessWorker(featherless_spec(**entry)))
    return registry


def build_registry(
    *,
    local: bool = True,
    remote: bool = False,
    models: tuple[str, ...] | None = None,
) -> WorkerRegistry:
    """Combined registry. Names must be unique across providers — the registry enforces it."""
    registry = WorkerRegistry()
    if local:
        for worker in build_local_registry(models):
            registry.add(worker)
    if remote:
        for worker in build_remote_registry(models):
            registry.add(worker)
    return registry
