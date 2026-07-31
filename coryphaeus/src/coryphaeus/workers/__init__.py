"""Worker pool: the protocol, the providers, the registry, and the concurrency governor."""

from .base import Worker, WorkerBusy, WorkerResult, WorkerSpec, units_for_params
from .fake import FakeWorker, fake_spec, make_fake_pool
from .featherless import FeatherlessWorker, MissingApiKey, featherless_spec
from .governor import Governor, UnitPool, governor_for
from .ollama import OllamaWorker, ollama_spec

# MissingApiKey is per-adapter (featherless, openrouter, and the cloud providers each define
# their own, naming their own env var); the package-level name keeps pointing at featherless's
# for backward compatibility. Import openrouter's from its module when you need that one.
from .openrouter import MissingPin, OpenRouterWorker, openrouter_spec
from .registry import CATALOG_VERSION, WorkerRegistry

__all__ = [
    "CATALOG_VERSION",
    "FakeWorker",
    "FeatherlessWorker",
    "Governor",
    "MissingApiKey",
    "MissingPin",
    "OllamaWorker",
    "OpenRouterWorker",
    "UnitPool",
    "Worker",
    "WorkerBusy",
    "WorkerRegistry",
    "WorkerResult",
    "WorkerSpec",
    "fake_spec",
    "featherless_spec",
    "governor_for",
    "make_fake_pool",
    "ollama_spec",
    "openrouter_spec",
    "units_for_params",
]
