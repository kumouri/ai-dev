"""The worker registry — name resolution plus the catalog the conductor actually reads.

The catalog is *prompt budget*, not documentation. It is the only thing the conductor knows about
its workers, so its wording is an experimental variable in its own right (see ROADMAP phase 1) and
is versioned rather than casually edited.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .base import Worker, WorkerSpec
from .governor import Governor, governor_for

#: Bump when the rendered catalog text changes, so runs remain comparable.
CATALOG_VERSION = "v1"


class WorkerRegistry:
    """An ordered, name-addressed collection of workers."""

    def __init__(self, workers: Iterable[Worker] = ()) -> None:
        self._workers: dict[str, Worker] = {}
        for worker in workers:
            self.add(worker)

    def add(self, worker: Worker) -> Worker:
        name = worker.spec.name
        if name in self._workers:
            raise ValueError(f"duplicate worker name: {name!r}")
        self._workers[name] = worker
        return worker

    def __len__(self) -> int:
        return len(self._workers)

    def __iter__(self) -> Iterator[Worker]:
        return iter(self._workers.values())

    def __contains__(self, name: object) -> bool:
        return name in self._workers

    def get(self, name: str) -> Worker:
        try:
            return self._workers[name]
        except KeyError:
            raise KeyError(f"unknown worker {name!r}; known: {', '.join(self.names())}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(self._workers)

    def specs(self) -> tuple[WorkerSpec, ...]:
        return tuple(w.spec for w in self._workers.values())

    def by_provider(self, provider: str) -> tuple[Worker, ...]:
        return tuple(w for w in self._workers.values() if w.spec.provider == provider)

    def catalog_text(self) -> str:
        """The worker menu shown to the conductor."""
        return "\n".join(spec.catalog_line() for spec in self.specs())

    def governor(self, **kwargs: object) -> Governor:
        """A governor sized for the providers in this registry."""
        return governor_for(self.specs(), **kwargs)
