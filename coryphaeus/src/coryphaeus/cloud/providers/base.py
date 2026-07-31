"""The provider contract. Two real backends implement it; the fake one tests everything above it.

Kept deliberately small: five operations are enough to provision, watch, reach, and kill a box.
Anything a specific provider needs beyond this (offer filtering, volume mounting) lives inside
that backend, behind these five verbs.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """A provider API call failed in a way retries did not fix. Message carries the provider's
    own reason — a refused provision must be explainable to a human reading a log at 7am."""


class InstanceState(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    STOPPED = "stopped"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class GpuOffer:
    """One rentable configuration, normalized across providers."""

    provider: str
    offer_id: str
    gpu_name: str
    vram_gb: int
    price_per_hour: float
    #: Providers price interruptible capacity differently; None = on-demand only.
    interruptible_price_per_hour: float | None = None
    #: Host-set network pricing exists on marketplace providers; 0.0 elsewhere.
    egress_per_gb: float = 0.0
    #: Free-form provider metadata the backend may need to provision this exact offer.
    raw: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Instance:
    """A provisioned box."""

    provider: str
    instance_id: str
    state: InstanceState
    gpu_name: str
    price_per_hour: float
    #: SSH endpoint once RUNNING; None while PENDING.
    ssh_host: str | None = None
    ssh_port: int | None = None
    raw: dict = field(default_factory=dict)


@runtime_checkable
class CloudProvider(Protocol):
    """What the launcher needs from any GPU rental provider."""

    name: str

    async def offers(self, *, min_vram_gb: int, max_price_per_hour: float) -> Sequence[GpuOffer]:
        """Currently rentable offers meeting the floor specs, cheapest first."""
        ...

    async def provision(
        self,
        offer: GpuOffer,
        *,
        image: str,
        env: dict[str, str],
        volume_gb: int,
        label: str,
    ) -> Instance:
        """Rent the offer, injecting ``env`` as instance environment. Returns PENDING/RUNNING."""
        ...

    async def describe(self, instance_id: str) -> Instance:
        """Current state. UNKNOWN on a describe failure — the caller decides how to treat it."""
        ...

    async def terminate(self, instance_id: str) -> None:
        """Stop billing. MUST be idempotent: terminating a terminated instance is a no-op, not an
        error, because the launcher calls this from every exit path including crash handlers."""
        ...

    async def cost_so_far(self, instance_id: str) -> float | None:
        """Accrued spend for this instance in dollars, if the provider reports it; else None and
        the ledger falls back to wall-clock x price."""
        ...
