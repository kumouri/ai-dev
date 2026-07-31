"""A deterministic, offline cloud provider — the cloud twin of ``workers.fake``.

Everything above this layer (launcher, budget gate, auto-terminate, watch loops) is tested
against it: no network, no key, no wall clock. Time does not pass here — it is *polled* into
existence. An instance becomes RUNNING after ``ready_after`` describe() calls and cost accrues
per poll, so a test's Nth assertion sees the same world on every machine at every speed.

Failure is scripted, not simulated: flip :attr:`fail_provision` / :attr:`fail_describe` and the
fake misbehaves exactly the way the real backends do on their worst day — ``ProviderError`` from
provision, an UNKNOWN-state instance from describe (the base contract says describe never
raises). That is how crash-handler and give-up paths get exercised without renting anything.
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import GpuOffer, Instance, InstanceState, ProviderError

#: Two offers with deliberately different shapes: one cheap small card, one big expensive one,
#: distinct egress prices. Enough spread for filter/sort tests without inventing a catalog.
DEFAULT_OFFERS = (
    GpuOffer(
        provider="fake",
        offer_id="fake-4090",
        gpu_name="Fake RTX 4090",
        vram_gb=24,
        price_per_hour=0.30,
        interruptible_price_per_hour=0.15,
        egress_per_gb=0.002,
    ),
    GpuOffer(
        provider="fake",
        offer_id="fake-a100",
        gpu_name="Fake A100 80GB",
        vram_gb=80,
        price_per_hour=1.10,
        interruptible_price_per_hour=0.55,
        egress_per_gb=0.0,
    ),
)


class FakeCloudProvider:
    """A scriptable, inspectable CloudProvider.

    Args:
        offers: the fixed catalog; defaults to :data:`DEFAULT_OFFERS`.
        ready_after: describe() calls before an instance reports RUNNING. 2 (default) forces
            the caller through at least one PENDING poll — the wait loop is the code under
            test, so the fake must not let it be skipped by accident.
        cost_per_poll: dollars accrued per describe() of a live instance. Polls stand in for
            time, so spend grows while a test watches and freezes at termination — the exact
            invariant the budget ledger settles against.
        fail_provision: when True, provision raises :class:`ProviderError`.
        fail_describe: when True, describe returns UNKNOWN-state instances (never raises,
            matching the base contract) — flip it mid-test to script an API outage.
    """

    name = "fake"

    def __init__(
        self,
        *,
        offers: Sequence[GpuOffer] | None = None,
        ready_after: int = 2,
        cost_per_poll: float = 0.01,
        fail_provision: bool = False,
        fail_describe: bool = False,
    ) -> None:
        self._offers = tuple(offers) if offers is not None else DEFAULT_OFFERS
        self.ready_after = ready_after
        self.cost_per_poll = cost_per_poll
        self.fail_provision = fail_provision
        self.fail_describe = fail_describe
        #: Every provision, with the exact image/env/volume/label seen — the layer above must
        #: be able to assert what it *sent*, not just what it got back.
        self.provisions: list[dict] = []
        #: Every terminate call, including repeats and unknown ids — crash handlers double-call
        #: on purpose, and a test needs to see that both calls happened and neither raised.
        self.terminations: list[str] = []
        self._instances: dict[str, dict] = {}
        self._sequence = 0

    async def offers(self, *, min_vram_gb: int, max_price_per_hour: float) -> Sequence[GpuOffer]:
        matching = [
            offer
            for offer in self._offers
            if offer.vram_gb >= min_vram_gb and offer.price_per_hour <= max_price_per_hour
        ]
        return sorted(matching, key=lambda offer: offer.price_per_hour)

    async def provision(
        self,
        offer: GpuOffer,
        *,
        image: str,
        env: dict[str, str],
        volume_gb: int,
        label: str,
    ) -> Instance:
        if self.fail_provision:
            raise ProviderError("fake: provision refused (scripted by fail_provision)")
        self._sequence += 1
        instance_id = f"fake-{self._sequence}"
        self.provisions.append(
            {
                "instance_id": instance_id,
                "offer_id": offer.offer_id,
                "image": image,
                "env": dict(env),
                "volume_gb": volume_gb,
                "label": label,
            }
        )
        self._instances[instance_id] = {
            "offer": offer,
            "label": label,
            "polls": 0,
            "terminated": False,
        }
        return self._instance(instance_id, InstanceState.PENDING)

    async def describe(self, instance_id: str) -> Instance:
        if self.fail_describe:
            return self._instance(instance_id, InstanceState.UNKNOWN)
        record = self._instances.get(instance_id)
        if record is None:
            # Mirrors the real backends' 404 mapping: "nothing to describe" is what terminated
            # looks like from outside, and the watch loop must read it as "stopped billing".
            return self._instance(instance_id, InstanceState.TERMINATED)
        if record["terminated"]:
            return self._instance(instance_id, InstanceState.TERMINATED)
        record["polls"] += 1
        if record["polls"] >= self.ready_after:
            return self._instance(instance_id, InstanceState.RUNNING, ssh=True)
        return self._instance(instance_id, InstanceState.PENDING)

    async def list_instances(self) -> Sequence[Instance]:
        """Current-state snapshot of everything ever provisioned, terminated ones included as
        TERMINATED — the sweep's filtering is exactly what needs testing. Reading the account
        must not advance it: unlike describe, listing ticks no poll clock, so a sweep can run
        any number of times without booting a pending box or billing a live one."""
        listed: list[Instance] = []
        for instance_id, record in self._instances.items():
            if record["terminated"]:
                state = InstanceState.TERMINATED
            elif record["polls"] >= self.ready_after:
                state = InstanceState.RUNNING
            else:
                state = InstanceState.PENDING
            listed.append(self._instance(instance_id, state, ssh=state is InstanceState.RUNNING))
        return listed

    async def terminate(self, instance_id: str) -> None:
        self.terminations.append(instance_id)
        record = self._instances.get(instance_id)
        if record is not None:
            record["terminated"] = True

    async def cost_so_far(self, instance_id: str) -> float | None:
        """Polls x rate — deterministic, and frozen once terminated because ``describe`` stops
        counting then: rented time ends at termination, and so must the meter."""
        record = self._instances.get(instance_id)
        if record is None:
            return None
        return record["polls"] * self.cost_per_poll

    def _instance(self, instance_id: str, state: InstanceState, *, ssh: bool = False) -> Instance:
        record = self._instances.get(instance_id)
        offer = record["offer"] if record else None
        return Instance(
            provider=self.name,
            instance_id=instance_id,
            state=state,
            gpu_name=offer.gpu_name if offer else "",
            price_per_hour=offer.price_per_hour if offer else 0.0,
            ssh_host="fake.invalid" if ssh else None,  # RFC 2606 reserved — never resolvable
            ssh_port=2222 if ssh else None,
            # The orphan sweep joins on raw["label"] across every backend, fake included.
            raw={"label": record["label"] if record else ""},
        )
