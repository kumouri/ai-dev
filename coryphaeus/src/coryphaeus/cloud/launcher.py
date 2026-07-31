"""Rent → run → retrieve → terminate, with the teardown guaranteed by structure.

The one invariant this module exists to enforce: **every path out of a launch ends at terminate
and settle** — success, remote failure, provision timeout, wall-clock hard kill, KeyboardInterrupt,
a provider API flake, all of them. That lives in a single ``try/finally``, not in per-case
handlers, because a billing leak is exactly the kind of bug that hides in the exit path nobody
wrote a handler for.

Money ordering is the ledger's contract: ``reserve()`` runs **before the first provider call** at
the worst case the launch could possibly cost, and ``settle()`` runs in the same ``finally`` as
terminate. A budget refusal therefore costs zero API calls, and a crash between the two leaves an
over-counting reservation — the failure mode that costs vigilance, never money.

Everything that would need a network in tests is an injected seam: the provider (the
``CloudProvider`` protocol), remote exec/pull/push (SSH — see ``remote.py``), and notify. The
orchestration itself is thereby fully testable offline.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import remote as remote_mod
from .budget import BudgetLedger
from .providers.base import CloudProvider, GpuOffer, Instance, InstanceState

#: Storage estimate used in the worst-case reservation and the wall-clock cost fallback.
#: ≈ $0.10/GB-month — the ballpark both target providers publish — prorated to the hour.
#: An estimate on purpose: real storage billing is provider-shaped, and cost_so_far() overrides
#: this whenever the provider reports actuals.
VOLUME_USD_PER_GB_HOUR = 0.10 / 730.0

#: Every phase a fully successful launch emits, in order. The conditional extras
#: (pushed, *_failed, wall_clock_exceeded) appear only on the paths that earn them.
CANONICAL_PHASES = (
    "launch_started",
    "reserved",
    "offer_selected",
    "provisioned",
    "running",
    "payload_started",
    "payload_finished",
    "artifacts_pulled",
    "terminate_requested",
    "terminated",
    "settled",
    "launch_finished",
)


class LauncherError(RuntimeError):
    """Base for launch failures the caller can reason about."""


class NoOfferError(LauncherError):
    """No offer met the floor specs. Message says what was seen, so the fix is a number change."""


class ProvisionTimeout(LauncherError):
    """The box never reached RUNNING inside the provision window."""


class WallClockExceeded(LauncherError):
    """The max_wall_hours hard kill fired — a hung remote cannot bill past the reservation."""


class RemoteFailure(LauncherError):
    """The payload ran and exited nonzero."""

    def __init__(self, exit_code: int, message: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class ArtifactPullError(LauncherError):
    """The payload succeeded but its artifacts could not be retrieved — that is a failed run:
    the artifacts *are* the deliverable."""


#: Run ``command`` on the instance, streaming output, returning the exit code.
RemoteExec = Callable[[Instance, str], Awaitable[int]]
#: Pull ``remote_dir`` down into a local directory, returning the transfer's exit code.
RemotePull = Callable[[Instance, str, Path], Awaitable[int]]
#: Push one local file up to ``remote_path``, returning the transfer's exit code.
RemotePush = Callable[[Instance, Path, str], Awaitable[int]]
#: Human-facing one-liner at the end of a run (Telegram, etc.). Failures are logged, never fatal.
Notify = Callable[[str], Awaitable[None]]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class EventLog:
    """Phase transitions to stdout AND an append-only JSONL — the forensic trail.

    Each row is ``{"phase": ..., "at": ..., "detail": {...}}``. Opened, fsynced, and closed per
    event: the whole point is that a launcher that dies mid-run leaves every phase it reached on
    disk, and at ~a dozen events per multi-hour run the durability is free.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def emit(self, phase: str, **detail: object) -> None:
        at = _utc_now()
        print(f"[{at}] {phase}" + (f" {json.dumps(detail, default=str)}" if detail else ""),
              flush=True)  # fmt: skip
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"phase": phase, "at": at, "detail": detail}, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def pick_offer(
    offers: Sequence[GpuOffer], *, min_vram_gb: int, max_price_per_hour: float
) -> GpuOffer:
    """Cheapest offer meeting the floor specs.

    Re-sorts instead of trusting the provider's "cheapest first" promise — a backend ordering bug
    should cost nothing worse than a redundant ``min()``. Ties break to the earliest listed.
    """
    qualifying = [
        o for o in offers if o.vram_gb >= min_vram_gb and o.price_per_hour <= max_price_per_hour
    ]
    if not qualifying:
        raise NoOfferError(
            f"no offer with >= {min_vram_gb} GB VRAM at <= ${max_price_per_hour:.2f}/hr "
            f"({len(offers)} offers seen). Raise --max-price or lower --min-vram deliberately."
        )
    return min(qualifying, key=lambda o: o.price_per_hour)


def worst_case_usd(price_per_hour: float, max_hours: float, volume_gb: int) -> float:
    """The reservation estimate: the full window at the given rate, plus storage for the window."""
    return max_hours * (price_per_hour + volume_gb * VOLUME_USD_PER_GB_HOUR)


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    """Everything one launch needs, provider aside. Frozen: a spec that mutates mid-run would
    desynchronize the reservation from the run it reserved for."""

    image: str
    #: One shell command string; the CLI composes the whole chain into it (see cloud_run.py).
    command: str
    #: Doubles as the ledger reservation id AND the provider label — one join key, so the orphan
    #: reaper can match provider instances against ledger rows.
    label: str
    #: Instance environment, injected via the provider — the channel for secrets, precisely so
    #: they never appear in the SSH command line, the events log, or a process listing.
    env: dict[str, str] = field(default_factory=dict)
    min_vram_gb: int = 24
    max_price_per_hour: float = 0.60
    #: Drives BOTH the reservation estimate and the wall-clock hard kill — one number, so the
    #: kill can never let a run bill past what was reserved.
    max_hours: float = 8.0
    volume_gb: int = 30
    #: 25 min, not 10: a ~10 GB image at the Vast backend's own 200 Mbps network floor is
    #: ~6.5 min of transfer before extraction, and marketplace hosts pull cold more often than
    #: not — two live 600s timeouts on pending boxes (2026-07-31, $0.02 each) sized this.
    provision_timeout_s: float = 1500.0
    poll_interval_s: float = 10.0
    #: Where artifacts accumulate on the box (runs, checkpoints, telemetry).
    remote_artifact_dir: str = "/workspace/runs"
    #: Local destination for the pull; None skips artifact retrieval entirely.
    artifact_dir: Path | None = None
    #: (local file, remote path) pairs copied up before the payload — for run inputs like
    #: calibrated question files that live under gitignored runs/ and so cannot be cloned.
    push: tuple[tuple[Path, str], ...] = ()


@dataclass(frozen=True, slots=True)
class LaunchResult:
    offer: GpuOffer
    instance_id: str
    exit_code: int
    actual_usd: float
    run_dir: Path
    events_path: Path


async def _try_notify(events: EventLog, notify: Notify | None, text: str) -> None:
    """Notify without ever being the reason a run fails: send() shares the exit path with
    terminate(), so a notification failure is logged and swallowed."""
    if notify is None:
        return
    try:
        await notify(text)
        events.emit("notified")
    except Exception as exc:
        events.emit("notify_failed", error=repr(exc))


async def _wait_until_running(
    provider: CloudProvider, instance: Instance, *, timeout_s: float, poll_s: float
) -> Instance:
    """Poll describe() until RUNNING with an SSH endpoint, bounded by the provision window."""
    deadline = time.monotonic() + timeout_s
    current = instance
    while True:
        if current.state is InstanceState.RUNNING and current.ssh_host:
            return current
        if current.state in (InstanceState.STOPPED, InstanceState.TERMINATED):
            raise LauncherError(
                f"instance {current.instance_id} reached {current.state.value} before RUNNING — "
                "the box died on the pad"
            )
        if time.monotonic() >= deadline:
            raise ProvisionTimeout(
                f"instance {current.instance_id} not RUNNING after {timeout_s:.0f}s "
                f"(last state: {current.state.value})"
            )
        await asyncio.sleep(poll_s)
        # UNKNOWN falls through to keep polling: a describe hiccup is the provider's API
        # blinking, not the box dying, and the deadline above bounds the patience.
        current = await provider.describe(instance.instance_id)


async def _pull_artifacts(
    events: EventLog,
    remote_pull: RemotePull,
    instance: Instance,
    spec: LaunchSpec,
    exit_code: int,
) -> None:
    """Pull artifacts down — after failures too.

    terminate wipes the box's disk, and on a --resume chain the checkpoints sitting there are the
    difference between resuming at hour 7 and re-paying hours 0-7. So: payload succeeded → a pull
    failure is fatal (the artifacts are the deliverable); payload failed → the pull is best-effort
    forensics and must not mask the real error.
    """
    if spec.artifact_dir is None:
        return
    try:
        pull_rc = await remote_pull(instance, spec.remote_artifact_dir, spec.artifact_dir)
    except Exception as exc:
        if exit_code == 0:
            raise ArtifactPullError(f"artifact pull raised: {exc!r}") from exc
        events.emit("artifact_pull_failed", error=repr(exc))
        return
    if pull_rc == 0:
        events.emit("artifacts_pulled", dest=str(spec.artifact_dir))
    elif exit_code == 0:
        raise ArtifactPullError(f"artifact pull exited {pull_rc}")
    else:
        events.emit("artifact_pull_failed", exit_code=pull_rc)


async def launch(
    provider: CloudProvider,
    spec: LaunchSpec,
    ledger: BudgetLedger,
    *,
    run_dir: Path,
    remote_exec: RemoteExec | None = None,
    remote_pull: RemotePull | None = None,
    remote_push: RemotePush | None = None,
    notify: Notify | None = None,
) -> LaunchResult:
    """One full cloud run. Raises on any failure; terminates and settles on *every* exit."""
    run_dir.mkdir(parents=True, exist_ok=True)
    events = EventLog(run_dir / "launcher-events.jsonl")

    if remote_exec is None or remote_pull is None or remote_push is None:
        # Real SSH seams. known_hosts is per-run (see remote.py's host-key policy).
        keyfile = remote_mod.default_keyfile()
        known_hosts = run_dir / "known_hosts"
        remote_exec = remote_exec or remote_mod.make_ssh_exec(
            keyfile=keyfile, known_hosts=known_hosts
        )
        remote_pull = remote_pull or remote_mod.make_rsync_pull(
            keyfile=keyfile, known_hosts=known_hosts
        )
        remote_push = remote_push or remote_mod.make_scp_push(
            keyfile=keyfile, known_hosts=known_hosts
        )

    # Reserve at the *cap*, not the picked offer's price: pick_offer can only choose at or below
    # max_price_per_hour, so the cap is the honest worst case — and pricing the reservation from
    # spec alone lets the budget refuse BEFORE the first provider round-trip. A budget gate that
    # needs the network to say no is a gate with a hole in it.
    worst = worst_case_usd(spec.max_price_per_hour, spec.max_hours, spec.volume_gb)
    events.emit(
        "launch_started",
        label=spec.label,
        provider=provider.name,
        worst_case_usd=round(worst, 4),
        max_hours=spec.max_hours,
    )
    ledger.reserve(spec.label, worst, note=f"{provider.name} launch")  # raises BudgetExceeded
    events.emit("reserved", reservation_id=spec.label, estimated_usd=round(worst, 4))
    await _try_notify(
        events,
        notify,
        f"[coryphaeus] {spec.label}: started on {provider.name} "
        f"(worst case ${worst:.2f}, {spec.max_hours:g}h cap)",
    )

    offer: GpuOffer | None = None
    instance: Instance | None = None
    provisioned_at: float | None = None
    exit_code: int | None = None
    settled: dict[str, float] = {}

    try:
        offer = pick_offer(
            await provider.offers(
                min_vram_gb=spec.min_vram_gb, max_price_per_hour=spec.max_price_per_hour
            ),
            min_vram_gb=spec.min_vram_gb,
            max_price_per_hour=spec.max_price_per_hour,
        )
        events.emit(
            "offer_selected",
            offer_id=offer.offer_id,
            gpu=offer.gpu_name,
            vram_gb=offer.vram_gb,
            price_per_hour=offer.price_per_hour,
        )

        instance = await provider.provision(
            offer,
            image=spec.image,
            env=spec.env,
            volume_gb=spec.volume_gb,
            label=spec.label,
        )
        provisioned_at = time.monotonic()
        events.emit("provisioned", instance_id=instance.instance_id, state=instance.state.value)

        try:
            # The hard kill. Covers everything billable — boot wait, payload, artifact pull — so
            # a hang anywhere in the window still ends at the finally below within max_hours.
            async with asyncio.timeout(spec.max_hours * 3600.0):
                instance = await _wait_until_running(
                    provider,
                    instance,
                    timeout_s=spec.provision_timeout_s,
                    poll_s=spec.poll_interval_s,
                )
                events.emit("running", ssh_host=instance.ssh_host, ssh_port=instance.ssh_port)

                for local_path, remote_path in spec.push:
                    push_rc = await remote_push(instance, local_path, remote_path)
                    if push_rc != 0:
                        raise LauncherError(
                            f"push of {local_path.name} to {remote_path} exited {push_rc}"
                        )
                    events.emit("pushed", file=local_path.name, remote_path=remote_path)

                events.emit("payload_started", command=spec.command)
                exit_code = await remote_exec(instance, spec.command)
                events.emit("payload_finished", exit_code=exit_code)

                await _pull_artifacts(events, remote_pull, instance, spec, exit_code)

                if exit_code != 0:
                    raise RemoteFailure(
                        exit_code, f"remote payload exited {exit_code} on {instance.instance_id}"
                    )
        except TimeoutError as exc:
            # No artifact pull on this path, deliberately: the kill exists to stop billing, and
            # rsyncing gigabytes off a hung box would extend exactly what it must end.
            events.emit("wall_clock_exceeded", max_hours=spec.max_hours)
            raise WallClockExceeded(
                f"hard kill after {spec.max_hours:g}h — terminating {instance.instance_id}"
            ) from exc
    finally:
        # EVERY exit path lands here: return, RemoteFailure, ProvisionTimeout, WallClockExceeded,
        # KeyboardInterrupt, provider errors. Teardown is unconditional and each step is guarded
        # so a flaky terminate can never mask the causal error or skip the settle.
        error = sys.exc_info()[1]
        if instance is not None:
            events.emit("terminate_requested", instance_id=instance.instance_id)
            try:
                # The protocol guarantees idempotence — terminating an already-dead instance is a
                # no-op — so this needs no state check first.
                await provider.terminate(instance.instance_id)
                events.emit("terminated", instance_id=instance.instance_id)
            except Exception as exc:
                # Logged, not raised: the orphan reaper (cloud_run.py --terminate-orphans) is the
                # belt for exactly this, and the unsettled-leaning ledger keeps the pressure on.
                events.emit("terminate_failed", instance_id=instance.instance_id, error=repr(exc))

        actual = 0.0
        basis = "nothing_provisioned"
        extra: dict[str, float] = {}
        if instance is not None:
            reported: float | None = None
            try:
                reported = await provider.cost_so_far(instance.instance_id)
            except Exception as exc:
                events.emit("cost_query_failed", error=repr(exc))
            if reported is not None:
                actual, basis = float(reported), "provider"
            else:
                assert provisioned_at is not None  # set in the same statement as instance
                # Capped at max_hours: terminate follows the hard kill within seconds, so real
                # billing past the cap means terminate failed — which the events row and the
                # still-running instance surface loudly, rather than a silent bigger settle.
                hours = min((time.monotonic() - provisioned_at) / 3600.0, spec.max_hours)
                assert offer is not None  # provision consumed it
                actual = worst_case_usd(offer.price_per_hour, hours, spec.volume_gb)
                basis = "wall_clock"
                extra["billable_hours"] = hours
        settled["actual_usd"] = actual  # the result reports cost even if the settle write fails
        try:
            ledger.settle(spec.label, actual, note=basis)
            events.emit("settled", actual_usd=actual, basis=basis, **extra)
        except Exception as exc:
            # Swallowed on purpose: an unsettled reservation over-counts (fail-closed, in money's
            # favor) and a settle bug must not replace the causal exception mid-propagation.
            events.emit("settle_failed", error=repr(exc))

        outcome = "finished ok" if error is None else f"failed: {error!r}"
        await _try_notify(
            events, notify, f"[coryphaeus] {spec.label}: {outcome} — settled ${actual:.2f}"
        )

        events.emit("launch_finished", ok=error is None, error=repr(error) if error else None)

    # Only the success path reaches this — every failure raised out of the try above.
    assert offer is not None and instance is not None and exit_code is not None
    return LaunchResult(
        offer=offer,
        instance_id=instance.instance_id,
        exit_code=exit_code,
        actual_usd=settled.get("actual_usd", 0.0),
        run_dir=run_dir,
        events_path=events.path,
    )
