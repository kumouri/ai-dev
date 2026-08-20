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
import contextlib
import json
import os
import shlex
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from . import remote as remote_mod
from .budget import BudgetLedger
from .providers.base import CloudProvider, GpuOffer, Instance, InstanceState, ProviderError

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
    "ssh_ready",
    "payload_started",
    "payload_finished",
    "pull_started",
    "artifacts_pulled",
    "terminate_requested",
    "terminated",
    "settled",
    "launch_finished",
)

#: Ceiling on ONE pull attempt, not the whole retrieval. 90 minutes, not 30: the healthiest pull
#: of 2026-07-31 took 78 minutes for ~6 GB over a marketplace uplink — while the wedged one sat
#: SIX HOURS after transferring everything, eating the rest of the max_hours window and turning
#: the hard kill into the pull's de-facto (and very expensive) timeout. One retry follows a
#: timeout: the pull is rsync underneath, so a near-end wedge resumes and completes in minutes.
#: Override via ``CORYPHAEUS_PULL_TIMEOUT_S`` (wired in cloud_run.py).
DEFAULT_PULL_TIMEOUT_S = 5400.0


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
#: Contract: ``(instance, remote_dir, local_dir, *, excludes=()) -> int`` — ``excludes`` are
#: rsync-style patterns the transfer may skip (the scp fallback ignores them, documented in
#: ``remote.make_rsync_pull``). Typed ``...`` because Callable cannot spell a keyword arg.
RemotePull = Callable[..., Awaitable[int]]
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


def rank_offers(
    offers: Sequence[GpuOffer], *, min_vram_gb: int, max_price_per_hour: float
) -> list[GpuOffer]:
    """Every qualifying offer, cheapest first — the capacity fallback's menu.

    Re-sorts instead of trusting the provider's "cheapest first" promise — a backend ordering bug
    should cost nothing worse than a redundant sort. Ties keep listing order (sort is stable).
    """
    qualifying = sorted(
        (o for o in offers if o.vram_gb >= min_vram_gb and o.price_per_hour <= max_price_per_hour),
        key=lambda o: o.price_per_hour,
    )
    if not qualifying:
        raise NoOfferError(
            f"no offer with >= {min_vram_gb} GB VRAM at <= ${max_price_per_hour:.2f}/hr "
            f"({len(offers)} offers seen). Raise --max-price or lower --min-vram deliberately."
        )
    return qualifying


def pick_offer(
    offers: Sequence[GpuOffer], *, min_vram_gb: int, max_price_per_hour: float
) -> GpuOffer:
    """Cheapest offer meeting the floor specs (the head of :func:`rank_offers`)."""
    return rank_offers(offers, min_vram_gb=min_vram_gb, max_price_per_hour=max_price_per_hour)[0]


#: How many qualifying offers one launch may try before conceding the market has no capacity.
#: Three covers "the cheapest class is dry" without turning a systemic outage into a shopping
#: spree of doomed rentals.
MAX_OFFER_ATTEMPTS = 3

#: Provider phrasings that mean "the market has none of THESE right now" — the only refusal
#: class where trying the next offer is correct. Conservative on purpose: schema and auth errors
#: repeat identically on every offer, and falling through on them would turn one clear error
#: into three muddled ones. Source: RunPod's create-pod 500 (live, 2026-08-01).
_CAPACITY_REFUSAL_MARKS = (
    "no instances currently available",
    "no instances available",
)


def _is_capacity_refusal(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(mark in text for mark in _CAPACITY_REFUSAL_MARKS)


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
    #: Knocks on the door after "running": sshd wakes and onstart installs keys AFTER the
    #: container starts, so the first contact retries rather than fails. 12 x 10s = two minutes
    #: of patience, which also absorbs slow container init on marketplace hosts.
    ssh_ready_attempts: int = 12
    ssh_ready_delay_s: float = 10.0
    poll_interval_s: float = 10.0
    #: Where artifacts accumulate on the box (runs, checkpoints, telemetry).
    remote_artifact_dir: str = "/workspace/runs"
    #: Local destination for the pull; None skips artifact retrieval entirely.
    artifact_dir: Path | None = None
    #: Per-attempt ceiling on the artifact pull — see DEFAULT_PULL_TIMEOUT_S for the sizing story.
    pull_timeout_s: float = DEFAULT_PULL_TIMEOUT_S
    #: Wedge detector: if the local artifact footprint grows by ZERO bytes for this long
    #: mid-attempt, the transfer is dead, not slow — cut it and retry. Sized from the measured
    #: failure (2026-08-20): a wedged ssh channel sat 1h43m moving nothing, and the retry that
    #: followed completed in 13.7 s. Patience is not a strategy against a dead channel.
    pull_stall_window_s: float = 180.0
    #: How often the wedge detector samples the footprint (a directory walk — cheap).
    pull_progress_poll_s: float = 15.0
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


def _artifact_footprint(artifact_dir: Path) -> tuple[int, int]:
    """(files, bytes) actually on local disk — the honest measure of what a pull delivered.

    Needed because the pull *process* and the pull *outcome* can disagree: the 2026-07-31 wedge
    sat six hours after rsync had already written all 5.7 GB. What is on disk is the ground
    truth; the transfer's exit status is only a claim about it.
    """
    files = 0
    size = 0
    for path in artifact_dir.rglob("*"):
        if path.is_file():
            files += 1
            size += path.stat().st_size
    return files, size


async def _pull_artifacts(
    events: EventLog,
    remote_pull: RemotePull,
    instance: Instance,
    spec: LaunchSpec,
    exit_code: int,
) -> None:
    """Pull artifacts down — after failures too, and never for longer than the pull window.

    terminate wipes the box's disk, and on a --resume chain the checkpoints sitting there are the
    difference between resuming at hour 7 and re-paying hours 0-7. So: payload succeeded → a pull
    failure is fatal (the artifacts are the deliverable); payload failed → the pull is best-effort
    forensics and must not mask the real error.

    Each attempt is bounded twice: ``spec.pull_timeout_s`` is the hard ceiling, and the wedge
    detector cuts an attempt early when the local footprint stops growing for
    ``spec.pull_stall_window_s`` — a dead channel is not a slow transfer, and the measured wedge
    (2026-08-20) sat 1h43m doing nothing its 13.7 s retry didn't. A cut attempt gets ONE retry
    (rsync resumes, so partial progress is kept), and after the second the local disk gets the
    last word: artifacts present → the run proceeds with a ``pull_timeout_partial`` event naming
    exactly what landed; nothing present on a successful payload → the deliverable is lost and
    that is a failed run. On a FAILED payload the pull is scoped to telemetry only
    (``checkpoints/`` excluded): the evidence comes home, the dead run's shards do not.
    """
    if spec.artifact_dir is None:
        return
    # Gate-aware scope (2026-08-20): a failed payload's weights are dead spend. v5 hauled
    # 27.6 GB of shards off a run its own gate had just refused — 64% of that run's cost,
    # spent after the verdict was in. Telemetry IS a failed run's deliverable (it is the
    # refilter's evidence and the reason the probe existed); the checkpoints are not. On
    # success everything comes home as before — a passed probe's checkpoint is the resume
    # seed and the whole point.
    excludes = () if exit_code == 0 else ("checkpoints/",)
    scope = "full" if exit_code == 0 else "telemetry-only"
    for attempt in (1, 2):
        events.emit("pull_started", attempt=attempt, timeout_s=spec.pull_timeout_s, scope=scope)
        started = time.monotonic()
        pull_task = asyncio.ensure_future(
            remote_pull(instance, spec.remote_artifact_dir, spec.artifact_dir, excludes=excludes)
        )
        stalled = False
        try:
            async with asyncio.timeout(spec.pull_timeout_s):
                # Wedge detector: rsync --partial writes as it transfers, so footprint growth
                # is ground truth for progress. Zero new bytes for pull_stall_window_s means a
                # dead channel, and cutting it beats waiting — the measured wedge sat 1h43m
                # doing nothing that its 13.7 s retry didn't.
                last_size = _artifact_footprint(spec.artifact_dir)[1]
                last_change = time.monotonic()
                while True:
                    done, _ = await asyncio.wait({pull_task}, timeout=spec.pull_progress_poll_s)
                    if done:
                        break
                    size = _artifact_footprint(spec.artifact_dir)[1]
                    if size != last_size:
                        last_size, last_change = size, time.monotonic()
                    elif time.monotonic() - last_change >= spec.pull_stall_window_s:
                        stalled = True
                        break
        except TimeoutError:
            pull_task.cancel()
            with contextlib.suppress(BaseException):
                await pull_task
            events.emit(
                "pull_timeout",
                attempt=attempt,
                elapsed_s=round(time.monotonic() - started, 1),
            )
            continue
        if stalled:
            pull_task.cancel()
            with contextlib.suppress(BaseException):
                await pull_task
            events.emit(
                "pull_stalled",
                attempt=attempt,
                elapsed_s=round(time.monotonic() - started, 1),
                bytes_on_disk=last_size,
            )
            continue
        try:
            pull_rc = pull_task.result()
        except Exception as exc:
            if exit_code == 0:
                raise ArtifactPullError(f"artifact pull raised: {exc!r}") from exc
            events.emit("artifact_pull_failed", error=repr(exc))
            return
        if pull_rc == 0:
            files, size = _artifact_footprint(spec.artifact_dir)
            events.emit(
                "artifacts_pulled",
                dest=str(spec.artifact_dir),
                elapsed_s=round(time.monotonic() - started, 1),
                files=files,
                bytes=size,
            )
            return
        if exit_code == 0:
            raise ArtifactPullError(f"artifact pull exited {pull_rc}")
        events.emit("artifact_pull_failed", exit_code=pull_rc)
        return

    # Two attempts ended by timeout or stall. The transfer never answered — but the files it
    # may have already written are real either way, and terminate is about to wipe the only
    # other copy.
    files, size = _artifact_footprint(spec.artifact_dir)
    if files and exit_code == 0:
        events.emit(
            "pull_timeout_partial",
            dest=str(spec.artifact_dir),
            files=files,
            bytes=size,
        )
        return
    if exit_code == 0:
        raise ArtifactPullError(
            f"artifact pull timed out or stalled twice ({spec.pull_timeout_s:.0f}s ceiling, "
            f"{spec.pull_stall_window_s:.0f}s stall window) with nothing in the local artifact "
            "dir — the deliverable is lost"
        )
    events.emit(
        "artifact_pull_failed",
        error=(
            f"timed out or stalled twice ({spec.pull_timeout_s:.0f}s ceiling); "
            f"{files} file(s) on disk"
        ),
    )


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
        # The whole qualifying menu, not just the head: with driver floors constraining hosts,
        # the cheapest GPU class can be genuinely dry ("no instances currently available",
        # RunPod live 2026-08-01) while pricier qualifying classes sit right behind it. Every
        # candidate is at or below max_price_per_hour, so the reservation above — priced at the
        # CAP, not the offer — stays the honest worst case across any fallback.
        candidates = rank_offers(
            await provider.offers(
                min_vram_gb=spec.min_vram_gb, max_price_per_hour=spec.max_price_per_hour
            ),
            min_vram_gb=spec.min_vram_gb,
            max_price_per_hour=spec.max_price_per_hour,
        )
        capacity_refusals: list[ProviderError] = []
        for rank, candidate in enumerate(candidates[:MAX_OFFER_ATTEMPTS], start=1):
            events.emit(
                "offer_selected",
                offer_id=candidate.offer_id,
                gpu=candidate.gpu_name,
                vram_gb=candidate.vram_gb,
                price_per_hour=candidate.price_per_hour,
                rank=rank,
            )
            try:
                instance = await provider.provision(
                    candidate,
                    image=spec.image,
                    env=spec.env,
                    volume_gb=spec.volume_gb,
                    label=spec.label,
                )
            except ProviderError as exc:
                if not _is_capacity_refusal(exc):
                    raise
                capacity_refusals.append(exc)
                events.emit(
                    "offer_fallback",
                    skipped_offer_id=candidate.offer_id,
                    gpu=candidate.gpu_name,
                    price_per_hour=candidate.price_per_hour,
                    reason=str(exc)[:200],
                )
                continue
            offer = candidate
            break
        if instance is None or offer is None:
            tried = ", ".join(
                f"{c.gpu_name} ${c.price_per_hour:.2f}/hr" for c in candidates[:MAX_OFFER_ATTEMPTS]
            )
            raise LauncherError(
                f"all {len(capacity_refusals)} tried offer(s) refused for capacity ({tried}) — "
                f"last: {capacity_refusals[-1]}"
            )
        provisioned_at = time.monotonic()
        # Host identity rides the event because instance ids change per rental: 2026-07-31's
        # five same-night provision failures were unattributable from instance_id alone, and
        # a repeat-offender host is invisible without it (feeds CORYPHAEUS_VAST_EXCLUDE).
        offer_raw = offer.raw if isinstance(offer.raw, dict) else {}
        events.emit(
            "provisioned",
            instance_id=instance.instance_id,
            state=instance.state.value,
            host_id=offer_raw.get("host_id"),
            machine_id=offer_raw.get("machine_id"),
        )

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

                # First SSH contact doubles as the readiness gate, and it prepares the push
                # destinations. Two live lessons in one line (2026-07-31, take 4, $0.002):
                # "running" means the CONTAINER started, not that sshd has finished waking or
                # that onstart has written authorized_keys yet — so this retries over a couple
                # of minutes instead of failing on the first knock. And the destination
                # directories may simply not exist: /workspace is a RunPod convention, not a
                # law of nature — a Vast image has no such directory until we make one.
                ready_cmd = " && ".join(
                    ["true"]
                    + [
                        f"mkdir -p {shlex.quote(str(PurePosixPath(remote).parent))}"
                        for _, remote in spec.push
                    ]
                    + [f"mkdir -p {shlex.quote(spec.remote_artifact_dir)}"]
                )
                ssh_ready = False
                for knock in range(spec.ssh_ready_attempts):
                    if knock:
                        await asyncio.sleep(spec.ssh_ready_delay_s)
                    if await remote_exec(instance, ready_cmd) == 0:
                        ssh_ready = True
                        events.emit("ssh_ready", knocks=knock + 1)
                        break
                if not ssh_ready:
                    raise LauncherError(
                        f"ssh never became ready after {spec.ssh_ready_attempts} attempts — the "
                        "box ran but could not be reached (key not installed, or sshd absent "
                        "from the image)"
                    )

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
