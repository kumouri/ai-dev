"""The launcher's money-and-teardown invariants, proven offline.

The stub provider here is deliberately in-test rather than the pool's fake backend: the provider
backends land independently of the launcher, and these tests must never depend on their timing —
the ~35-line stub below is the whole `CloudProvider` protocol, which is the point of keeping the
protocol at five verbs.

Every test shares one `log` list that the ledger, provider, and remote seams all append to, so
ordering assertions read as the actual story: reserve before any provider call, terminate and
settle on every exit path.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from coryphaeus.cloud.budget import BudgetExceeded, BudgetLedger
from coryphaeus.cloud.launcher import (
    CANONICAL_PHASES,
    ArtifactPullError,
    LauncherError,
    LaunchSpec,
    NoOfferError,
    ProvisionTimeout,
    RemoteFailure,
    WallClockExceeded,
    launch,
    pick_offer,
    worst_case_usd,
)
from coryphaeus.cloud.providers.base import (
    GpuOffer,
    Instance,
    InstanceState,
    ProviderError,
)
from coryphaeus.cloud.remote import build_rsync_cmd, build_scp_cmd, build_ssh_cmd

OFFER = GpuOffer(
    provider="stub",
    offer_id="o-1",
    gpu_name="RTX 4090",
    vram_gb=24,
    price_per_hour=0.40,
    raw={"host_id": 299337, "machine_id": 42748},
)


class StubProvider:
    """Minimal CloudProvider: canned offers, PENDING→RUNNING via describe, everything logged."""

    name = "stub"

    def __init__(self, log, *, offers=(OFFER,), describe_states=(InstanceState.RUNNING,),
                 cost=None):  # fmt: skip
        self.log = log
        self._offers = list(offers)
        self._states = list(describe_states)
        self._cost = cost

    def _instance(self, state: InstanceState) -> Instance:
        running = state is InstanceState.RUNNING
        return Instance(
            provider="stub",
            instance_id="i-001",
            state=state,
            gpu_name="RTX 4090",
            price_per_hour=OFFER.price_per_hour,
            ssh_host="203.0.113.10" if running else None,  # TEST-NET-3: never a real box
            ssh_port=41022 if running else None,
        )

    async def offers(self, *, min_vram_gb, max_price_per_hour):
        self.log.append("offers")
        return list(self._offers)

    async def provision(self, offer, *, image, env, volume_gb, label):
        self.log.append("provision")
        return self._instance(InstanceState.PENDING)

    async def describe(self, instance_id):
        self.log.append("describe")
        state = self._states.pop(0) if len(self._states) > 1 else self._states[0]
        return self._instance(state)

    async def terminate(self, instance_id):
        self.log.append("terminate")  # idempotent no-op, exactly as the protocol demands

    async def cost_so_far(self, instance_id):
        self.log.append("cost_so_far")
        return self._cost


class CapacityFussyProvider(StubProvider):
    """A StubProvider whose provision refuses the first N offers the way a dry market does —
    or, when ``schema_error`` is set, the way a broken payload does (identically, every time)."""

    def __init__(self, log, *, offers, refuse_first=1, schema_error=False, **kwargs):
        super().__init__(log, offers=offers, **kwargs)
        self._refusals_left = refuse_first
        self._schema_error = schema_error

    async def provision(self, offer, *, image, env, volume_gb, label):
        if self._schema_error:
            self.log.append("provision-schema-refused")
            raise ProviderError(
                "stub: provision refused — http 422: additional properties 'x' not allowed"
            )
        if self._refusals_left > 0:
            self._refusals_left -= 1
            self.log.append("provision-capacity-refused")
            raise ProviderError(
                "stub: provision refused — http 500: create pod: "
                "There are no instances currently available"
            )
        return await super().provision(
            offer, image=image, env=env, volume_gb=volume_gb, label=label
        )


class RecordingLedger(BudgetLedger):
    """A real ledger that also stamps reserve/settle into the shared timeline."""

    def __init__(self, path, log, **kwargs):
        super().__init__(path, **kwargs)
        self._log = log

    def reserve(self, *args, **kwargs):
        self._log.append("reserve")
        return super().reserve(*args, **kwargs)

    def settle(self, *args, **kwargs):
        self._log.append("settle")
        return super().settle(*args, **kwargs)


def make_exec(log, *, exit_code=0, delay=0.0, exc: BaseException | None = None):
    async def _exec(instance, command):
        log.append("exec")
        # The readiness knock rides the same exec seam as the payload (one SSH path in
        # production). Failure scripting here applies to the PAYLOAD: the knock — recognizable
        # by its "true && mkdir" prologue — succeeds, exactly like a live box whose sshd is up
        # while the training command is what fails.
        if command.startswith("true"):
            return 0
        if exc is not None:
            raise exc
        if delay:
            await asyncio.sleep(delay)
        return exit_code

    return _exec


def make_pull(log, *, exit_code=0):
    async def _pull(instance, remote_dir, local_dir):
        log.append("pull")
        return exit_code

    return _pull


def make_wedging_pull(log, *, hang_first=999):
    """A pull that wedges (awaits forever) on its first ``hang_first`` calls, then succeeds —
    the 2026-07-31 failure shape, where the transfer process sat six hours after delivering."""
    calls = {"n": 0}

    async def _pull(instance, remote_dir, local_dir):
        calls["n"] += 1
        log.append("pull")
        if calls["n"] <= hang_first:
            await asyncio.Event().wait()
        return 0

    return _pull


def make_push(log, *, exit_code=0):
    async def _push(instance, local_path, remote_path):
        log.append("push")
        return exit_code

    return _push


def spec_for(tmp_path: Path, **overrides) -> LaunchSpec:
    defaults: dict = {
        "image": "example/image:latest",
        "command": "echo hello",
        "label": "coryphaeus-test",
        # 36 wall-clock seconds — roomy for a fast test, so the hard kill only fires when a
        # test sets max_hours tiny on purpose.
        "max_hours": 0.01,
        "volume_gb": 30,
        "provision_timeout_s": 1.0,
        "poll_interval_s": 0.001,
        "artifact_dir": tmp_path / "artifacts",
    }
    defaults.update(overrides)
    return LaunchSpec(**defaults)


def ledger_for(tmp_path: Path, log, *, ceiling: float = 100.0) -> RecordingLedger:
    return RecordingLedger(tmp_path / "budget.jsonl", log, ceiling_usd=ceiling)


async def run_launch(tmp_path, log, provider, spec, ledger, **seams):
    return await launch(
        provider,
        spec,
        ledger,
        run_dir=tmp_path / "run",
        remote_exec=seams.pop("remote_exec", None) or make_exec(log),
        remote_pull=seams.pop("remote_pull", None) or make_pull(log),
        remote_push=seams.pop("remote_push", None) or make_push(log),
        **seams,
    )


def read_events(tmp_path: Path) -> list[dict]:
    path = tmp_path / "run" / "launcher-events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def settle_rows(ledger: BudgetLedger) -> list[dict]:
    return [row for row in ledger.rows() if row["event"] == "settle"]


# --- the launch lifecycle ------------------------------------------------------------------------


async def test_happy_path_runs_the_full_story_in_order(tmp_path):
    """Reserve → provider work → payload → pull → terminate → settle, nothing skipped."""
    log: list[str] = []
    push_file = tmp_path / "questions.jsonl"
    push_file.write_text("{}\n", encoding="utf-8")
    spec = spec_for(tmp_path, push=((push_file, "/workspace/questions.jsonl"),))
    result = await run_launch(tmp_path, log, StubProvider(log), spec, ledger_for(tmp_path, log))

    # The first exec is the readiness knock: it proves sshd answers AND creates the push
    # destinations (/workspace is a RunPod convention, not a law of nature — take 4 paid $0.002
    # to learn a Vast image ships without it).
    assert log == [
        "reserve", "offers", "provision", "describe", "exec", "push", "exec", "pull",
        "terminate", "cost_so_far", "settle",
    ]  # fmt: skip
    assert result.exit_code == 0
    assert result.instance_id == "i-001"
    assert result.actual_usd > 0.0


async def test_provisioned_event_names_the_host_behind_the_instance(tmp_path):
    """Instance ids change per rental; host/machine ids identify a repeat offender. 2026-07-31:
    five same-night provision failures were unattributable because only instance_id was
    recorded — this detail is what CORYPHAEUS_VAST_EXCLUDE gets fed from."""
    log: list[str] = []
    await run_launch(
        tmp_path, log, StubProvider(log), spec_for(tmp_path), ledger_for(tmp_path, log)
    )
    provisioned = next(e for e in read_events(tmp_path) if e["phase"] == "provisioned")
    assert provisioned["detail"]["host_id"] == 299337
    assert provisioned["detail"]["machine_id"] == 42748


async def test_remote_failure_still_terminates_and_settles(tmp_path):
    log: list[str] = []
    ledger = ledger_for(tmp_path, log)
    with pytest.raises(RemoteFailure) as excinfo:
        await run_launch(
            tmp_path, log, StubProvider(log), spec_for(tmp_path), ledger,
            remote_exec=make_exec(log, exit_code=7),
        )  # fmt: skip
    assert excinfo.value.exit_code == 7
    assert log.index("terminate") < log.index("settle")
    # Artifacts are pulled even after a failed payload: terminate wipes the disk, and the
    # checkpoints sitting there are what --resume resumes from.
    assert "pull" in log
    assert len(settle_rows(ledger)) == 1


PRICIER_OFFER = GpuOffer(
    provider="stub",
    offer_id="o-2",
    gpu_name="RTX 3090 Ti",
    vram_gb=24,
    price_per_hour=0.55,
    raw={"host_id": 155125, "machine_id": 43503},
)


async def test_capacity_refusal_falls_through_to_the_next_offer(tmp_path):
    """RunPod live, 2026-08-01: with the driver floor constraining hosts, the cheapest GPU
    class was dry ('no instances currently available') while pricier qualifying classes sat in
    the same offers() result — and the launch failed instead of trying them. Now it tries them.
    The reservation is priced at the CAP, not the offer, so it stays the honest worst case
    across any fallback — asserted, not assumed."""
    log: list[str] = []
    ledger = ledger_for(tmp_path, log)
    provider = CapacityFussyProvider(log, offers=(OFFER, PRICIER_OFFER), refuse_first=1)
    spec = spec_for(tmp_path)
    result = await run_launch(tmp_path, log, provider, spec, ledger)

    assert result.exit_code == 0
    assert result.offer.offer_id == "o-2"  # the pricier survivor
    phases = [e["phase"] for e in read_events(tmp_path)]
    assert phases.count("offer_selected") == 2
    assert phases.count("offer_fallback") == 1
    fallback = next(e for e in read_events(tmp_path) if e["phase"] == "offer_fallback")
    assert fallback["detail"]["skipped_offer_id"] == "o-1"
    assert "no instances currently available" in fallback["detail"]["reason"]
    # One reservation, priced at the cap — identical to a no-fallback run.
    reserves = [r for r in ledger.rows() if r["event"] == "reserve"]
    assert len(reserves) == 1
    assert reserves[0]["estimated_usd"] == pytest.approx(
        round(worst_case_usd(spec.max_price_per_hour, spec.max_hours, spec.volume_gb), 4)
    )


async def test_schema_refusal_does_not_fall_through(tmp_path):
    """A broken payload refuses identically on every offer; falling through would turn one
    clear error into three muddled ones. It must surface immediately — and still settle."""
    log: list[str] = []
    ledger = ledger_for(tmp_path, log)
    provider = CapacityFussyProvider(log, offers=(OFFER, PRICIER_OFFER), schema_error=True)
    with pytest.raises(ProviderError, match="additional properties"):
        await run_launch(tmp_path, log, provider, spec_for(tmp_path), ledger)
    assert log.count("provision-schema-refused") == 1  # one try, no shopping spree
    assert "offer_fallback" not in [e["phase"] for e in read_events(tmp_path)]
    assert ledger.month_spend().reserved_usd == 0.0  # settled on the way out


async def test_all_offers_capacity_refused_is_a_named_launcher_error(tmp_path):
    log: list[str] = []
    ledger = ledger_for(tmp_path, log)
    provider = CapacityFussyProvider(log, offers=(OFFER, PRICIER_OFFER), refuse_first=99)
    with pytest.raises(LauncherError, match="refused for capacity"):
        await run_launch(tmp_path, log, provider, spec_for(tmp_path), ledger)
    assert log.count("provision-capacity-refused") == 2  # both offers tried, bound respected
    assert ledger.month_spend().reserved_usd == 0.0


async def test_provision_timeout_terminates_without_ever_running_the_payload(tmp_path):
    log: list[str] = []
    provider = StubProvider(log, describe_states=(InstanceState.PENDING,))
    spec = spec_for(tmp_path, provision_timeout_s=0.05, poll_interval_s=0.01)
    with pytest.raises(ProvisionTimeout):
        await run_launch(tmp_path, log, provider, spec, ledger_for(tmp_path, log))
    assert "exec" not in log
    assert "terminate" in log
    assert "settle" in log


async def test_wall_clock_hard_kill_terminates_and_settles_at_capped_cost(tmp_path):
    """A hung remote is cut off at max_hours and billed at most the reserved window."""
    log: list[str] = []
    ledger = ledger_for(tmp_path, log)
    spec = spec_for(tmp_path, max_hours=0.5 / 3600.0)  # a 0.5 s window
    with pytest.raises(WallClockExceeded):
        await run_launch(
            tmp_path, log, StubProvider(log), spec, ledger,
            remote_exec=make_exec(log, delay=5.0),  # "hung": would run 10x the window
        )  # fmt: skip
    assert "terminate" in log and "settle" in log
    settled = [e for e in read_events(tmp_path) if e["phase"] == "settled"]
    assert settled[0]["detail"]["basis"] == "wall_clock"
    # The cap: billable hours are clamped to max_hours even though teardown ran after the axe.
    assert settled[0]["detail"]["billable_hours"] == pytest.approx(spec.max_hours)
    # And the reservation is gone — nothing left counting against the month.
    assert ledger.month_spend().reserved_usd == 0.0


async def test_wedged_pull_is_cut_retried_and_artifacts_on_disk_win(tmp_path):
    """The 2026-07-31 shape: rsync delivered everything, then the process sat six hours until
    the max_hours axe billed the whole window. Now: each attempt is cut at pull_timeout_s, the
    retry gets one more chance, and after two timeouts the local disk gets the last word — files
    present → the run proceeds and names exactly what landed, instead of failing a run whose
    deliverable is sitting on disk."""
    log: list[str] = []
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "rollouts.jsonl").write_text('{"cost_usd": 0.1}\n', encoding="utf-8")
    spec = spec_for(tmp_path, pull_timeout_s=0.05)
    result = await run_launch(
        tmp_path, log, StubProvider(log), spec, ledger_for(tmp_path, log),
        remote_pull=make_wedging_pull(log),
    )  # fmt: skip
    assert result.exit_code == 0
    assert log.count("pull") == 2  # cut, retried once, never a third
    assert log.index("terminate") < log.index("settle")
    phases = [e["phase"] for e in read_events(tmp_path)]
    assert phases.count("pull_started") == 2
    assert phases.count("pull_timeout") == 2
    partial = next(e for e in read_events(tmp_path) if e["phase"] == "pull_timeout_partial")
    assert partial["detail"]["files"] == 1
    assert partial["detail"]["bytes"] > 0


async def test_wedged_pull_with_nothing_on_disk_is_a_failed_run(tmp_path):
    """Same wedge, but nothing ever landed: the deliverable is lost and saying otherwise would
    be a lie. Terminate + settle still run — a failed pull must never become a billing leak."""
    log: list[str] = []
    ledger = ledger_for(tmp_path, log)
    spec = spec_for(tmp_path, pull_timeout_s=0.05)
    with pytest.raises(ArtifactPullError, match="timed out twice"):
        await run_launch(
            tmp_path, log, StubProvider(log), spec, ledger,
            remote_pull=make_wedging_pull(log),
        )  # fmt: skip
    assert "terminate" in log and "settle" in log
    assert ledger.month_spend().reserved_usd == 0.0


async def test_pull_that_recovers_on_the_second_attempt_is_a_normal_success(tmp_path):
    """rsync resumes: a near-end wedge on attempt one completes in moments on attempt two."""
    log: list[str] = []
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "model.safetensors").write_bytes(b"weights")
    spec = spec_for(tmp_path, pull_timeout_s=0.05)
    result = await run_launch(
        tmp_path, log, StubProvider(log), spec, ledger_for(tmp_path, log),
        remote_pull=make_wedging_pull(log, hang_first=1),
    )  # fmt: skip
    assert result.exit_code == 0
    assert log.count("pull") == 2
    events = read_events(tmp_path)
    assert [e["phase"] for e in events].count("pull_timeout") == 1
    pulled = next(e for e in events if e["phase"] == "artifacts_pulled")
    assert pulled["detail"]["files"] == 1
    assert pulled["detail"]["bytes"] == len(b"weights")


async def test_keyboard_interrupt_mid_run_terminates_and_settles(tmp_path):
    """Ctrl-C is the exit path people forget; the finally must not."""
    log: list[str] = []
    with pytest.raises(KeyboardInterrupt):
        await run_launch(
            tmp_path, log, StubProvider(log), spec_for(tmp_path), ledger_for(tmp_path, log),
            remote_exec=make_exec(log, exc=KeyboardInterrupt()),
        )  # fmt: skip
    assert "terminate" in log
    assert "settle" in log


async def test_budget_refusal_happens_before_any_provider_call(tmp_path):
    """A gate that needs the network to say no is a gate with a hole in it."""
    log: list[str] = []
    ledger = ledger_for(tmp_path, log, ceiling=0.001)
    with pytest.raises(BudgetExceeded):
        await run_launch(tmp_path, log, StubProvider(log), spec_for(tmp_path), ledger)
    assert log == ["reserve"]  # the refusal itself — and not one provider verb after it
    assert ledger.rows() == []  # a refused reserve writes nothing


async def test_no_qualifying_offer_settles_the_reservation_and_terminates_nothing(tmp_path):
    log: list[str] = []
    ledger = ledger_for(tmp_path, log)
    provider = StubProvider(log, offers=())
    with pytest.raises(NoOfferError):
        await run_launch(tmp_path, log, provider, spec_for(tmp_path), ledger)
    assert "terminate" not in log  # nothing was provisioned, so nothing to kill
    assert settle_rows(ledger)[0]["actual_usd"] == 0.0
    assert ledger.month_spend().total_usd == 0.0  # the month is left untouched


async def test_flaky_terminate_neither_masks_the_error_nor_skips_the_settle(tmp_path):
    """The causal error must survive teardown; the orphan reaper is the belt for the box."""
    log: list[str] = []

    class FlakyTerminate(StubProvider):
        async def terminate(self, instance_id):
            self.log.append("terminate")
            raise ProviderError("terminate API flaked")

    ledger = ledger_for(tmp_path, log)
    with pytest.raises(RemoteFailure):  # not ProviderError — the real failure wins
        await run_launch(
            tmp_path, log, FlakyTerminate(log), spec_for(tmp_path), ledger,
            remote_exec=make_exec(log, exit_code=3),
        )  # fmt: skip
    assert len(settle_rows(ledger)) == 1
    assert any(e["phase"] == "terminate_failed" for e in read_events(tmp_path))


# --- settlement sources --------------------------------------------------------------------------


async def test_settle_uses_provider_cost_when_reported(tmp_path):
    log: list[str] = []
    ledger = ledger_for(tmp_path, log)
    await run_launch(tmp_path, log, StubProvider(log, cost=1.23), spec_for(tmp_path), ledger)
    assert settle_rows(ledger)[0]["actual_usd"] == 1.23
    settled = [e for e in read_events(tmp_path) if e["phase"] == "settled"]
    assert settled[0]["detail"]["basis"] == "provider"


async def test_settle_falls_back_to_wall_clock_times_price(tmp_path):
    log: list[str] = []
    ledger = ledger_for(tmp_path, log)
    await run_launch(tmp_path, log, StubProvider(log, cost=None), spec_for(tmp_path), ledger)
    settled = [e for e in read_events(tmp_path) if e["phase"] == "settled"]
    assert settled[0]["detail"]["basis"] == "wall_clock"
    hours = settled[0]["detail"]["billable_hours"]
    assert 0.0 < hours <= 0.01
    assert settled[0]["detail"]["actual_usd"] == pytest.approx(
        worst_case_usd(OFFER.price_per_hour, hours, 30)
    )


async def test_notify_pushes_start_and_end_and_its_failure_is_never_fatal(tmp_path):
    """docs/CLOUD.md contract: a push at start and end — and a notify failure is logged, never
    raised, because send() shares the exit path with terminate()."""
    log: list[str] = []
    messages: list[str] = []

    async def flaky_notify(text: str) -> None:
        messages.append(text)
        raise RuntimeError("telegram down")

    result = await run_launch(
        tmp_path, log, StubProvider(log), spec_for(tmp_path), ledger_for(tmp_path, log),
        notify=flaky_notify,
    )  # fmt: skip
    assert result.exit_code == 0  # the run's outcome is untouched by the notify failures
    assert len(messages) == 2
    assert "started" in messages[0] and "settled" in messages[1]
    assert sum(1 for e in read_events(tmp_path) if e["phase"] == "notify_failed") == 2


# --- the forensic trail --------------------------------------------------------------------------


async def test_events_file_records_every_canonical_phase_in_order(tmp_path):
    log: list[str] = []
    push_file = tmp_path / "q.jsonl"
    push_file.write_text("{}\n", encoding="utf-8")
    spec = spec_for(tmp_path, push=((push_file, "/workspace/q.jsonl"),))
    await run_launch(tmp_path, log, StubProvider(log), spec, ledger_for(tmp_path, log))

    events = read_events(tmp_path)
    assert all({"phase", "at", "detail"} <= row.keys() for row in events)
    phases = [row["phase"] for row in events]
    indices = [phases.index(phase) for phase in CANONICAL_PHASES]
    assert indices == sorted(indices), f"phases out of order: {phases}"
    assert "pushed" in phases


# --- offer selection -----------------------------------------------------------------------------


def offer(offer_id: str, price: float, vram: int = 24) -> GpuOffer:
    return GpuOffer(
        provider="stub", offer_id=offer_id, gpu_name="gpu", vram_gb=vram, price_per_hour=price
    )


def test_pick_offer_chooses_cheapest_qualifying():
    """Cheaper-but-too-small must lose to cheapest-that-fits — and provider ordering is not
    trusted, so the winner sits mid-list on purpose."""
    offers = [
        offer("pricey", 0.55),
        offer("small-and-cheap", 0.25, vram=16),  # disqualified: below the VRAM floor
        offer("winner", 0.35),
        offer("too-expensive", 0.75, vram=48),  # disqualified: above the price cap
    ]
    picked = pick_offer(offers, min_vram_gb=24, max_price_per_hour=0.60)
    assert picked.offer_id == "winner"


def test_pick_offer_refuses_when_nothing_qualifies():
    with pytest.raises(NoOfferError):
        pick_offer([offer("small", 0.30, vram=16)], min_vram_gb=24, max_price_per_hour=0.60)


# --- SSH/rsync command construction (pure — no network, no binaries) -----------------------------


def test_build_ssh_cmd_arguments():
    argv = build_ssh_cmd(
        "203.0.113.10",
        "echo ok",
        port=41022,
        keyfile=Path("keyfile"),
        known_hosts=Path("kh"),
    )
    assert argv[0] == "ssh"
    joined = " ".join(argv)
    assert "-o BatchMode=yes" in joined  # unattended: a password prompt must fail, not hang
    assert "-o StrictHostKeyChecking=accept-new" in joined  # ephemeral hosts; see remote.py
    assert "-o UserKnownHostsFile=kh" in joined
    assert "-o IdentitiesOnly=yes" in joined
    assert argv[argv.index("-p") + 1] == "41022"
    assert argv[argv.index("-i") + 1] == "keyfile"
    assert argv[-2:] == ["root@203.0.113.10", "echo ok"]


def test_build_ssh_cmd_omits_what_was_not_given():
    argv = build_ssh_cmd("203.0.113.10", "true")
    assert "-p" not in argv and "-i" not in argv
    assert "BatchMode=yes" in argv  # the non-negotiables stay regardless


def test_build_rsync_cmd_arguments():
    argv = build_rsync_cmd(
        "203.0.113.10",
        "/workspace/runs",
        Path("local-artifacts"),
        port=41022,
        keyfile=Path("keyfile"),
    )
    assert argv[0] == "rsync"
    assert "--partial" in argv  # a truncated checkpoint that can resume beats a deleted one
    transport = argv[argv.index("-e") + 1]
    assert transport.startswith("ssh ")
    assert "-p 41022" in transport  # the port rides inside the transport string
    assert "BatchMode=yes" in transport
    # Trailing slash is load-bearing: contents into local_dir, not a nested directory.
    assert argv[-2] == "root@203.0.113.10:/workspace/runs/"
    assert argv[-1] == "local-artifacts"


def test_build_scp_cmd_uses_scp_port_spelling():
    argv = build_scp_cmd(
        ["root@203.0.113.10:/workspace/runs"], "dest", port=41022, keyfile=Path("keyfile")
    )
    assert argv[0] == "scp"
    assert argv[argv.index("-P") + 1] == "41022"  # capital -P: scp is not ssh
    assert "-p" not in argv
    assert argv[-2:] == ["root@203.0.113.10:/workspace/runs", "dest"]


async def test_ssh_that_never_answers_still_terminates_and_settles(tmp_path):
    """A box that runs but cannot be reached (key never installed, sshd absent) must exhaust its
    knocks, terminate, and settle — not hang, not leak."""

    async def deaf_exec(instance, command):
        return 255  # ssh's connection/auth failure code

    log: list[str] = []
    ledger = ledger_for(tmp_path, log)
    spec = spec_for(tmp_path, ssh_ready_attempts=2, ssh_ready_delay_s=0.01)
    with pytest.raises(LauncherError, match="never became ready"):
        await run_launch(tmp_path, log, StubProvider(log), spec, ledger, remote_exec=deaf_exec)
    assert "terminate" in log and "settle" in log


async def test_the_knock_creates_the_push_destinations(tmp_path):
    """/workspace is a RunPod convention, not a law of nature: the knock's mkdir -p is what
    makes the push destination exist on ANY image (take 4 paid $0.002 for this line)."""
    commands: list[str] = []

    async def recording_exec(instance, command):
        commands.append(command)
        return 0

    log: list[str] = []
    await run_launch(
        tmp_path, log, StubProvider(log), spec_for(tmp_path), ledger_for(tmp_path, log),
        remote_exec=recording_exec,
    )  # fmt: skip
    knock = commands[0]
    assert knock.startswith("true")
    assert "mkdir -p /workspace" in knock
