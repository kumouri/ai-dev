"""The command chain the launcher hands a rented box — pinned because the box debugs at rental
rates.

Born from a near-miss (2026-07-31): the manifest went multi-provider while an overnight sequence
was mid-flight, and an unfiltered registry build on the box — which is shipped the Featherless key
alone — would have crashed the gated full run at its first registry construction. The contract:
**every train stage names the providers whose keys the launcher actually pushes.** The pool a box
may use is exactly the pool it can authenticate to, and the chain must say so out loud.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# cloud_run lives in scripts/, which is not a package — load it the way Python actually can.
_spec = importlib.util.spec_from_file_location(
    "cloud_run",
    Path(__file__).resolve().parents[1] / "scripts" / "cloud_run.py",
)
_cloud_run = importlib.util.module_from_spec(_spec)
sys.modules["cloud_run"] = _cloud_run
_spec.loader.exec_module(_cloud_run)
build_chain = _cloud_run.build_chain


def chain_args(**overrides) -> argparse.Namespace:
    base = dict(
        chain="full",
        k=4,
        steps=200,
        probe_steps=40,
        max_hours=8.0,
        gate_threshold=0.2,
        questions_file="coryphaeus/runs/calibrated.jsonl",
        worker_providers="featherless",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_every_train_stage_names_the_shipped_providers():
    chain = build_chain(chain_args(), "run-label")
    stages = chain.split(" && ")
    train_stages = [s for s in stages if "train_grpo.py" in s]
    assert len(train_stages) == 2  # probe + full
    for stage in train_stages:
        assert "--providers featherless" in stage, stage


def test_the_gate_stage_needs_no_provider_filter():
    """The gate reads telemetry; it builds no registry and must not grow worker knobs."""
    chain = build_chain(chain_args(chain="validate"), "run-label")
    gate_stage = next(s for s in chain.split(" && ") if "gate_zero_std.py" in s)
    assert "--providers" not in gate_stage


def test_probe_only_chain_carries_the_filter_too():
    assert "--providers featherless" in build_chain(chain_args(chain="probe"), "run-label")


def test_probe_and_full_share_one_label_derived_seed():
    """train_grpo's --seed defaults to 0, so before this every probe graded the SAME question
    window (the 2026-08-01 and 2026-08-20 gates measured one 20-question sample twice). The
    chain must pass a seed derived from the label: probe and full share it (the resumed run
    continues the same sampling plan), different labels draw different windows, and the same
    label reproduces its draw."""

    def seeds(chain: str) -> set[str]:
        return {
            stage.split("--seed ")[1].split()[0]
            for stage in chain.split(" && ")
            if "train_grpo.py" in stage
        }

    first = seeds(build_chain(chain_args(), "run-label"))
    assert len(first) == 1
    assert first == seeds(build_chain(chain_args(), "run-label"))  # reproducible
    assert first != seeds(build_chain(chain_args(), "another-label"))  # varies by launch


def test_remote_command_sources_the_runpod_env_file_before_anything():
    """RunPod images write the container env — the provision-injected worker keys — to
    /etc/rp_environment and source it only from ~/.bashrc, which a login shell never reads:
    a live pod's SSH sessions showed zero worker keys while the file sat there (2026-08-01,
    verified by hand before writing this line). Absent files are a no-op, so the sourcing is
    provider-agnostic and must come FIRST."""
    command = _cloud_run.build_remote_command(
        repo_url="https://example.com/repo.git", repo_ref="develop", chain="echo hi"
    )
    assert "[ -f /etc/rp_environment ] && . /etc/rp_environment; " in command
    assert command.index("rp_environment") < command.index("git clone")


def test_provision_timeout_matches_the_bimodal_boot_distribution():
    """Measured over 30 rentals (2026-08-01): 16 hosts reached ssh_ready in 40-90 SECONDS, 14
    never came up at all, nothing in between. So the window's job is not patience — it is
    failing fast enough to re-roll, while staying an order of magnitude above every observed
    success. (An earlier default of 2700s came from the opposite reading — 'cold pulls
    guillotined at 1500s' — which the 2700s window itself disproved by producing identical
    failures 20 minutes later.)"""
    args = _cloud_run.parse_args(["--provider", "fake"])
    assert 600.0 <= args.provision_timeout <= 1800.0


def test_exit_codes_separate_verdicts_from_infrastructure():
    """Retry wrappers branch on these, and conflating them cost real money twice on
    2026-08-01: a gate FAIL and a dead host both exited 1, so one wrapper relaunched a finished
    retrain and another re-ran a probe doomed to fail identically."""
    assert (_cloud_run.EXIT_OK, _cloud_run.EXIT_INFRA) == (0, 1)
    assert _cloud_run.EXIT_USAGE == 2
    assert _cloud_run.EXIT_REFUSED == 3
    assert _cloud_run.EXIT_PAYLOAD == 4
    # The distinction that matters: "retry may help" and "the box gave you an answer" must
    # never be the same number.
    assert _cloud_run.EXIT_INFRA != _cloud_run.EXIT_PAYLOAD


def test_the_chain_carries_whatever_providers_were_selected():
    chain = build_chain(chain_args(worker_providers="openrouter"), "run-label")
    for stage in (s for s in chain.split(" && ") if "train_grpo.py" in s):
        assert "--providers openrouter" in stage


# --- the selected providers drive keys and money too ---------------------------------------------


def test_unknown_worker_provider_is_refused_by_name():
    import pytest

    with pytest.raises(SystemExit, match="ollama"):
        _cloud_run.parse_worker_providers("featherless,ollama")


def test_worker_env_ships_exactly_the_selected_keys():
    cfg = argparse.Namespace(featherless_api_key="fl-key", openrouter_api_key="or-key")
    assert _cloud_run.worker_env(cfg, ("featherless",)) == {"FEATHERLESS_API_KEY": "fl-key"}
    assert _cloud_run.worker_env(cfg, ("featherless", "openrouter")) == {
        "FEATHERLESS_API_KEY": "fl-key",
        "OPENROUTER_API_KEY": "or-key",
    }


def test_worker_env_refuses_before_renting_when_a_selected_key_is_absent():
    import pytest

    cfg = argparse.Namespace(featherless_api_key="fl-key", openrouter_api_key=None)
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        _cloud_run.worker_env(cfg, ("openrouter",))


PAID_ENTRIES = [
    {"provider": "featherless", "name": "qwen25-14b"},
    {"provider": "openrouter", "name": "or-cheap", "price_in_per_m": 0.02, "price_out_per_m": 0.04},
    {"provider": "openrouter", "name": "or-dear", "price_in_per_m": 0.135, "price_out_per_m": 0.40},
]


def test_flat_rate_selections_reserve_nothing():
    assert (
        _cloud_run.token_reserve_usd(
            PAID_ENTRIES, chain="full", steps=200, k=4, providers=("featherless",)
        )
        == 0.0
    )


def test_token_reserve_prices_the_worst_case_at_the_priciest_seat():
    """(40 probe + 200 full) steps x k=4 x 5 calls, every call at or-dear's prices — the settle
    reports reality; the reservation must bound it."""
    per_call = (0.135 * 600 + 0.40 * 1024) / 1_000_000
    expected = round(per_call * 240 * 4 * 5, 4)
    got = _cloud_run.token_reserve_usd(
        PAID_ENTRIES, chain="full", steps=200, k=4, providers=("featherless", "openrouter")
    )
    assert got == expected > 0


def test_probe_chains_reserve_only_the_probe_steps():
    full = _cloud_run.token_reserve_usd(
        PAID_ENTRIES, chain="full", steps=200, k=4, providers=("openrouter",)
    )
    probe = _cloud_run.token_reserve_usd(
        PAID_ENTRIES, chain="probe", steps=200, k=4, providers=("openrouter",)
    )
    assert probe == round(full * 40 / 240, 4)


def test_probe_steps_drive_both_the_chain_and_the_reservation():
    """The gate's sample size is one knob: what the probe trains is what the reservation
    prices. At n=20 three straight verdicts (30%, 25%, exactly 20%) each sat one group from
    flipping — the default is 40 so the gate's word means something."""
    import pytest

    chain = build_chain(chain_args(), "run-label")
    probe_stage = next(s for s in chain.split(" && ") if "train_grpo.py" in s)
    assert "--max-steps 40" in probe_stage
    shorter = _cloud_run.token_reserve_usd(
        PAID_ENTRIES, chain="probe", steps=200, k=4, providers=("openrouter",), probe_steps=20
    )
    default = _cloud_run.token_reserve_usd(
        PAID_ENTRIES, chain="probe", steps=200, k=4, providers=("openrouter",)
    )
    # approx, not ==: both values are independently rounded to 4 places, so doubling the
    # rounded 20-step figure can differ from the rounded 40-step figure by a ten-thousandth.
    assert default == pytest.approx(shorter * 2, abs=0.0002)
    assert _cloud_run.parse_args(["--provider", "fake"]).probe_steps == 40
