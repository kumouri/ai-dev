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


def test_provision_timeout_default_survives_the_cold_pull():
    """Pinned because it was learned twice: seven same-night 'junk hosts' all died pending at
    exactly the old 1500s default — honest cold image pulls guillotined at 96%. If this default
    shrinks again, it should be a deliberate decision staring at this test, not a tidy-up."""
    args = _cloud_run.parse_args(["--provider", "fake"])
    assert args.provision_timeout >= 2700.0


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
    """(20 probe + 200 full) steps x k=4 x 5 calls, every call at or-dear's prices — the settle
    reports reality; the reservation must bound it."""
    per_call = (0.135 * 600 + 0.40 * 1024) / 1_000_000
    expected = round(per_call * 220 * 4 * 5, 4)
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
    assert probe == round(full * 20 / 220, 4)
