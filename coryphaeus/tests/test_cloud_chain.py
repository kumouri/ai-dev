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
