"""Catalogue → worker spec, and the property that matters: the catalog must be routable.

The catalog line is the *only* thing the conductor knows about a worker. A missing size or a missing
specialist tag is not cosmetic — it is routing information silently absent, which shows up later as
"the conductor didn't learn to route" when in fact it was never told anything.
"""

from __future__ import annotations

import pytest

from coryphaeus.catalog import derive_tags, manifest_entry, params_from_model_class, short_name
from coryphaeus.pools import build_remote_registry, load_manifest


@pytest.mark.parametrize(
    ("model_class", "expected"),
    [
        ("qwen25-7b", 7.0),
        ("qwen25-32b", 32.0),
        ("llama33-70b", 70.0),
        ("qwen2-72b", 72.0),
        ("qwen2-0b5", 0.5),  # 'b' is the decimal point
        ("qwen3-1b7", 1.7),
        ("tinyllama-1b1", 1.1),
        ("gemma4-31b", 31.0),
        ("kimi-linear-48b", 48.0),
        ("rwkv5-7b", 7.0),
        (None, None),
        ("", None),
        ("something-odd", None),
    ],
)
def test_params_from_model_class(model_class, expected):
    assert params_from_model_class(model_class) == expected


@pytest.mark.parametrize(
    ("model_id", "params_b", "expected"),
    [
        ("Qwen/Qwen2.5-Math-7B-Instruct", 7.0, ("math", "fast")),
        ("Qwen/Qwen2.5-14B-Instruct", 14.0, ("balanced",)),
        ("Qwen/Qwen2.5-32B-Instruct", 32.0, ("strong",)),
        ("Qwen/Qwen2.5-Coder-7B", 7.0, ("code", "fast")),
        ("some/model", None, ()),
    ],
)
def test_derive_tags(model_id, params_b, expected):
    assert derive_tags(model_id, params_b) == expected


def test_concurrency_cost_is_not_tagged():
    """Deliberate: the reward does not price concurrency cost yet, so a tag would be wasted."""
    assert "serializing" not in derive_tags("Qwen/Qwen2.5-72B-Instruct", 72.0)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("Qwen/Qwen2.5-Math-7B-Instruct", "qwen25-math-7b"),
        ("Qwen/Qwen2.5-32B-Instruct", "qwen25-32b"),
        ("Qwen/Qwen3-32B", "qwen3-32b"),
        ("meta-llama/Llama-3.3-70B-Instruct", "llama33-70b"),
    ],
)
def test_short_name(model_id, expected):
    assert short_name(model_id) == expected


def test_units_come_from_the_provider_not_from_size():
    """A 32B costs 2 units. The size heuristic said 4, which halves concurrency for nothing."""
    entry = manifest_entry(
        {"id": "Qwen/Qwen2.5-32B-Instruct", "model_class": "qwen25-32b", "concurrency_cost": 2}
    )
    assert entry["units"] == 2
    assert entry["params_b"] == 32.0
    assert entry["name"] == "qwen25-32b"


def test_missing_concurrency_cost_defaults_to_one_unit():
    entry = manifest_entry({"id": "x/y-7B", "model_class": "qwen25-7b"})
    assert entry["units"] == 1


# --- the property that actually matters ---------------------------------------------------------


def test_pinned_manifest_is_routable():
    """Every pinned worker must produce a catalog line with a real size.

    '(unknown size)' silently strips the conductor's only basis for routing. This test is the reason
    the manifest carries params_b at all.
    """
    entries = load_manifest()
    if not entries:
        pytest.skip("no pinned manifest in this checkout")
    unsized = [e["name"] for e in entries if not e.get("params_b")]
    assert not unsized, f"these workers would read 'unknown size': {unsized}"


def test_pinned_manifest_declares_units_for_every_worker():
    entries = load_manifest()
    if not entries:
        pytest.skip("no pinned manifest in this checkout")
    assert all(int(e.get("units", 0)) >= 1 for e in entries)


def test_openrouter_seats_carry_pin_price_and_allowed_quantization():
    """Every OpenRouter seat must say WHERE it runs, WHAT it costs, and HOW it is quantized.

    The pin is what makes a calibration describe one served system; the prices are what feed the
    token-spend ledger (a missing price makes a paid worker look free); and the quantization set is
    the locked policy — bf16 preferred, fp8 allowed, int4-class seats excluded.
    """
    entries = [e for e in load_manifest() if e.get("provider") == "openrouter"]
    if not entries:
        pytest.skip("no openrouter seats pinned in this checkout")
    for e in entries:
        assert e.get("pin"), f"{e['name']}: unpinned openrouter seat"
        assert e.get("price_in_per_m") is not None, f"{e['name']}: no input price"
        assert e.get("price_out_per_m") is not None, f"{e['name']}: no output price"
        assert e.get("quantization") in {"bf16", "fp16", "fp8"}, (
            f"{e['name']}: quantization {e.get('quantization')!r} outside the allowed set"
        )


def test_training_pool_excludes_serializing_workers(monkeypatch):
    """max_units keeps the account able to run several rollouts at once.

    A single 4-unit worker consumes a 4-unit budget outright, so every other rollout waits on it.
    """
    entries = load_manifest()
    if not entries:
        pytest.skip("no pinned manifest in this checkout")
    monkeypatch.setenv("FEATHERLESS_API_KEY", "test-key-not-used-offline")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-used-offline")
    from coryphaeus import config

    config.settings.cache_clear()
    try:
        full = build_remote_registry()
        training = build_remote_registry(max_units=2)
        assert len(training) < len(full)
        assert all(spec.units <= 2 for spec in training.specs())
        # And the catalog it hands the conductor is still informative.
        assert "unknown size" not in training.catalog_text()
    finally:
        config.settings.cache_clear()
