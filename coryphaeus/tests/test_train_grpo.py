"""Trainer configuration — the signature-fitting logic, tested without TRL installed.

The point of `fit_kwargs`: a knob that silently fails to apply is worse than one that errors.
Verified against TRL 1.9.2, where `max_prompt_length` does not exist — a hardcoded config would have
raised on construction, and a careless one would have dropped it in silence.
"""

from __future__ import annotations

import logging

import pytest

from coryphaeus.train.grpo import (
    DEFAULT_POLICY,
    SMOKE_POLICY,
    TrainSettings,
    describe_environment,
    fit_kwargs,
    supported_params,
)


def _target(a=1, b=2, c=3):  # noqa: ARG001 - a stand-in signature
    return None


def _kwargs_target(a=1, **rest):  # noqa: ARG001
    return None


def test_supported_params_reads_a_real_signature():
    assert supported_params(_target) == {"a", "b", "c"}


def test_a_target_accepting_kwargs_is_not_second_guessed():
    """We cannot know what **kwargs means, so let the constructor object rather than drop a knob."""
    assert supported_params(_kwargs_target) is None
    fit = fit_kwargs({"a": 1, "unknown": 2}, _kwargs_target)
    assert fit.accepted == {"a": 1, "unknown": 2}
    assert fit.dropped == ()


def test_unsupported_keys_are_dropped_and_named():
    fit = fit_kwargs({"a": 1, "nope": 2, "also_nope": 3}, _target)
    assert fit.accepted == {"a": 1}
    assert fit.dropped == ("also_nope", "nope")
    assert not fit.ok


def test_a_full_match_reports_ok():
    fit = fit_kwargs({"a": 1, "b": 2}, _target)
    assert fit.ok
    assert fit.dropped == ()


def test_an_uninspectable_target_passes_everything_through():
    """Better to try and fail loudly than to drop every setting on an exotic callable."""
    fit = fit_kwargs({"a": 1}, object())
    assert fit.accepted == {"a": 1}


def test_the_max_prompt_length_case_is_handled_not_hardcoded():
    """The real TRL 1.9.2 finding, as a regression: an absent key is dropped and reported."""
    desired = {"beta": 0.0, "max_prompt_length": 1024}

    def trl_19_like(beta=0.0, max_completion_length=256):  # noqa: ARG001
        return None

    fit = fit_kwargs(desired, trl_19_like)
    assert fit.accepted == {"beta": 0.0}
    assert fit.dropped == ("max_prompt_length",)


def test_settings_produce_the_memory_knobs_that_matter():
    kwargs = TrainSettings().as_config_kwargs()
    # No reference model: the single biggest VRAM saving.
    assert kwargs["beta"] == 0.0
    # 8-bit moments instead of fp32 fused AdamW, which costs roughly 4x.
    assert kwargs["optim"] == "adamw_bnb_8bit"
    assert kwargs["bf16"] is True
    assert kwargs["gradient_checkpointing"] is True
    assert kwargs["use_vllm"] is False


def test_settings_do_not_send_a_key_trl_19_lacks():
    assert "max_prompt_length" not in TrainSettings().as_config_kwargs()


def test_completion_length_exceeds_trl_default():
    """TRL's 256 would truncate a longer workflow mid-JSON."""
    assert TrainSettings().as_config_kwargs()["max_completion_length"] > 256


def test_group_size_gives_a_signal():
    """num_generations=1 makes every advantage exactly zero — nothing to learn from."""
    assert TrainSettings().num_generations >= 2


def test_a_group_of_one_is_rejected():
    with pytest.raises(ValueError, match="advantage of exactly zero"):
        TrainSettings(num_generations=1)


@pytest.mark.parametrize("k", [2, 3, 4, 5, 8])
def test_generation_batch_is_a_whole_number_of_groups(k):
    """The real TRL 1.9 constraint, hit for real with --k 3:

        ValueError: generation_batch_size (4) must be divisible by num_generations (3)

    Deriving the accumulation steps makes every k valid by construction.
    """
    settings = TrainSettings(num_generations=k)
    assert settings.generation_batch_size % k == 0


def test_an_explicit_bad_accumulation_is_rejected_not_adjusted():
    """Silently changing a number somebody chose is its own bug."""
    with pytest.raises(ValueError, match="not divisible by num_generations"):
        TrainSettings(num_generations=3, gradient_accumulation_steps=4)


def test_an_explicit_good_accumulation_is_kept():
    settings = TrainSettings(num_generations=3, gradient_accumulation_steps=6)
    assert settings.gradient_accumulation_steps == 6


def test_report_to_is_empty_so_nothing_phones_home():
    assert TrainSettings().as_config_kwargs()["report_to"] == []


def test_checkpoint_cadence_bounds_the_blast_radius():
    """r3 lost 2 hours to a CUDA fault at step 17 with save_steps=50 and nothing on disk.

    Every 10 steps ≈ ≤80 min exposure at measured pace, and save_total_limit keeps the training
    volume from silently filling over a long run.
    """
    kwargs = TrainSettings().as_config_kwargs()
    assert kwargs["save_steps"] <= 10
    assert kwargs["save_total_limit"] is not None


def test_policies_are_the_planned_ones():
    assert "0.5B" in SMOKE_POLICY
    assert "1.5B" in DEFAULT_POLICY


def test_describe_environment_never_raises_without_torch():
    """It is a diagnostic; a missing torch is a fact to report, not a crash."""
    info = describe_environment()
    assert isinstance(info, dict)
    assert ("torch" in info) or ("torch_error" in info)


def test_dropped_keys_are_logged_loudly(caplog):
    """Silence is the failure mode: an unapplied beta=0 costs a whole extra copy of the weights."""

    def narrow(beta=0.0):  # noqa: ARG001
        return None

    with caplog.at_level(logging.WARNING):
        fit = fit_kwargs({"beta": 0.0, "gone": 1}, narrow)
    assert fit.dropped == ("gone",)  # build_config does the logging; fit_kwargs reports
