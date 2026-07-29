"""GRPO trainer wiring.

TRL is imported **lazily**, so `coryphaeus.train` stays importable — and the whole reward path stays
testable — on a machine with no torch and no GPU.

**Why this filters config keys instead of passing them straight through.** `GRPOConfig` and
`GRPOTrainer` have renamed and moved parameters between TRL releases. Hardcoding a name means the
module breaks on upgrade; passing unknown keys means a `TypeError` deep in a constructor. So the
desired settings are declared as a plain dict, matched against the *installed* signature, and
anything unsupported is reported by name. A knob that silently failed to apply is the dangerous
case — `beta=0` not taking effect quietly costs ~3 GB of VRAM to a reference model you thought you
had disabled.

This is not hypothetical. Verified against **TRL 1.9.2 / transformers 5.14.1** (2026-07-29):
`max_prompt_length` does not exist there at all, so a hardcoded config would have raised on
construction. `reward_funcs` *is* the right parameter name, and `GRPOTrainer` also exposes
`rollout_func` and `environment_factory` — a route worth considering later if owning generation as
well as scoring turns out to be cleaner than the reward-function seam.

**On `mask_truncated_completions` (left at its default, `False`).** TRL can drop completions that
hit the length cap out of the loss. Tempting — a truncated emission is garbage — but it is the wrong
choice here: a workflow cut off mid-JSON fails to parse and scores zero, and that zero is how the
policy learns to emit workflows that *fit*. Masking it would remove the lesson, which is the same
argument `docs/adr/0001` makes for never repairing a malformed workflow.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: VRAM notes for a 24 GB card, bf16, no reference model (beta=0):
#:   params ~2 bytes/param, grads ~2, adamw_8bit states ~2 (vs ~12 for fp32 AdamW).
#: A 1.5B policy lands around 12-14 GB with activations; 0.5B around 5-6 GB. A reference model adds
#: another copy of the weights, which is why beta defaults to 0 here.
DEFAULT_POLICY = "Qwen/Qwen2.5-1.5B-Instruct"
SMOKE_POLICY = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass(slots=True)
class TrainSettings:
    """Everything the trainer needs, independent of TRL's parameter naming."""

    model: str = DEFAULT_POLICY
    output_dir: Path = Path("runs/grpo")
    #: Rollouts per question. GRPO's advantage is relative *within* the group, so 1 gives no
    #: signal. Bounded by the provider's concurrency budget, not by VRAM — measure before raising.
    num_generations: int = 4
    per_device_train_batch_size: int = 1
    #: 0 means "derive it" — see :meth:`__post_init__`. TRL requires the generation batch to be a
    #: whole number of groups, and deriving it is how any ``k`` works without arithmetic homework.
    gradient_accumulation_steps: int = 0
    learning_rate: float = 1e-6
    max_steps: int = -1
    num_train_epochs: float = 1.0
    #: A ≤5-step workflow with a fenced JSON block fits comfortably; TRL's own default is 256, which
    #: would truncate the longer ones. Truncation is *not* masked out of the loss — see the note
    #: below on ``mask_truncated_completions``.
    max_completion_length: int = 512
    temperature: float = 0.9
    #: 0.0 disables the KL term *and* the reference model — the biggest VRAM saving available. Set
    #: explicitly rather than relied on: it happens to be TRL 1.9's default too, but a default that
    #: silently changes would cost a whole extra copy of the weights.
    beta: float = 0.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    #: TRL defaults to ``adamw_torch_fused``, whose fp32 moments cost roughly 4x this. On a 24 GB
    #: card shared with nothing else that is still the difference between fitting and not.
    optim: str = "adamw_bnb_8bit"
    logging_steps: int = 1
    save_steps: int = 50
    seed: int = 0
    use_vllm: bool = False
    report_to: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Resolve and check the batch/group arithmetic TRL enforces.

        TRL 1.9 computes ``generation_batch_size = per_device_train_batch_size *
        gradient_accumulation_steps`` and rejects a value that is not a whole number of groups:

            ValueError: generation_batch_size (4) must be divisible by num_generations (3)

        Observed for real with ``--k 3``. Deriving the accumulation steps from ``num_generations``
        makes any ``k`` valid by construction; an explicit value is validated instead of quietly
        adjusted, because silently changing a number someone chose is its own bug.
        """
        if self.num_generations < 2:
            raise ValueError(
                f"num_generations={self.num_generations} gives every rollout an advantage of "
                "exactly zero — GRPO's signal is relative *within* a group, so there is nothing "
                "to learn from a group of one."
            )
        if self.gradient_accumulation_steps <= 0:
            self.gradient_accumulation_steps = self.num_generations
        generation_batch = self.per_device_train_batch_size * self.gradient_accumulation_steps
        if generation_batch % self.num_generations != 0:
            raise ValueError(
                f"per_device_train_batch_size ({self.per_device_train_batch_size}) * "
                f"gradient_accumulation_steps ({self.gradient_accumulation_steps}) = "
                f"{generation_batch}, which is not divisible by num_generations "
                f"({self.num_generations}). TRL requires whole groups. Set "
                f"gradient_accumulation_steps to a multiple of {self.num_generations} "
                f"(e.g. {self.num_generations}), or leave it at 0 to have it derived."
            )

    @property
    def generation_batch_size(self) -> int:
        """What TRL will compute, exposed so a run can record it."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def as_config_kwargs(self) -> dict:
        return {
            "output_dir": str(self.output_dir),
            "num_generations": self.num_generations,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "num_train_epochs": self.num_train_epochs,
            "max_completion_length": self.max_completion_length,
            "temperature": self.temperature,
            "beta": self.beta,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optim": self.optim,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "seed": self.seed,
            "use_vllm": self.use_vllm,
            "report_to": self.report_to,
        }


def supported_params(target: object) -> set[str] | None:
    """Parameter names the installed ``target`` accepts.

    Returns ``None`` for "anything goes" — either the target takes ``**kwargs``, in which case we
    cannot tell what is meaningful, or its signature is unreadable. Both cases must **not** filter:
    letting a constructor object loudly beats dropping a knob that would have worked.
    """
    try:
        signature = inspect.signature(target)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return None
    return set(signature.parameters)


@dataclass(frozen=True, slots=True)
class ConfigFit:
    """What of the desired settings the installed TRL could actually take."""

    accepted: dict
    dropped: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.dropped


def fit_kwargs(desired: dict, target: object) -> ConfigFit:
    """Match desired settings against a callable's real signature.

    Returns the accepted subset plus the names that were dropped, so the caller can *say* what it
    could not apply. Silence here is the failure mode worth avoiding.
    """
    valid = supported_params(target)
    if valid is None:
        return ConfigFit(accepted=dict(desired), dropped=())
    accepted = {k: v for k, v in desired.items() if k in valid}
    dropped = tuple(sorted(set(desired) - set(accepted)))
    return ConfigFit(accepted=accepted, dropped=dropped)


def build_config(settings: TrainSettings):
    """Construct a ``GRPOConfig``, reporting any setting the installed TRL does not support."""
    from trl import GRPOConfig  # noqa: PLC0415

    fit = fit_kwargs(settings.as_config_kwargs(), GRPOConfig)
    if fit.dropped:
        logger.warning(
            "this TRL build does not accept: %s — those are NOT applied. Check whether any "
            "of them mattered (beta=0 silently not applying costs ~3 GB to a reference model).",
            ", ".join(fit.dropped),
        )
    return GRPOConfig(**fit.accepted), fit


def build_trainer(settings: TrainSettings, dataset, reward_fn):
    """Assemble a ``GRPOTrainer``.

    ``reward_fn`` is the callable from :func:`coryphaeus.train.bridge.make_reward_fn` — it executes
    workflows against the worker pool, so a training step's wall-clock is dominated by the network
    rather than by backprop. That is expected; see ``docs/ROADMAP.md``.
    """
    from trl import GRPOTrainer  # noqa: PLC0415

    config, fit = build_config(settings)

    desired = {
        "model": settings.model,
        "args": config,
        "train_dataset": dataset,
        "reward_funcs": [reward_fn],
    }
    trainer_fit = fit_kwargs(desired, GRPOTrainer.__init__)
    required = ("model", "args", "train_dataset", "reward_funcs")
    missing = [k for k in required if k not in trainer_fit.accepted]
    if missing:
        accepts = supported_params(GRPOTrainer.__init__)
        raise RuntimeError(
            f"the installed TRL's GRPOTrainer does not take {missing}. It accepts: "
            f"{sorted(accepts) if accepts else 'anything (**kwargs)'}. Pin a TRL version "
            "guessing — see docs/ROADMAP.md phase 3."
        )
    return GRPOTrainer(**trainer_fit.accepted), fit


def describe_environment() -> dict:
    """A record of what actually ran. Cheap, and the first thing you want when a run looks odd."""
    info: dict = {}
    try:
        import torch  # noqa: PLC0415

        info["torch"] = torch.__version__
        info["cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            info["device"] = torch.cuda.get_device_name(0)
            info["capability"] = list(torch.cuda.get_device_capability(0))
            info["vram_free_gib"] = round(free / 2**30, 2)
            info["vram_total_gib"] = round(total / 2**30, 2)
    except Exception as exc:  # noqa: BLE001 - a missing torch is a fact, not a crash
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import trl  # noqa: PLC0415

        info["trl"] = trl.__version__
    except Exception as exc:  # noqa: BLE001
        info["trl_error"] = f"{type(exc).__name__}: {exc}"
    return info
