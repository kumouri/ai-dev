"""GRPO training: the dataset builder and the async→sync reward bridge.

Deliberately importable without torch or TRL installed — the dataset and bridge are pure-Python and
fully testable offline, so the whole reward path can be verified before a GPU is involved.
"""

from .bridge import BridgeClosed, RewardBridge, make_reward_fn
from .dataset import TrainRow, build_row, build_rows, describe, to_hf_dataset

__all__ = [
    "BridgeClosed",
    "RewardBridge",
    "TrainRow",
    "build_row",
    "build_rows",
    "describe",
    "make_reward_fn",
    "to_hf_dataset",
]
