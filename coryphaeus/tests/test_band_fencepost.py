"""The band fencepost — a regression pin for the v2 refine's silent shell exclusion.

The v2 pass shipped with band (0.17, 0.83) while its docstring promised "1-5 of 6". But
1/6 = 0.1667 < 0.17 and 5/6 = 0.8333 > 0.83, so BOTH boundary shells fell through a float gap —
the kept set silently became "2-4 of 6", and the display histogram (float-range bins with matching
holes) summed to 103 of 153 probed questions. Doc said one thing, arithmetic did another.

The corrected default is (0.15, 0.70): the hard-contested shell (1/6) is IN, the near-easy shell
(5/6) is OUT — r4's dead steps were the too-easy side, so the band skews hard on purpose.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "calibrate_questions",
    Path(__file__).resolve().parents[1] / "scripts" / "calibrate_questions.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["calibrate_questions"] = _mod
_spec.loader.exec_module(_mod)


def _default_band() -> tuple[float, float]:
    args = _mod.parse_args(["--refine", "x.jsonl"])
    return tuple(args.band)


def test_the_hard_contested_shell_is_inside_the_default_band():
    """1 success in 6 = a question the weak worker barely cracks — routing gradient lives here."""
    lo, hi = _default_band()
    assert lo <= 1 / 6 <= hi


def test_all_middle_shells_are_inside():
    lo, hi = _default_band()
    for k in (1, 2, 3, 4):
        assert lo <= k / 6 <= hi, f"{k}/6 fell out of the band"


def test_the_near_easy_shell_is_excluded_on_purpose():
    """5/6 is the r4 failure mode wearing a thin disguise; the band skews hard deliberately."""
    lo, hi = _default_band()
    assert 5 / 6 > hi


def test_the_extremes_are_excluded():
    lo, hi = _default_band()
    assert 0.0 < lo and 6 / 6 > hi


def test_the_original_fencepost_would_fail_these():
    """The exact v2 bug, pinned: (0.17, 0.83) excludes both shells its docs claimed to keep."""
    lo, hi = 0.17, 0.83
    assert not (lo <= 1 / 6 <= hi)
    assert not (lo <= 5 / 6 <= hi)
