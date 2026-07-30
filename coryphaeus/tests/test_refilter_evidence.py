"""Evidence refiltering — the free alternative to re-probing after a gate FAIL.

The recency mode exists because of a measured miss: the any-run rule kept 3 of r5's 5 dead
questions on the strength of stale r4 evidence (step-0 policy, pre-self-fix prompt). Under drift,
old contestedness is history, not prediction.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "refilter_from_evidence",
    Path(__file__).resolve().parents[1] / "scripts" / "refilter_from_evidence.py",
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["refilter_from_evidence"] = _mod
_spec.loader.exec_module(_mod)
classify = _mod.classify


def _write_run(tmp_path: Path, name: str, groups: dict[str, list[list[float]]]) -> Path:
    run = tmp_path / name
    run.mkdir()
    with (run / "rollouts.jsonl").open("w", encoding="utf-8") as fh:
        for qid, gs in groups.items():
            for g in gs:
                for score in g:
                    fh.write(json.dumps({"question_id": qid, "score": score}) + "\n")
    return run


def test_mixed_anywhere_wins_by_default(tmp_path):
    _write_run(tmp_path, "20260730T0100Z-a", {"q1": [[1, 0, 1, 1]]})  # mixed early
    _write_run(tmp_path, "20260730T0200Z-b", {"q1": [[1, 1, 1, 1]]})  # unanimous later
    mixed, dead = classify([str(tmp_path / "*")], 4)
    assert "q1" in mixed and "q1" not in dead


def test_recency_lets_the_latest_run_overrule_history(tmp_path):
    """The r5 situation: mixed at step 0, unanimous at step 15 — the policy outgrew it."""
    _write_run(tmp_path, "20260730T0100Z-a", {"q1": [[1, 0, 1, 1]]})
    _write_run(tmp_path, "20260730T0200Z-b", {"q1": [[1, 1, 1, 1]]})
    mixed, dead = classify([str(tmp_path / "*")], 4, recency=True)
    assert "q1" in dead and "q1" not in mixed


def test_recency_still_keeps_a_freshly_mixed_question(tmp_path):
    _write_run(tmp_path, "20260730T0100Z-a", {"q1": [[1, 1, 1, 1]]})
    _write_run(tmp_path, "20260730T0200Z-b", {"q1": [[1, 0, 1, 1]]})
    mixed, dead = classify([str(tmp_path / "*")], 4, recency=True)
    assert "q1" in mixed


def test_unmeasured_questions_appear_in_neither_set(tmp_path):
    _write_run(tmp_path, "20260730T0100Z-a", {"q1": [[1, 0, 1, 1]]})
    mixed, dead = classify([str(tmp_path / "*")], 4)
    assert "q2" not in mixed and "q2" not in dead


def test_incomplete_groups_do_not_classify(tmp_path):
    _write_run(tmp_path, "20260730T0100Z-a", {"q1": [[1, 0]]})  # only 2 rollouts
    mixed, dead = classify([str(tmp_path / "*")], 4)
    assert "q1" not in mixed and "q1" not in dead


def test_multiple_groups_within_one_run_all_count(tmp_path):
    _write_run(tmp_path, "20260730T0100Z-a", {"q1": [[1, 1, 1, 1], [1, 0, 1, 1]]})
    mixed, dead = classify([str(tmp_path / "*")], 4)
    assert "q1" in mixed  # the second group was mixed


def test_chronology_comes_from_directory_names(tmp_path):
    """Timestamped run-dir names ARE the clock; glob order must not matter."""
    _write_run(tmp_path, "20260730T0900Z-late", {"q1": [[1, 1, 1, 1]]})
    _write_run(tmp_path, "20260730T0100Z-early", {"q1": [[1, 0, 1, 1]]})
    mixed, dead = classify([str(tmp_path / "*")], 4, recency=True)
    assert "q1" in dead  # the LATE run (unanimous) rules, despite glob returning early-first
