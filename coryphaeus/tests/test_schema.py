"""Validation rules. Each rule exists to turn a runtime failure into a named reward signal."""

from __future__ import annotations

import pytest

from coryphaeus.schema import MAX_STEPS, SELF, Step, WorkflowError, build, validate

WORKERS = ("tiny", "mid", "big")


def step(index: int, worker: str = "tiny", deps: tuple[int, ...] = ()) -> Step:
    return Step(index=index, subtask=f"do thing {index}", worker=worker, deps=deps)


def test_build_defaults_final_to_last_step():
    wf = build([step(1), step(2)], known_workers=WORKERS)
    assert wf.final == 2
    assert wf.final_step.index == 2


def test_self_is_always_an_allowed_worker():
    wf = build([step(1, worker=SELF)], known_workers=WORKERS)
    assert wf.workers_used() == (SELF,)


def test_workers_used_is_first_use_order_and_deduplicated():
    wf = build([step(1, "big"), step(2, "tiny"), step(3, "big")], known_workers=WORKERS)
    assert wf.workers_used() == ("big", "tiny")


@pytest.mark.parametrize(
    ("steps", "final", "reason"),
    [
        ([], 1, "no_steps"),
        ([step(i) for i in range(1, MAX_STEPS + 2)], 1, "too_many_steps"),
        ([Step(index=2, subtask="x", worker="tiny")], 1, "bad_index"),
        ([Step(index=1, subtask="   ", worker="tiny")], 1, "empty_subtask"),
        ([Step(index=1, subtask="x", worker="nope")], 1, "unknown_worker"),
        ([step(1), Step(index=2, subtask="x", worker="tiny", deps=(9,))], 2, "bad_dep"),
        ([Step(index=1, subtask="x", worker="tiny", deps=(1,))], 1, "self_dep"),
        ([step(1), step(2), Step(index=3, subtask="x", worker="tiny", deps=(3,))], 3, "self_dep"),
        ([step(1, deps=(2,)), step(2)], 1, "forward_dep"),
        ([step(1)], 5, "bad_final"),
        ([step(1)], 0, "bad_final"),
    ],
)
def test_invalid_workflows_report_a_stable_reason(steps, final, reason):
    with pytest.raises(WorkflowError) as exc:
        validate(steps, final, known_workers=WORKERS)
    assert exc.value.reason == reason


def test_forward_dep_rejection_makes_cycles_impossible():
    """Backward-only deps mean index order is always executable — no cycle can be expressed."""
    with pytest.raises(WorkflowError) as exc:
        validate([step(1, deps=(2,)), step(2, deps=(1,))], 2, known_workers=WORKERS)
    assert exc.value.reason == "forward_dep"


def test_to_dict_round_trips_the_shape():
    wf = build([step(1), step(2, deps=(1,))], known_workers=WORKERS)
    assert wf.to_dict() == {
        "steps": [
            {"index": 1, "subtask": "do thing 1", "worker": "tiny", "deps": []},
            {"index": 2, "subtask": "do thing 2", "worker": "tiny", "deps": [1]},
        ],
        "final": 2,
    }
