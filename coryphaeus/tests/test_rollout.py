"""Rollout groups, GRPO advantages, and the end-to-end loop on a fake pool."""

from __future__ import annotations

import json

from coryphaeus.datasets.loaders import Question
from coryphaeus.policy import ScriptedConductor
from coryphaeus.reward import Score
from coryphaeus.rollout import RolloutRecord, group_advantages, one_rollout, rollout_group
from coryphaeus.workers import Governor, WorkerRegistry, fake_spec
from coryphaeus.workers.fake import FakeWorker


async def _no_sleep(_seconds: float) -> None:
    return None


def _fake_governor() -> Governor:
    return Governor(budgets={"fake": 16}, sleeper=_no_sleep, base_delay=0.0, jitter=0.0)


def _record(value: float) -> RolloutRecord:
    return RolloutRecord(
        question_id="q",
        rollout_index=0,
        arm="conductor",
        raw="",
        score=Score(value, bool(value), None, "1"),
    )


def test_advantages_are_group_relative_and_zero_mean():
    advantages = group_advantages([_record(1.0), _record(0.0), _record(1.0), _record(0.0)])
    assert abs(sum(advantages)) < 1e-12
    assert advantages[0] > 0 > advantages[1]


def test_a_degenerate_group_has_no_signal():
    """All-correct or all-wrong yields zeros — no signal when every rollout agrees."""
    assert group_advantages([_record(1.0)] * 4) == [0.0] * 4
    assert group_advantages([_record(0.0)] * 4) == [0.0] * 4


def test_single_rollout_group_has_no_advantage():
    assert group_advantages([_record(1.0)]) == [0.0]


async def test_end_to_end_correct_rollout_scores_one(registry, governor, questions, emit):
    """A workflow routed to a worker that knows the answer scores 1.0, through the real path."""
    big = registry.get("big")
    question = next(q for q in questions if big.would_be_correct(q.text))
    raw = emit([{"subtask": "solve it", "worker": "big", "deps": []}])
    record = await one_rollout(raw, question, registry, governor=governor, rollout_index=0)
    assert record.parsed
    assert record.score.value == 1.0
    assert record.score.correct


async def test_end_to_end_wrong_rollout_scores_zero(registry, governor, questions, emit):
    big = registry.get("big")
    question = next(q for q in questions if not big.would_be_correct(q.text))
    raw = emit([{"subtask": "solve it", "worker": "big", "deps": []}])
    record = await one_rollout(raw, question, registry, governor=governor, rollout_index=0)
    assert record.parsed
    assert record.score.value == 0.0


async def test_unparseable_emission_never_touches_a_worker(registry, governor, questions):
    before = sum(w.calls for w in registry)
    record = await one_rollout(
        "I refuse to plan.", questions[0], registry, governor=governor, rollout_index=0
    )
    assert not record.parsed
    assert record.parse_reason == "json_decode"
    assert record.score.value == 0.0
    assert sum(w.calls for w in registry) == before  # no work was dispatched


async def test_rollout_group_runs_k_and_attaches_advantages(registry, governor, questions, emit):
    emissions = [
        emit([{"subtask": "solve", "worker": "tiny", "deps": []}]),
        emit([{"subtask": "solve", "worker": "big", "deps": []}]),
        "garbage, no plan here",
        emit([{"subtask": "solve", "worker": "mid", "deps": []}]),
    ]
    group = await rollout_group(
        ScriptedConductor(emissions), registry, questions[0], governor=governor, k=4
    )
    assert len(group) == 4
    assert [r.rollout_index for r in group] == [0, 1, 2, 3]
    assert all(r.advantage is not None for r in group)
    assert any(not r.parsed for r in group)  # the garbage one


async def test_rollout_rows_are_json_safe(registry, governor, questions, emit):
    raw = emit(
        [
            {"subtask": "a", "worker": "tiny", "deps": []},
            {"subtask": "b", "worker": "big", "deps": [1]},
        ],
        final=2,
    )
    record = await one_rollout(raw, questions[0], registry, governor=governor, rollout_index=3)
    row = record.to_row(run_id="test-run")
    json.dumps(row)  # must not raise
    assert row["n_steps"] == 2
    assert row["workers_used"] == ["tiny", "big"]
    assert row["rollout_index"] == 3
    assert row["gold"] == questions[0].gold


async def test_routing_can_beat_every_single_worker_on_a_known_pool(questions, emit):
    """A sanity check on the metric itself, using a pool whose competence is configured.

    Three workers each know a disjoint third of the questions. A perfect router scores 1.0; the best
    single worker scores about a third. If this ever fails, the harness is measuring something other
    than routing quality and no live number from it should be trusted.
    """
    names = ("a", "b", "c")
    workers = [
        FakeWorker(
            fake_spec(name),
            answers={q.text: q.gold for j, q in enumerate(questions) if j % 3 == i},
            skill=1.0,
            seed=i,
        )
        for i, name in enumerate(names)
    ]
    registry = WorkerRegistry(workers)
    governor = _fake_governor()

    solo = dict.fromkeys(names, 0.0)
    routed = 0.0
    for j, question in enumerate(questions):
        for name in names:
            raw = emit([{"subtask": "solve", "worker": name, "deps": []}])
            record = await one_rollout(raw, question, registry, governor=governor, rollout_index=0)
            solo[name] += record.score.value
        best = names[j % 3]
        raw = emit([{"subtask": "solve", "worker": best, "deps": []}])
        record = await one_rollout(raw, question, registry, governor=governor, rollout_index=0)
        routed += record.score.value

    assert routed == len(questions)
    assert max(solo.values()) < routed


def test_question_equality_and_immutability():
    question = Question(id="x", text="t", gold="1")
    assert question == Question(id="x", text="t", gold="1")
    assert question.gold == "1"
