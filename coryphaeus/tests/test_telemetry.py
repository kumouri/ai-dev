"""Run artifacts — the world-model dataset must survive a round trip intact."""

from __future__ import annotations

import json

from coryphaeus.orchestrate import execute
from coryphaeus.schema import Step, build
from coryphaeus.telemetry import SCHEMA_VERSION, RunWriter, read_jsonl, step_rows


async def test_step_rows_carry_the_world_model_features(registry, governor, questions):
    wf = build(
        [
            Step(index=1, subtask="decompose the problem", worker="tiny"),
            Step(index=2, subtask="solve and verify", worker="big", deps=(1,)),
        ],
        known_workers=registry.names(),
    )
    execution = await execute(wf, registry, questions[0].text, governor=governor)
    rows = step_rows(
        run_id="r1",
        question_id=questions[0].id,
        arm="conductor",
        rollout_index=0,
        execution=execution,
    )

    assert len(rows) == 2
    first, second = rows
    # Features: what was asked, of whom, with what context.
    assert first["subtask"] == "decompose the problem"
    assert first["worker"] == "tiny"
    assert second["deps"] == [1]
    # Outcome: whether it worked and what it cost.
    for row in rows:
        assert row["ok"] is True
        assert row["tokens_in"] > 0
        assert row["latency_s"] >= 0
        assert "cost_usd" in row
        assert row["question_id"] == questions[0].id


async def test_step_rows_record_the_requested_worker_alongside_the_resolved_one(
    registry, governor, questions
):
    """A self-assigned step stays attributable — 'self' and what it resolved to are both facts."""
    from coryphaeus.workers import fake_spec
    from coryphaeus.workers.fake import FakeWorker

    conductor = FakeWorker(fake_spec("conductor"), answers={questions[0].text: questions[0].gold})
    wf = build([Step(index=1, subtask="mine", worker="self")], known_workers=registry.names())
    execution = await execute(
        wf, registry, questions[0].text, governor=governor, self_worker=conductor
    )
    row = step_rows(
        run_id="r1", question_id="q", arm="conductor", rollout_index=0, execution=execution
    )[0]
    assert row["worker_requested"] == "self"
    assert row["worker"] == "conductor"


def test_run_writer_round_trip(tmp_path):
    with RunWriter.create("unit", root=tmp_path) as writer:
        writer.write_meta(arm="conductor", note="hello")
        writer.write_rollout({"question_id": "q1", "score": 1.0})
        writer.write_step({"question_id": "q1", "step_index": 1})
        run_dir = writer.run_dir

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["arm"] == "conductor"

    rollouts = read_jsonl(run_dir / "rollouts.jsonl")
    steps = read_jsonl(run_dir / "steps.jsonl")
    assert rollouts == [{"schema_version": SCHEMA_VERSION, "question_id": "q1", "score": 1.0}]
    assert steps[0]["step_index"] == 1


def test_writing_outside_the_context_manager_is_an_error(tmp_path):
    writer = RunWriter.create("unit", root=tmp_path)
    try:
        writer.write_rollout({"x": 1})
    except RuntimeError as exc:
        assert "context manager" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a RuntimeError")


def test_read_jsonl_refuses_to_silently_skip_corruption(tmp_path):
    path = tmp_path / "steps.jsonl"
    path.write_text('{"ok": true}\nnot json at all\n', encoding="utf-8")
    try:
        read_jsonl(path)
    except ValueError as exc:
        assert "malformed JSONL" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("silently dropping a corrupt row would hide data loss")


def test_read_jsonl_of_a_missing_file_is_empty(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []
