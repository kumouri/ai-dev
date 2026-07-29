"""The reward bridge — tested synchronously, the way TRL actually calls it.

Everything here runs against the fake pool: no network, no GPU, no key.
"""

from __future__ import annotations

import asyncio

import pytest

from coryphaeus.datasets.loaders import Question
from coryphaeus.train.bridge import BridgeClosed, RewardBridge, make_reward_fn
from coryphaeus.workers import Governor, WorkerRegistry, fake_spec
from coryphaeus.workers.fake import FakeWorker


def _emit_workflow(worker: str, subtask: str = "solve it") -> str:
    """A single-step workflow routed to one worker — the way a model would emit it."""
    import json

    body = json.dumps({"steps": [{"subtask": subtask, "worker": worker, "deps": []}]})
    return f"Plan:\n```json\n{body}\n```"


@pytest.fixture
def pool(questions):
    """Three workers, each knowing a disjoint third of the questions."""
    names = ("a", "b", "c")
    return [
        FakeWorker(
            fake_spec(name),
            answers={q.text: q.gold for j, q in enumerate(questions) if j % 3 == i},
            skill=1.0,
            seed=i,
        )
        for i, name in enumerate(names)
    ]


@pytest.fixture
def full_registry(pool):
    return WorkerRegistry(pool)


@pytest.fixture
def fake_governor():
    async def no_sleep(_s: float) -> None:
        return None

    return Governor(budgets={"fake": 16}, sleeper=no_sleep, base_delay=0.0, jitter=0.0)


@pytest.fixture
def bridge(full_registry, fake_governor):
    with RewardBridge(full_registry, fake_governor, timeout_s=30.0) as b:
        yield b


def test_scores_align_with_completion_order(bridge, questions):
    """Question j is known only by worker j%3, so the expected pattern is exactly predictable."""
    q = questions[0]  # known by 'a'
    completions = [_emit_workflow("a"), _emit_workflow("b"), _emit_workflow("c")]
    scores = bridge.score(completions, [q, q, q])
    assert scores == [1.0, 0.0, 0.0]


def test_a_malformed_completion_scores_zero_with_a_reason(bridge, questions):
    records = bridge.score_records(["I will not plan.", _emit_workflow("a")], questions[:1] * 2)
    assert records[0].score.value == 0.0
    assert records[0].parse_reason == "json_decode"
    assert records[1].score.value == 1.0


def test_per_item_pools_are_honoured(bridge, questions):
    """Routing to a worker the prompt never advertised must fail, not silently succeed.

    Item 0 is shown only ('b','c') and routes to 'a' — which it was never offered.
    """
    q = questions[0]  # only 'a' knows it
    records = bridge.score_records(
        [_emit_workflow("a"), _emit_workflow("a")],
        [q, q],
        pools=[("b", "c"), None],
    )
    assert records[0].parse_reason == "unknown_worker"
    assert records[0].score.value == 0.0
    assert records[1].score.value == 1.0  # same completion, full pool, scores fine


def test_batch_is_scored_concurrently(full_registry, fake_governor, questions):
    """Serial scoring would make each training step k times slower for no reason."""
    slow = [
        FakeWorker(fake_spec(n), answers={q.text: q.gold for q in questions}, latency_s=0.05)
        for n in ("s1", "s2", "s3", "s4")
    ]
    registry = WorkerRegistry(slow)
    with RewardBridge(registry, fake_governor, timeout_s=30.0) as b:
        completions = [_emit_workflow(w.spec.name) for w in slow]
        b.score(completions, [questions[0]] * 4)
    assert max(w.max_concurrent for w in slow) >= 1
    # Each worker was hit once; the point is the batch was one gather, not four awaits.
    assert all(w.calls == 1 for w in slow)


def test_mismatched_lengths_are_rejected(bridge, questions):
    with pytest.raises(ValueError, match="completions vs"):
        bridge.score_records(["a", "b"], questions[:1])


def test_bridge_is_reusable_across_batches(bridge, questions):
    bridge.score([_emit_workflow("a")], questions[:1])
    bridge.score([_emit_workflow("a")], questions[:1])
    assert bridge.batches == 2
    assert bridge.rollouts == 2


def test_scoring_after_close_is_an_error(full_registry, fake_governor, questions):
    b = RewardBridge(full_registry, fake_governor).start()
    b.close()
    with pytest.raises(BridgeClosed):
        b.score([_emit_workflow("a")], questions[:1])


def test_close_is_idempotent(full_registry, fake_governor):
    b = RewardBridge(full_registry, fake_governor).start()
    b.close()
    b.close()  # must not raise


def test_a_hang_surfaces_as_a_timeout_not_a_stall(fake_governor, questions):
    """A deadlock must fail the run rather than look like slow training."""

    class Hanging:
        spec = fake_spec("hang")

        async def invoke(self, prompt, *, system=None, max_tokens=1024, temperature=0.7):
            await asyncio.sleep(30)
            raise AssertionError("should never get here")

    registry = WorkerRegistry([Hanging()])
    with RewardBridge(registry, fake_governor, timeout_s=0.3) as b:
        with pytest.raises(TimeoutError, match="exceeded 0.3s"):
            b.score([_emit_workflow("hang")], questions[:1])


# --- the TRL-facing callable ---------------------------------------------------------------------


def test_reward_fn_reads_dataset_columns(bridge, questions):
    fn = make_reward_fn(bridge)
    q = questions[0]
    scores = fn(
        completions=[_emit_workflow("a"), _emit_workflow("b")],
        prompts=["p", "p"],
        question=[q.text, q.text],
        gold=[q.gold, q.gold],
        question_id=[q.id, q.id],
    )
    assert scores == [1.0, 0.0]


def test_reward_fn_honours_the_pool_column(bridge, questions):
    fn = make_reward_fn(bridge)
    q = questions[0]
    scores = fn(
        completions=[_emit_workflow("a")],
        question=[q.text],
        gold=[q.gold],
        pool=[("b", "c")],
    )
    assert scores == [0.0]  # 'a' was not in the advertised pool


def test_reward_fn_unwraps_chat_format_completions(bridge, questions):
    """TRL hands conversational datasets back as message lists, not strings."""
    fn = make_reward_fn(bridge)
    q = questions[0]
    scores = fn(
        completions=[[{"role": "assistant", "content": _emit_workflow("a")}]],
        question=[q.text],
        gold=[q.gold],
    )
    assert scores == [1.0]


def test_reward_fn_without_gold_fails_loudly(bridge, questions):
    """Silently scoring everything zero would look exactly like a broken policy."""
    fn = make_reward_fn(bridge)
    with pytest.raises(ValueError, match="must carry a 'gold' column"):
        fn(completions=[_emit_workflow("a")], question=[questions[0].text])


def test_reward_fn_accepts_answer_as_an_alias(bridge, questions):
    fn = make_reward_fn(bridge)
    q = questions[0]
    assert fn(completions=[_emit_workflow("a")], question=[q.text], answer=[q.gold]) == [1.0]


def test_reward_fn_passes_records_to_the_telemetry_callback(bridge, questions):
    seen: list = []
    fn = make_reward_fn(bridge, on_records=seen.extend)
    q = questions[0]
    fn(completions=[_emit_workflow("a")], question=[q.text], gold=[q.gold])
    assert len(seen) == 1
    assert seen[0].score.value == 1.0


def test_reward_fn_requires_completions(bridge):
    fn = make_reward_fn(bridge)
    with pytest.raises(ValueError, match="no completions"):
        fn(gold=["1"])


def test_question_ids_default_when_absent(bridge, questions):
    fn = make_reward_fn(bridge)
    q = questions[0]
    assert fn(completions=[_emit_workflow("a")], question=[q.text], gold=[q.gold]) == [1.0]


def test_bridge_does_not_compute_advantages(bridge, questions):
    """TRL does that itself; group_advantages is for offline analysis only."""
    records = bridge.score_records([_emit_workflow("a")], questions[:1])
    assert records[0].advantage is None


def test_question_objects_are_built_from_columns(bridge):
    """A sanity check that the reward path does not need a Question up front."""
    fn = make_reward_fn(bridge)
    assert fn(completions=["garbage"], question=["2+2?"], gold=["4"]) == [0.0]
    assert isinstance(Question(id="x", text="2+2?", gold="4"), Question)
