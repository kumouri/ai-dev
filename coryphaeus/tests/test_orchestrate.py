"""Execution: dependency threading, failure paths, self-assignment, and accounting."""

from __future__ import annotations

from coryphaeus.orchestrate import execute, run_solo
from coryphaeus.schema import Step, build
from coryphaeus.workers import Governor, WorkerRegistry, WorkerResult, fake_spec
from coryphaeus.workers.fake import FakeWorker


async def _no_sleep(_seconds: float) -> None:
    return None


def _governor() -> Governor:
    return Governor(budgets={"fake": 16}, sleeper=_no_sleep, base_delay=0.0, jitter=0.0)


async def test_single_step_returns_that_step_text(registry, governor, questions):
    q = questions[0]
    wf = build([Step(index=1, subtask="solve it", worker="big")], known_workers=registry.names())
    result = await execute(wf, registry, q.text, governor=governor)
    assert result.n_steps == 1
    assert result.ok
    assert result.final_text


async def test_a_step_sees_only_its_declared_deps(registry, governor, questions):
    q = questions[0]
    wf = build(
        [
            Step(index=1, subtask="STEP-ONE-MARKER", worker="tiny"),
            Step(index=2, subtask="STEP-TWO-MARKER", worker="mid"),
            Step(index=3, subtask="aggregate", worker="big", deps=(2,)),
        ],
        known_workers=registry.names(),
    )
    await execute(wf, registry, q.text, governor=governor)

    big = registry.get("big")
    prompt = big.prompts[-1]
    assert "[step 2]" in prompt
    assert "[step 1]" not in prompt  # step 1's result was not declared, so it must not leak


async def test_no_deps_means_no_prior_results_block(registry, governor, questions):
    wf = build(
        [
            Step(index=1, subtask="a", worker="tiny"),
            Step(index=2, subtask="b", worker="mid", deps=()),
        ],
        known_workers=registry.names(),
    )
    await execute(wf, registry, questions[0].text, governor=governor)
    assert "Results from earlier steps" not in registry.get("mid").prompts[-1]


#: The final-step instruction, matched as a phrase. Matching on "\boxed" alone gives false
#: positives, because an upstream worker's *answer* is itself boxed and arrives as a prior result.
FINAL_CUE = "End your reply with the final answer"


async def test_only_the_final_step_is_asked_to_box_its_answer(registry, governor, questions):
    wf = build(
        [
            Step(index=1, subtask="a", worker="tiny"),
            Step(index=2, subtask="b", worker="mid", deps=(1,)),
        ],
        known_workers=registry.names(),
    )
    await execute(wf, registry, questions[0].text, governor=governor)
    assert FINAL_CUE not in registry.get("tiny").prompts[-1]
    assert FINAL_CUE in registry.get("mid").prompts[-1]


async def test_final_may_be_an_earlier_step(registry, governor, questions):
    wf = build(
        [
            Step(index=1, subtask="the real answer", worker="big"),
            Step(index=2, subtask="a side note", worker="tiny", deps=(1,)),
        ],
        final=1,
        known_workers=registry.names(),
    )
    result = await execute(wf, registry, questions[0].text, governor=governor)
    assert result.final_text == result.outcomes[0].result.text  # step 1's output is the answer
    assert FINAL_CUE in registry.get("big").prompts[-1]  # step 1 got the final instruction
    assert FINAL_CUE not in registry.get("tiny").prompts[-1]  # step 2 did not


async def test_failed_step_becomes_a_note_downstream_not_an_exception(questions):
    class Broken:
        spec = fake_spec("broken")

        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def invoke(self, prompt, *, system=None, max_tokens=1024, temperature=0.7):
            self.prompts.append(prompt)
            return WorkerResult(worker="broken", text="", ok=False, error="http 500")

    answers = {questions[0].text: questions[0].gold}
    good = FakeWorker(fake_spec("good"), answers=answers)
    registry = WorkerRegistry([Broken(), good])
    wf = build(
        [
            Step(index=1, subtask="try", worker="broken"),
            Step(index=2, subtask="recover", worker="good", deps=(1,)),
        ],
        known_workers=registry.names(),
    )
    result = await execute(wf, registry, questions[0].text, governor=_governor())

    assert result.failed_steps == (1,)
    assert result.ok  # the FINAL step succeeded, so the rollout is scoreable
    assert "[worker broken failed" in good.prompts[-1]


async def test_failed_final_step_marks_the_execution_not_ok(questions):
    class Broken:
        spec = fake_spec("broken")

        async def invoke(self, prompt, *, system=None, max_tokens=1024, temperature=0.7):
            return WorkerResult(worker="broken", text="", ok=False, error="http 500")

    registry = WorkerRegistry([Broken()])
    wf = build([Step(index=1, subtask="try", worker="broken")], known_workers=registry.names())
    result = await execute(wf, registry, questions[0].text, governor=_governor())
    assert not result.ok
    assert result.error == "final step failed"


async def test_self_assignment_routes_to_the_conductor_worker(registry, governor, questions):
    conductor = FakeWorker(fake_spec("conductor"), answers={questions[0].text: questions[0].gold})
    wf = build(
        [Step(index=1, subtask="I'll handle it", worker="self")], known_workers=registry.names()
    )
    result = await execute(
        wf, registry, questions[0].text, governor=governor, self_worker=conductor
    )
    assert conductor.calls == 1
    assert result.outcomes[0].resolved_worker == "conductor"


async def test_self_assignment_without_a_conductor_fails_loudly(registry, governor, questions):
    """Silently substituting another worker would corrupt the attribution telemetry exists for."""
    wf = build([Step(index=1, subtask="mine", worker="self")], known_workers=registry.names())
    result = await execute(wf, registry, questions[0].text, governor=governor)
    assert not result.ok
    assert "self_unavailable" in (result.outcomes[0].result.error or "")


async def test_accounting_sums_across_steps(registry, governor, questions):
    wf = build(
        [
            Step(index=1, subtask="a", worker="tiny"),
            Step(index=2, subtask="b", worker="mid", deps=(1,)),
            Step(index=3, subtask="c", worker="big", deps=(1, 2)),
        ],
        known_workers=registry.names(),
    )
    result = await execute(wf, registry, questions[0].text, governor=governor)
    assert result.n_steps == 3
    assert result.tokens_in > 0
    assert result.tokens_out > 0
    assert result.workers_used == ("tiny", "mid", "big")


async def test_run_solo_looks_like_a_one_step_execution(registry, governor, questions):
    result = await run_solo(registry.get("big"), questions[0].text, governor=governor)
    assert result.n_steps == 1
    assert result.workers_used == ("big",)
    assert "\\boxed" in registry.get("big").prompts[-1]
