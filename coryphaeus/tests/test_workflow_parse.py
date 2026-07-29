"""Parsing real-model sloppiness — and refusing to launder real-model errors."""

from __future__ import annotations

import pytest

from coryphaeus.workflow import parse

WORKERS = ("tiny", "mid", "big")


def test_fenced_json_parses():
    raw = 'Plan:\n```json\n{"steps": [{"subtask": "solve it", "worker": "big", "deps": []}]}\n```'
    result = parse(raw, known_workers=WORKERS)
    assert result.ok
    assert result.workflow.n_steps == 1
    assert result.workflow.final == 1


def test_bare_json_without_a_fence_parses():
    raw = '{"steps": [{"subtask": "solve", "worker": "mid"}], "final": 1}'
    assert parse(raw, known_workers=WORKERS).ok


def test_json_buried_in_prose_parses():
    raw = 'I think the best plan is {"steps": [{"subtask": "solve", "worker": "mid"}]} — done.'
    assert parse(raw, known_workers=WORKERS).ok


def test_last_fenced_block_wins_over_a_draft():
    raw = (
        '```json\n{"steps": [{"subtask": "draft", "worker": "tiny"}]}\n```\n'
        'On reflection:\n```json\n{"steps": [{"subtask": "final", "worker": "big"}]}\n```'
    )
    result = parse(raw, known_workers=WORKERS)
    assert result.ok
    assert result.workflow.step(1).subtask == "final"


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"steps": [{"subtask": "a", "worker": "tiny",},],}\n```',  # trailing commas
        "```json\n{“steps”: [{“subtask”: “a”, “worker”: “tiny”}]}\n```",  # smart quotes
        '```json\n{"steps": [\n // the only step\n {"subtask": "a", "worker": "tiny"}]}\n```',
    ],
)
def test_repair_pass_fixes_formatting_noise_and_says_so(raw):
    result = parse(raw, known_workers=WORKERS)
    assert result.ok
    assert result.repaired is True


def test_clean_json_is_not_marked_repaired():
    raw = '```json\n{"steps": [{"subtask": "a", "worker": "tiny"}]}\n```'
    assert parse(raw, known_workers=WORKERS).repaired is False


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("I refuse to plan; the answer is 42.", "json_decode"),
        ("", "json_decode"),
        ('```json\n{"steps": "solve it"}\n```', "steps_not_a_list"),
        ('```json\n{"steps": []}\n```', "no_steps"),
        ('```json\n{"steps": ["just solve it"]}\n```', "bad_step_object"),
        ('```json\n{"steps": [{"worker": "tiny"}]}\n```', "empty_subtask"),
        ('```json\n{"steps": [{"subtask": "a"}]}\n```', "unknown_worker"),
        ('```json\n{"steps": [{"subtask": "a", "worker": "gpt-9"}]}\n```', "unknown_worker"),
        ('```json\n{"steps": [{"subtask": "a", "worker": "tiny", "deps": [{}]}]}\n```', "bad_dep"),
        (
            '```json\n{"steps": [{"subtask": "a", "worker": "tiny"}], "final": "last"}\n```',
            "bad_final",
        ),
    ],
)
def test_unparseable_emissions_fail_with_a_named_reason(raw, reason):
    result = parse(raw, known_workers=WORKERS)
    assert not result.ok
    assert result.reason == reason
    assert result.raw == raw  # the raw emission is always preserved for telemetry


def test_too_many_steps_is_rejected_not_truncated():
    steps = ", ".join(f'{{"subtask": "s{i}", "worker": "tiny"}}' for i in range(6))
    result = parse(f'```json\n{{"steps": [{steps}]}}\n```', known_workers=WORKERS)
    assert result.reason == "too_many_steps"


def test_default_deps_all_gives_every_earlier_step():
    raw = (
        '```json\n{"steps": [{"subtask": "a", "worker": "tiny"}, '
        '{"subtask": "b", "worker": "mid"}, {"subtask": "c", "worker": "big"}]}\n```'
    )
    wf = parse(raw, known_workers=WORKERS, default_deps="all").workflow
    assert [s.deps for s in wf.steps] == [(), (1,), (1, 2)]


def test_default_deps_prev_gives_a_chain():
    raw = (
        '```json\n{"steps": [{"subtask": "a", "worker": "tiny"}, '
        '{"subtask": "b", "worker": "mid"}, {"subtask": "c", "worker": "big"}]}\n```'
    )
    wf = parse(raw, known_workers=WORKERS, default_deps="prev").workflow
    assert [s.deps for s in wf.steps] == [(), (1,), (2,)]


def test_explicit_empty_deps_beats_the_default():
    raw = (
        '```json\n{"steps": [{"subtask": "a", "worker": "tiny"}, '
        '{"subtask": "b", "worker": "mid", "deps": []}]}\n```'
    )
    wf = parse(raw, known_workers=WORKERS, default_deps="all").workflow
    assert wf.step(2).deps == ()


def test_field_aliases_models_actually_emit():
    raw = (
        '```json\n{"steps": [{"task": "a", "model": "tiny"}, '
        '{"instruction": "b", "agent": "mid", "depends_on": [1]}], "final_step": 2}\n```'
    )
    result = parse(raw, known_workers=WORKERS)
    assert result.ok
    assert result.workflow.step(2).deps == (1,)
    assert result.workflow.final == 2


def test_scalar_and_stringy_deps_are_coerced():
    raw = (
        '```json\n{"steps": [{"subtask": "a", "worker": "tiny"}, '
        '{"subtask": "b", "worker": "mid", "deps": 1}, '
        '{"subtask": "c", "worker": "big", "deps": "steps 1 and 2"}]}\n```'
    )
    wf = parse(raw, known_workers=WORKERS).workflow
    assert wf.step(2).deps == (1,)
    assert wf.step(3).deps == (1, 2)


def test_self_assignment_is_accepted():
    raw = '```json\n{"steps": [{"subtask": "I will do it", "worker": "self"}]}\n```'
    assert parse(raw, known_workers=WORKERS).ok
