"""Aggregation and the verdict — the phase-0 deliverable is a number, so the maths must be right."""

from __future__ import annotations

from coryphaeus.reporting import (
    best_solo,
    render_parse_failures,
    render_table,
    render_verdict,
    summarize,
)


def solo_row(worker: str, correct: bool, **extra) -> dict:
    return {
        "arm": f"solo:{worker}",
        "correct": correct,
        "parsed": False,  # a solo arm has no workflow at all
        "parse_reason": None,
        "score_reason": "ok",
        "workers_used": [worker],
        "n_steps": 1,
        "cost_usd": 0.0,
        "latency_s": 1.0,
        "tokens_in": 100,
        "tokens_out": 50,
        **extra,
    }


def conductor_row(name: str, correct: bool, *, parse_reason: str | None = None, **extra) -> dict:
    return {
        "arm": f"conductor:{name}",
        "correct": correct,
        "parsed": parse_reason is None,
        "parse_reason": parse_reason,
        "score_reason": "ok",
        "workers_used": extra.pop("workers_used", ["big"]),
        "n_steps": extra.pop("n_steps", 2),
        "cost_usd": 0.0,
        "latency_s": 2.0,
        "tokens_in": 300,
        "tokens_out": 120,
        **extra,
    }


def test_solo_rows_are_not_counted_as_parse_failures():
    """A solo arm never emitted a workflow, so it cannot have failed to parse one."""
    summaries = summarize([solo_row("big", True), solo_row("big", False)])
    assert summaries[0].parse_failures == 0
    assert render_parse_failures([solo_row("big", True)]) == "No parse failures."


def test_conductor_parse_failures_are_counted_and_ranked():
    rows = [
        conductor_row("q4", False, parse_reason="json_decode"),
        conductor_row("q4", False, parse_reason="json_decode"),
        conductor_row("q4", False, parse_reason="unknown_worker"),
        conductor_row("q4", True),
    ]
    summary = summarize(rows)[0]
    assert summary.parse_failures == 3
    assert summary.n == 4
    assert summary.parse_failure_rate == 0.75
    report = render_parse_failures(rows)
    assert report.splitlines()[0].strip().startswith("json_decode: 2")


def test_accuracy_and_means():
    rows = [solo_row("big", True), solo_row("big", True), solo_row("big", False)]
    summary = summarize(rows)[0]
    assert summary.n == 3
    assert summary.correct == 2
    assert abs(summary.accuracy - 2 / 3) < 1e-9
    assert summary.mean_latency == 1.0
    assert summary.mean_tokens == 150
    assert summary.mean_steps == 1.0


def test_best_solo_ignores_conductor_arms():
    rows = [
        solo_row("tiny", False),
        solo_row("big", True),
        conductor_row("q4", True),
        conductor_row("q4", True),
    ]
    summaries = summarize(rows)
    bar = best_solo(summaries)
    assert bar is not None
    assert bar.label == "big"
    assert bar.accuracy == 1.0


def test_solo_arms_sort_before_conductor_arms():
    rows = [conductor_row("q4", True), solo_row("tiny", False), solo_row("big", True)]
    labels = [s.label for s in summarize(rows)]
    assert labels.index("big") < labels.index("q4")
    assert labels.index("tiny") < labels.index("q4")


def test_verdict_says_beats_when_the_conductor_wins():
    rows = [solo_row("big", True), solo_row("big", False)]  # 50%
    rows += [conductor_row("q4", True), conductor_row("q4", True)]  # 100%
    verdict = render_verdict(summarize(rows))
    assert "BEATS" in verdict
    assert "+50.0%" in verdict


def test_verdict_says_loses_when_it_does():
    rows = [solo_row("big", True), solo_row("big", True)]  # 100%
    rows += [conductor_row("q4", True), conductor_row("q4", False)]  # 50%
    verdict = render_verdict(summarize(rows))
    assert "loses to" in verdict
    assert "-50.0%" in verdict


def test_verdict_reports_a_tie_honestly():
    rows = [solo_row("big", True), solo_row("big", False)]
    rows += [conductor_row("q4", True), conductor_row("q4", False)]
    assert "ties" in render_verdict(summarize(rows))


def test_verdict_with_no_conductor_states_only_the_bar():
    verdict = render_verdict(summarize([solo_row("big", True)]))
    assert "No conductor arm ran" in verdict


def test_verdict_with_no_solo_arm_says_there_is_no_bar():
    assert "no bar to clear" in render_verdict(summarize([conductor_row("q4", True)]))


def test_table_renders_every_arm_and_marks_conductors():
    rows = [solo_row("big", True), conductor_row("q4", True)]
    table = render_table(summarize(rows))
    assert "big" in table
    assert "→ q4" in table
    assert len(table.splitlines()) == 4  # header, rule, two arms


def test_routed_to_column_shows_where_work_went():
    rows = [
        conductor_row("q4", True, workers_used=["big", "q2"]),
        conductor_row("q4", True, workers_used=["big"]),
    ]
    summary = summarize(rows)[0]
    assert summary.top_workers().startswith("big×2")


def test_empty_input_is_not_a_crash():
    assert summarize([]) == []
    assert best_solo([]) is None
    assert "no bar to clear" in render_verdict([])
