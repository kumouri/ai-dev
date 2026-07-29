"""The adversarial table.

Extraction-and-comparison bugs are how a GRPO run silently learns nothing, so these cases are the
ones that matter most in the whole suite.
"""

from __future__ import annotations

import pytest

from coryphaeus.reward import (
    equivalent,
    extract_answer,
    normalize,
    score_answer,
    score_failure,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The answer is \\boxed{42}", "42"),
        ("... so \\boxed{\\frac{1}{2}} is the result.", "\\frac{1}{2}"),
        ("\\boxed{4\\sqrt{3}}", "4\\sqrt{3}"),
        ("first \\boxed{1} then corrected: \\boxed{2}", "2"),  # last box wins
        ("The final answer is 65.", "65"),
        ("Answer: 11/12", "11/12"),
        ("the answer is 7 apples", "7"),
        ("Working:\n2 + 2\nTherefore 4", "4"),
        ("Total: 1,234 units", "1,234"),
        ("no numbers at all here", None),
        ("", None),
    ],
)
def test_extraction(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$42$", "42"),
        ("42.", "42"),
        ("1,234", "1234"),
        ("\\text{60}", "60"),
        ("\\dfrac{1}{2}", "\\frac{1}{2}"),
        ("\\left(3\\right)", "(3)"),
        ("50\\%", "50"),
    ],
)
def test_normalization(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        ("42", "42"),
        ("42.0", "42"),
        ("1/2", "0.5"),
        ("\\frac{1}{2}", "1/2"),
        ("\\dfrac{11}{12}", "11/12"),
        ("4\\sqrt{3}", "4*sqrt(3)"),
        ("sqrt(48)", "4*sqrt(3)"),
        ("16\\pi", "16*pi"),
        ("2^4", "16"),
        ("1,234", "1234"),
        ("$65$", "65"),
        ("65.", "65"),
        ("-7", "-7"),
    ],
)
def test_equivalent_pairs(predicted, gold):
    assert equivalent(predicted, gold), f"{predicted!r} should equal {gold!r}"


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        ("41", "42"),
        ("1/3", "1/2"),
        ("4\\sqrt{2}", "4*sqrt(3)"),
        ("16", "16*pi"),
        ("", "42"),
        ("-42", "42"),
        ("7/30", "7/29"),
    ],
)
def test_non_equivalent_pairs(predicted, gold):
    assert not equivalent(predicted, gold), f"{predicted!r} should NOT equal {gold!r}"


def test_score_answer_correct():
    score = score_answer("Therefore \\boxed{11/12}", "11/12")
    assert score.value == 1.0
    assert score.correct
    assert score.reason == "ok"
    assert score.extracted == "11/12"


def test_score_answer_wrong_is_zero_but_still_ok_status():
    score = score_answer("Therefore \\boxed{5}", "11/12")
    assert score.value == 0.0
    assert not score.correct
    assert score.reason == "ok"  # scoring worked; the answer was wrong


def test_missing_answer_is_its_own_reason():
    score = score_answer("I could not work it out.", "42")
    assert score.value == 0.0
    assert score.reason == "no_answer"
    assert score.extracted is None


def test_parse_failure_scores_the_same_zero_as_a_wrong_answer():
    """The policy must feel an unparseable workflow as costly as being wrong (ADR-0001)."""
    failure = score_failure("too_many_steps", gold="42")
    wrong = score_answer("\\boxed{41}", "42")
    assert failure.value == wrong.value == 0.0
    assert failure.reason == "too_many_steps"


@pytest.mark.parametrize(
    ("predicted", "gold"),
    [
        ("16", "16*pi"),
        ("4\\sqrt{2}", "4*sqrt(3)"),
    ],
)
def test_no_false_positives_from_prefix_truncation(predicted, gold, monkeypatch):
    """Regression pins for the `math_verify` trap described in reward.py's docstring.

    Its first-match extractor is built for prose containing an answer; on a bare non-LaTeX gold it
    truncates ``16*pi`` to ``16`` and reports equality. A verifier that calls a wrong answer correct
    inflates every arm at once, so this must hold whether or not the extra is installed.
    """
    monkeypatch.setenv("CORYPHAEUS_USE_MATH_VERIFY", "0")
    assert not equivalent(predicted, gold)


def test_verifier_survives_hostile_input():
    """Malformed LaTeX must return False, never raise — one crash would kill a whole run."""
    for junk in ("\\frac{1}{", "}{", "\\", "((((", "1/0", "nan", "\\boxed{"):
        assert equivalent(junk, "42") is False
