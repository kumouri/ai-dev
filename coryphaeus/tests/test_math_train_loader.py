"""The boxed-answer extractor — the part of the MATH loader that decides reward integrity.

A gold answer that was never extracted scores every rollout zero and reads as policy collapse,
so the brace-walking gets adversarial cases. The HF download itself is `data`-marked territory;
everything here is offline.
"""

from __future__ import annotations

import pytest

from coryphaeus.datasets.loaders import extract_boxed


@pytest.mark.parametrize(
    ("solution", "expected"),
    [
        ("The answer is \\boxed{42}.", "42"),
        ("So \\boxed{\\frac{1}{2}} is the probability.", "\\frac{1}{2}"),  # nested braces
        ("\\boxed{2\\sqrt{3}}", "2\\sqrt{3}"),
        ("First \\boxed{5} then finally \\boxed{7}", "7"),  # LAST box wins
        ("\\boxed{{x : x > 0}}", "{x : x > 0}"),  # doubled braces survive the walk
        (
            "\\boxed{\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}}",
            "\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}",
        ),
        ("\\boxed{  spaced  }", "spaced"),
        ("no box at all", None),
        ("\\boxed{unclosed", None),  # malformed: skip the row, never guess
        ("", None),
    ],
)
def test_extract_boxed(solution, expected):
    assert extract_boxed(solution) == expected


def test_intermediate_boxes_do_not_shadow_the_final_answer():
    solution = (
        "We compute \\boxed{x=3} as an intermediate step. Substituting back, the final "
        "value is \\boxed{\\frac{27}{4}}."
    )
    assert extract_boxed(solution) == "\\frac{27}{4}"
