"""Verified-answer reward.

Extraction-and-comparison bugs are the classic reason a GRPO run learns nothing while looking
healthy: the policy improves, the verifier says no, the gradient is noise. So this module is
deliberately paranoid, and ``tests/test_reward.py`` holds an adversarial table of pairs that must
compare correctly.

Comparison is *mathematical*, not textual: ``1/2``, ``0.5``, ``\\frac{1}{2}`` and ``2\\sqrt3`` vs
``2*sqrt(3)`` all have to work. The verifier that does this is the sympy comparison in this module.

**Why not ``math_verify`` by default.** The obvious move is HuggingFace's ``math_verify``, and the
``verify`` extra installs it — but its extractor is built for *full model output*, using first-match
extraction to find an answer buried in prose. Handed an already-extracted short answer with a
non-LaTeX gold it truncates: ``16*pi`` parses to ``16``, so ``16`` verifies as **equal** to
``16*pi``, and ``4\\sqrt{2}`` as equal to ``4*sqrt(3)``. A verifier that calls wrong answers correct
is the worst failure mode here — it inflates every arm at once and rewards the policy for being
wrong. So it is **opt-in** via ``CORYPHAEUS_USE_MATH_VERIFY=1`` — useful for cross-checking, and
sound on LaTeX golds — never the default. ``tests/test_reward.py`` pins both false positives.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

#: Score for a rollout whose workflow never parsed, or whose final step failed.
ZERO = 0.0

try:  # pragma: no cover - exercised by whichever path the environment has
    import logging

    from math_verify import parse as _mv_parse
    from math_verify import verify as _mv_verify

    HAVE_MATH_VERIFY = True
    # It warns loudly, per call, whenever its timeout is disabled. Over a 500-question run that is
    # thousands of lines of noise about a decision made deliberately below.
    logging.getLogger("math_verify").setLevel(logging.ERROR)
except Exception:  # noqa: BLE001 - any import failure means "use the fallback"
    HAVE_MATH_VERIFY = False


def use_math_verify() -> bool:
    """Whether to consult ``math_verify`` — opt-in only, for the reason in the module docstring."""
    flag = os.environ.get("CORYPHAEUS_USE_MATH_VERIFY", "").strip().lower()
    return HAVE_MATH_VERIFY and flag not in ("", "0", "false", "no")


#: ``math_verify`` guards against pathological expressions with a timeout: signal-based on POSIX,
#: multiprocessing-based on Windows. The Windows path needs a ``__main__`` spawn guard the callers
#: here cannot provide (pytest, asyncio), and raises ``PermissionError``/``OSError`` instead — so it
#: is disabled there and the caller's ``try/except`` is the only net. Acceptable because the inputs
#: are short extracted answers rather than arbitrary text, and Linux keeps the real timeout.
_MV_TIMEOUT: int | None = None if sys.platform == "win32" else 5

_ANSWER_CUES = (
    r"final answer is",
    r"final answer:",
    r"the answer is",
    r"answer:",
)

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


@dataclass(frozen=True, slots=True)
class Score:
    """The outcome of scoring one rollout."""

    value: float
    correct: bool
    extracted: str | None
    gold: str
    reason: str = "ok"

    @property
    def ok(self) -> bool:
        return self.reason == "ok"


def _extract_braced(text: str, start: int) -> tuple[str | None, int]:
    """Read a balanced ``{...}`` group at ``start``. Handles nesting, e.g. ``\\frac{1}{2}``."""
    if start >= len(text) or text[start] != "{":
        return None, start
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return None, len(text)


def extract_answer(text: str) -> str | None:
    """Pull the final answer out of a worker's reply.

    Priority: the last ``\\boxed{...}`` (what the prompt asks for), then an explicit answer cue,
    then the last number on the last non-empty line. Returns ``None`` when there is nothing to
    score — a real outcome, not an error, scored zero with reason ``no_answer``.
    """
    if not text:
        return None

    # 1. last \boxed{...}
    last: str | None = None
    for match in re.finditer(r"\\boxed\s*", text):
        content, _ = _extract_braced(text, match.end())
        if content is None:
            # \boxed 42 (no braces) — take the rest of the line.
            tail = text[match.end() :].strip().splitlines()
            if tail:
                content = tail[0].strip()
        if content:
            last = content
    if last:
        return last.strip()

    # 2. an explicit cue, last occurrence wins
    lowered = text.lower()
    best_pos = -1
    for cue in _ANSWER_CUES:
        for match in re.finditer(cue, lowered):
            if match.end() > best_pos:
                best_pos = match.end()
    if best_pos >= 0:
        tail = text[best_pos:].strip()
        line = tail.splitlines()[0].strip() if tail else ""
        line = line.strip("*: \t")
        if line:
            numbers = _NUMBER_RE.findall(line)
            return (numbers[-1] if numbers else line).strip(" .$")

    # 3. last number anywhere in the last non-empty line
    for line in reversed(text.strip().splitlines()):
        numbers = _NUMBER_RE.findall(line)
        if numbers:
            return numbers[-1]
    return None


def normalize(answer: str) -> str:
    """Textual normalisation applied before any mathematical comparison."""
    s = answer.strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    # Order matters: strip the escaped percent before the bare one, or "50\%" leaves a stray "\".
    s = s.replace("$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\\text\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)  # thousands separators
    s = s.rstrip(".")
    s = s.strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    return s


def _latex_to_sympy(expr: str) -> str:
    """A small, honest LaTeX→sympy translation covering what these datasets actually contain."""
    s = expr
    # \frac{a}{b} -> ((a)/(b)), innermost first
    pattern = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    for _ in range(6):
        new = pattern.sub(r"((\1)/(\2))", s)
        if new == s:
            break
        s = new
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt\s*(\d+)", r"sqrt(\1)", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.replace("\\pi", "pi").replace("^", "**")
    s = re.sub(r"\\[a-zA-Z]+", "", s)  # drop remaining commands rather than choke on them
    s = s.replace("{", "(").replace("}", ")")
    # Implicit multiplication: 2sqrt(3) -> 2*sqrt(3), 2( -> 2*(
    s = re.sub(r"(\d)\s*(?=[a-zA-Z(])", r"\1*", s)
    return s.strip()


def _sympy_equal(a: str, b: str) -> bool:
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            standard_transformations,
        )
    except Exception:  # noqa: BLE001
        return False

    transformations = (
        *standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    try:
        left = sympy.parse_expr(_latex_to_sympy(a), transformations=transformations, evaluate=True)
        right = sympy.parse_expr(_latex_to_sympy(b), transformations=transformations, evaluate=True)
    except Exception:  # noqa: BLE001 - unparseable is a legitimate "not equal"
        return False

    try:
        if sympy.simplify(left - right) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        return bool(abs(float(left) - float(right)) < 1e-6)
    except Exception:  # noqa: BLE001
        return False


def equivalent(predicted: str, gold: str) -> bool:
    """Whether two answer strings denote the same value."""
    if predicted is None or gold is None:
        return False
    p, g = normalize(predicted), normalize(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    if p.lower() == g.lower():
        return True

    if use_math_verify():  # pragma: no cover - opt-in, see the module docstring
        try:
            gold_expr = _mv_parse(g, parsing_timeout=_MV_TIMEOUT)
            pred_expr = _mv_parse(p, parsing_timeout=_MV_TIMEOUT)
            if _mv_verify(gold_expr, pred_expr, timeout_seconds=_MV_TIMEOUT):
                return True
        except Exception:  # noqa: BLE001 - fall through to sympy
            pass

    return _sympy_equal(p, g)


def score_answer(final_text: str, gold: str) -> Score:
    """Score a final answer against gold: 1.0 for equivalent, else 0.0."""
    extracted = extract_answer(final_text or "")
    if extracted is None:
        return Score(ZERO, False, None, gold, reason="no_answer")
    correct = equivalent(extracted, gold)
    return Score(1.0 if correct else ZERO, correct, extracted, gold, reason="ok")


def score_failure(reason: str, gold: str = "") -> Score:
    """Score for a rollout that never produced an answer — a parse failure or a dead final step.

    Deliberately the same zero a wrong answer earns: under GRPO the policy must feel that emitting
    an unparseable workflow is as bad as being wrong. See ``docs/adr/0001``.
    """
    return Score(ZERO, False, None, gold, reason=reason)
