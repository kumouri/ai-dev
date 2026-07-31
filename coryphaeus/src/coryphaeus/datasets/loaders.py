"""Question loaders.

Datasets are **fetched, never vendored** — this repo is public and redistributing a corpus under our
own licence is not ours to do. The one exception is a small hand-written fixture that ships inside
the package so the offline test suite and a no-network smoke run always work.

AIME25 is deliberately absent. It is the held-out set; a loader would make it too easy to train on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@dataclass(frozen=True, slots=True)
class Question:
    """One verifiable question."""

    id: str
    text: str
    gold: str
    source: str = "unknown"
    meta: dict = field(default_factory=dict)


def load_jsonl(
    path: Path | str, *, limit: int | None = None, source: str | None = None
) -> list[Question]:
    """Load questions from JSONL rows keyed ``id``, ``question``/``problem``/``text``, and
    ``answer``/``gold``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such question file: {path}")
    questions: list[Question] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        text = row.get("question") or row.get("problem") or row.get("text")
        gold = row.get("answer") or row.get("gold")
        if not text or gold is None:
            raise ValueError(f"{path}:{lineno}: row needs a question and an answer")
        questions.append(
            Question(
                id=str(row.get("id", f"{path.stem}-{lineno}")),
                text=str(text),
                gold=str(gold),
                source=source or row.get("source") or path.stem,
                meta={
                    k: v
                    for k, v in row.items()
                    if k not in {"id", "question", "problem", "text", "answer", "gold", "source"}
                },
            )
        )
        if limit and len(questions) >= limit:
            break
    return questions


def load_fixture(name: str = "mini_math", *, limit: int | None = None) -> list[Question]:
    """Load a bundled fixture — no network, always available."""
    return load_jsonl(FIXTURE_DIR / f"{name}.jsonl", limit=limit, source=f"fixture:{name}")


def _require_datasets():
    # Load .env *before* importing datasets: huggingface_hub reads HF_HOME at import time, so a
    # cache redirect set in .env has to be in the environment already or it is silently ignored.
    from ..config import load_dotenv

    load_dotenv()
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "this loader needs the 'data' extra: uv sync --extra data "
            "(the offline suite uses load_fixture instead)"
        ) from exc
    return load_dataset


def load_gsm8k(*, split: str = "test", limit: int | None = None) -> list[Question]:
    """GSM8K. Gold is the value after the ``####`` marker in the reference solution."""
    load_dataset = _require_datasets()
    rows = load_dataset("openai/gsm8k", "main", split=split)
    questions: list[Question] = []
    for i, row in enumerate(rows):
        answer = str(row["answer"])
        gold = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
        questions.append(
            Question(
                id=f"gsm8k-{split}-{i}",
                text=str(row["question"]).strip(),
                gold=gold,
                source="gsm8k",
            )
        )
        if limit and len(questions) >= limit:
            break
    return questions


def load_math500(*, limit: int | None = None) -> list[Question]:
    """MATH500 — harder, and spread across difficulty levels, which GSM8K is not."""
    load_dataset = _require_datasets()
    rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
    questions: list[Question] = []
    for i, row in enumerate(rows):
        questions.append(
            Question(
                id=f"math500-{i}",
                text=str(row["problem"]).strip(),
                gold=str(row["answer"]).strip(),
                source="math500",
                meta={"level": row.get("level"), "subject": row.get("subject")},
            )
        )
        if limit and len(questions) >= limit:
            break
    return questions


def extract_boxed(solution: str) -> str | None:
    """The content of the LAST ``\\boxed{...}`` in a MATH solution, braces balanced.

    MATH's gold answers live inside the worked solution rather than in their own field, and a
    solution may box intermediate values — the final box is the answer. A regex cannot do this:
    answers like ``\\boxed{\\frac{1}{2}}`` nest braces arbitrarily, so this walks the braces.
    """
    marker = solution.rfind("\\boxed{")
    if marker == -1:
        return None
    start = marker + len("\\boxed{")
    depth = 1
    for i in range(start, len(solution)):
        char = solution[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return solution[start:i].strip()
    return None  # unbalanced braces: a malformed row, and the caller skips it


def load_math_train(*, limit: int | None = None, levels: tuple[int, ...] = ()) -> list[Question]:
    """The MATH *training* split — the corpus GSM8K's exhaustion verdict pointed at.

    The original ``hendrycks/competition_math`` hub id has a history of availability trouble, so
    this tries a small chain of ids and reports every failure if all are down — a wrong dataset id
    is otherwise a mid-run 404, the exact failure mode the pinned worker manifest exists to avoid.

    Rows whose solution has no parseable ``\\boxed{}`` are **skipped, counted, and reported** via
    the returned questions' metadata rather than silently dropped: a gold answer that was never
    there scores every rollout zero and reads as a policy collapse.

    Args:
        levels: keep only these difficulty levels (1–5), e.g. ``(3, 4, 5)``. Empty keeps all.
            GSM8K taught us the contested middle is the scarce resource; MATH's level field lets
            the calibration probe start from a band instead of rediscovering it.
    """
    load_dataset = _require_datasets()
    last_error: Exception | None = None
    rows = None
    for dataset_id in ("EleutherAI/hendrycks_math", "hendrycks/competition_math"):
        try:
            if dataset_id == "EleutherAI/hendrycks_math":
                # EleutherAI's mirror splits by subject; load and chain all seven configs.
                subjects = (
                    "algebra",
                    "counting_and_probability",
                    "geometry",
                    "intermediate_algebra",
                    "number_theory",
                    "prealgebra",
                    "precalculus",
                )
                parts = [load_dataset(dataset_id, subject, split="train") for subject in subjects]
                rows = [row for part in parts for row in part]
            else:
                rows = list(load_dataset(dataset_id, split="train"))
            break
        except Exception as exc:  # noqa: BLE001 - collected and re-raised with the full story
            last_error = exc
    if rows is None:
        raise RuntimeError(
            f"no MATH train dataset id worked; last error: {last_error}"
        ) from last_error

    questions: list[Question] = []
    skipped = 0
    for i, row in enumerate(rows):
        level_raw = str(row.get("level", ""))
        level = int(level_raw[-1]) if level_raw and level_raw[-1].isdigit() else None
        if levels and level not in levels:
            continue
        gold = extract_boxed(str(row.get("solution", "")))
        if gold is None:
            skipped += 1
            continue
        questions.append(
            Question(
                id=f"math-train-{i}",
                text=str(row["problem"]).strip(),
                gold=gold,
                source="math-train",
                meta={"level": level, "type": row.get("type"), "unboxed_skipped": skipped},
            )
        )
        if limit and len(questions) >= limit:
            break
    return questions


def load_questions(name: str, *, limit: int | None = None) -> list[Question]:
    """Dispatch by name: ``fixture``, ``gsm8k``, ``math500``, ``math-train``, or a JSONL path."""
    key = name.strip().lower()
    if key in {"fixture", "mini", "mini_math"}:
        return load_fixture(limit=limit)
    if key == "gsm8k":
        return load_gsm8k(limit=limit)
    if key in {"math500", "math-500", "math"}:
        return load_math500(limit=limit)
    if key in {"math-train", "math_train"}:
        return load_math_train(limit=limit)
    return load_jsonl(name, limit=limit)
