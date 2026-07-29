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


def load_questions(name: str, *, limit: int | None = None) -> list[Question]:
    """Dispatch by name: ``fixture``, ``gsm8k``, ``math500``, or a path to a JSONL file."""
    key = name.strip().lower()
    if key in {"fixture", "mini", "mini_math"}:
        return load_fixture(limit=limit)
    if key == "gsm8k":
        return load_gsm8k(limit=limit)
    if key in {"math500", "math-500", "math"}:
        return load_math500(limit=limit)
    return load_jsonl(name, limit=limit)
