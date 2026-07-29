"""Prompt templates, versioned.

These are experimental variables, not copy. Every run records ``PROMPT_VERSION`` so results stay
comparable, and any edit that could move a score bumps it.
"""

from __future__ import annotations

#: Bump on any change to the conductor prompt or the worker prompt below.
PROMPT_VERSION = "v1"

CONDUCTOR_SYSTEM = (
    "You are a conductor. You never answer questions yourself — you plan who answers them. "
    "You reply with a JSON workflow and nothing else."
)

CONDUCTOR_TEMPLATE = """\
You are given a question and a list of worker models. Do NOT solve the question. Produce a \
workflow that routes the work to the workers.

Available workers:
{catalog}

Rules:
- At most {max_steps} steps.
- Each step: a `subtask` (plain-language instruction for that worker), a `worker` (a name from the \
list above, or "self" to handle it yourself), and `deps` (a list of earlier step numbers whose \
results that worker should be shown; use [] for none).
- `deps` may only refer to EARLIER steps.
- `final` is the number of the step whose output is the answer.
- Steps are numbered 1, 2, 3... in the order you list them.
- Reply with ONLY a JSON object in a ```json code block. No commentary.

Question:
{question}

Example of the required shape:
```json
{{"steps": [
  {{"subtask": "Solve the problem, showing each step.", "worker": "{example_worker}", "deps": []}},
  {{"subtask": "Check the previous solution for arithmetic errors and give the final answer.", \
"worker": "{example_worker}", "deps": [1]}}
], "final": 2}}
```
"""

WORKER_SYSTEM = "You are a careful, concise problem solver."

WORKER_TEMPLATE = """\
Original question:
{question}
{prior}
Your task:
{subtask}
{final_instruction}"""

PRIOR_HEADER = "\nResults from earlier steps:\n"

#: Appended to the step whose output is scored, so the reward has something to extract.
FINAL_INSTRUCTION = "\nEnd your reply with the final answer in the form \\boxed{answer}."

SOLO_SYSTEM = WORKER_SYSTEM

SOLO_TEMPLATE = """\
{question}

Solve this. End your reply with the final answer in the form \\boxed{{answer}}."""


def render_conductor_prompt(
    question: str, catalog: str, *, max_steps: int, example_worker: str
) -> str:
    return CONDUCTOR_TEMPLATE.format(
        catalog=catalog,
        question=question,
        max_steps=max_steps,
        example_worker=example_worker,
    )


def render_worker_prompt(
    question: str,
    subtask: str,
    priors: list[tuple[int, str]],
    *,
    is_final: bool,
) -> str:
    """Render a step's prompt.

    ``priors`` is ``[(step_index, text), ...]`` for exactly the deps this step declared — the point
    of the whole mechanism is that a step sees only what it asked for.
    """
    prior_block = ""
    if priors:
        parts = [f"[step {idx}] {text.strip()}" for idx, text in priors]
        prior_block = PRIOR_HEADER + "\n\n".join(parts) + "\n"
    return WORKER_TEMPLATE.format(
        question=question,
        prior=prior_block,
        subtask=subtask,
        final_instruction=FINAL_INSTRUCTION if is_final else "",
    )
