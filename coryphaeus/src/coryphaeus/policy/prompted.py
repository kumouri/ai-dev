"""A prompted conductor — the phase-0 policy, and the baseline every trained policy must beat.

No training, no weights: an off-the-shelf small model is *asked* to emit a workflow. If this does
not beat the best single worker in the pool, the reward is aiming at something that does not exist
yet, and that is worth knowing before a GPU is involved.
"""

from __future__ import annotations

import asyncio

from ..schema import MAX_STEPS
from ..workers.base import Worker
from ..workers.governor import Governor
from .prompts import CONDUCTOR_SYSTEM, PROMPT_VERSION, render_conductor_prompt


class PromptedConductor:
    """Uses an ordinary worker as the conductor.

    Args:
        worker: the model doing the routing. Also the resolution of ``"self"`` in a workflow, so a
            self-assigned step really is handled by the conductor.
        governor: enforces the conductor's own provider budget — it competes for units like anything
            else, which is easy to forget and shows up as mysterious 429s.
        temperature: >0 on purpose. A GRPO group needs *diverse* rollouts; a deterministic policy
            yields k identical ones and an advantage of exactly zero.
    """

    def __init__(
        self,
        worker: Worker,
        governor: Governor,
        *,
        temperature: float = 0.8,
        max_tokens: int = 768,
        example_worker: str | None = None,
    ) -> None:
        self.worker = worker
        self.governor = governor
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.example_worker = example_worker
        self.last_results: list = []

    def _example_worker(self, catalog: str) -> str:
        """A concrete worker name for the prompt's example, so the shape is unambiguous."""
        if self.example_worker:
            return self.example_worker
        for line in catalog.splitlines():
            line = line.strip()
            if line.startswith("- "):
                return line[2:].split(" ", 1)[0]
        return "worker"

    async def propose(self, question: str, catalog: str, *, k: int = 1) -> list[str]:
        prompt = render_conductor_prompt(
            question,
            catalog,
            max_steps=MAX_STEPS,
            example_worker=self._example_worker(catalog),
        )
        results = await asyncio.gather(
            *(
                self.governor.invoke(
                    self.worker,
                    prompt,
                    system=CONDUCTOR_SYSTEM,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                for _ in range(k)
            )
        )
        self.last_results = list(results)
        return [r.text for r in results]

    def describe(self) -> dict:
        return {
            "kind": "prompted",
            "worker": self.worker.spec.name,
            "model": self.worker.spec.model,
            "provider": self.worker.spec.provider,
            "prompt_version": PROMPT_VERSION,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class ScriptedConductor:
    """Replays fixed emissions. For tests, and for measuring a hand-written routing strategy."""

    def __init__(self, emissions: list[str], *, name: str = "scripted") -> None:
        if not emissions:
            raise ValueError("ScriptedConductor needs at least one emission")
        self.emissions = emissions
        self.name = name
        self.worker = None
        self.calls = 0

    async def propose(self, question: str, catalog: str, *, k: int = 1) -> list[str]:
        self.calls += 1
        return [self.emissions[i % len(self.emissions)] for i in range(k)]

    def describe(self) -> dict:
        return {"kind": "scripted", "name": self.name, "emissions": len(self.emissions)}
