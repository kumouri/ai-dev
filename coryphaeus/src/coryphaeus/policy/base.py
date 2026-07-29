"""The policy contract — the seam a trained checkpoint slots into.

A policy turns (question, worker catalog) into ``k`` raw text emissions. It knows nothing about
validity, execution or scoring; the harness owns those, which is why swapping a prompted model for a
GRPO-trained one touches nothing else.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ConductorPolicy(Protocol):
    """Proposes workflows as raw text."""

    async def propose(self, question: str, catalog: str, *, k: int = 1) -> list[str]:
        """Return ``k`` raw emissions for one question.

        Sampling ``k`` at once (rather than being called ``k`` times) is deliberate: GRPO's
        advantage is computed within a group of rollouts for the *same* question, so the group is
        the natural unit and a policy may batch it however it likes.
        """
        ...

    def describe(self) -> dict:
        """Small JSON-safe dict recorded in run metadata (model, prompt version, temperature…)."""
        ...
