"""The async→sync seam a TRL reward function needs.

TRL calls reward functions **synchronously** from inside the training loop. Scoring a Coryphaeus
rollout is asynchronous and network-bound: parse the workflow, execute it against the worker pool,
verify the answer. So something has to bridge the two, and how it bridges matters:

* **One event loop on one background thread**, reused for the whole run. Creating a loop per call
  (``asyncio.run``) would tear down and rebuild every worker's HTTP connection pool on every step.
* **One ``gather`` per batch**, not per rollout. TRL hands over the whole group at once, and scoring
  8 rollouts serially would make each step 8× slower for no reason — the concurrency governor exists
  precisely to make that batch safe to fire at once.
* **An explicit timeout.** A deadlock between the loop and the governor would otherwise *hang*
  training rather than fail it, and a hung run looks identical to a slow one.

Note what this is *not*: it computes no advantages. TRL does that itself from the returned rewards.
``rollout.group_advantages`` is for offline analysis, not the training path.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence

from ..datasets.loaders import Question
from ..rollout import RolloutRecord, one_rollout
from ..workers.governor import Governor
from ..workers.registry import WorkerRegistry
from ..workflow import DepDefault


class BridgeClosed(RuntimeError):
    """Scoring was attempted after the bridge shut down."""


class RewardBridge:
    """Scores batches of conductor emissions from synchronous code.

    Args:
        registry: the full worker pool. Per-item sub-pools are taken as ``subset()`` views of it, so
            governor state stays shared.
        governor: concurrency budgets and retry policy.
        timeout_s: ceiling on scoring one batch. Exceeding it raises rather than hanging — see the
            module docstring.
        default_deps: how an omitted ``deps`` field is interpreted (see ``workflow.parse``).
        max_tokens: per-worker generation budget. Keep it generous: a tight budget on a reasoning
            model yields a truncated, plausible, wrong answer.
    """

    def __init__(
        self,
        registry: WorkerRegistry,
        governor: Governor,
        *,
        timeout_s: float = 900.0,
        default_deps: DepDefault = "all",
        max_tokens: int = 1024,
    ) -> None:
        self.registry = registry
        self.governor = governor
        self.timeout_s = timeout_s
        self.default_deps = default_deps
        self.max_tokens = max_tokens
        self.batches = 0
        self.rollouts = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # --- lifecycle ------------------------------------------------------------------------------

    def start(self) -> RewardBridge:
        if self._loop is not None:
            return self
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever, name="coryphaeus-reward-loop", daemon=True
        )
        thread.start()
        self._loop, self._thread = loop, thread
        return self

    def close(self) -> None:
        if self._loop is None:
            return
        loop, thread = self._loop, self._thread
        self._loop = self._thread = None
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=10.0)
        loop.close()

    def __enter__(self) -> RewardBridge:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- scoring --------------------------------------------------------------------------------

    def score_records(
        self,
        completions: Sequence[str],
        questions: Sequence[Question],
        pools: Sequence[Sequence[str] | None] | None = None,
    ) -> list[RolloutRecord]:
        """Score a batch, returning full records (not just rewards) so telemetry can be written.

        ``pools[i]`` is the sub-pool that item ``i``'s prompt advertised. It must be honoured: the
        prompt showed the conductor that catalogue, so routing is only legal within it, and
        ``workflow.parse`` turns anything else into a named ``unknown_worker`` failure.
        """
        if self._loop is None:
            raise BridgeClosed(
                "RewardBridge is not running — call start() or use it as a context manager"
            )
        if len(completions) != len(questions):
            raise ValueError(f"{len(completions)} completions vs {len(questions)} questions")

        registries = [
            self.registry if pool is None else self.registry.subset(pool)
            for pool in (pools if pools is not None else [None] * len(completions))
        ]

        async def run_batch() -> list[RolloutRecord]:
            return list(
                await asyncio.gather(
                    *(
                        one_rollout(
                            completion,
                            question,
                            registry,
                            governor=self.governor,
                            rollout_index=index,
                            arm="train",
                            default_deps=self.default_deps,
                            max_tokens=self.max_tokens,
                        )
                        for index, (completion, question, registry) in enumerate(
                            zip(completions, questions, registries, strict=True)
                        )
                    )
                )
            )

        future = asyncio.run_coroutine_threadsafe(run_batch(), self._loop)
        try:
            records = future.result(timeout=self.timeout_s)
        except TimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"scoring a batch of {len(completions)} exceeded {self.timeout_s}s. A hang here is "
                "usually the governor waiting on a unit budget that never frees, or a provider not "
                "answering; raise timeout_s only once you know which."
            ) from exc

        self.batches += 1
        self.rollouts += len(records)
        return records

    def score(
        self,
        completions: Sequence[str],
        questions: Sequence[Question],
        pools: Sequence[Sequence[str] | None] | None = None,
    ) -> list[float]:
        """Scores only — the shape a TRL reward function returns."""
        return [record.score.value for record in self.score_records(completions, questions, pools)]


def make_reward_fn(bridge: RewardBridge, *, on_records=None):
    """Build the callable TRL invokes as a reward function.

    TRL passes the batch's dataset columns through as keyword arguments, each a list aligned with
    ``completions`` — so ``question``, ``gold`` and ``pool`` arrive that way. Column names are read
    defensively rather than assumed, because TRL's calling convention has shifted between releases
    and a silently-missing ``gold`` would score every rollout zero and look like a broken policy.

    Args:
        on_records: optional callback given the full records, for telemetry. Keeping it a callback
            means the reward path itself stays free of file I/O.
    """

    def reward_fn(completions=None, prompts=None, **kwargs) -> list[float]:
        if completions is None:
            raise ValueError("no completions passed to the reward function")
        texts = [_as_text(c) for c in completions]

        golds = _column(kwargs, "gold", "answer", "solution")
        if golds is None:
            raise ValueError(
                "the dataset must carry a 'gold' column — without it every rollout scores zero and "
                f"the policy looks broken. Columns seen: {sorted(kwargs)}"
            )
        texts_q = _column(kwargs, "question", "problem", "text") or [""] * len(texts)
        ids = _column(kwargs, "question_id", "id") or [f"train-{i}" for i in range(len(texts))]
        pools = _column(kwargs, "pool", "workers")

        questions = [
            Question(id=str(qid), text=str(qtext), gold=str(gold), source="train")
            for qid, qtext, gold in zip(ids, texts_q, golds, strict=True)
        ]
        records = bridge.score_records(texts, questions, pools)
        if on_records is not None:
            on_records(records)
        return [record.score.value for record in records]

    return reward_fn


def _as_text(completion: object) -> str:
    """TRL hands completions as strings, or as chat message lists depending on the dataset type."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _column(kwargs: dict, *names: str) -> list | None:
    for name in names:
        value = kwargs.get(name)
        if value is not None:
            return list(value)
    return None
