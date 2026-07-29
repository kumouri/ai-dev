# Coryphaeus — research notes

## The result being replicated

**"Learning to Orchestrate Agents in Natural Language with the Conductor"** — Sakana AI, ICLR 2026,
[arXiv:2512.04388](https://arxiv.org/abs/2512.04388) ([blog](https://sakana.ai/learning-to-orchestrate/)).

As reported:

- A **7B policy** (Qwen2.5-7B) trained with **GRPO** that answers nothing itself. It emits a
  workflow: subtask text, the worker assigned to it, and which prior results that worker sees;
  **≤5 steps**; it may assign **itself** as a worker, giving recursion.
- Workers were GPT-5, Gemini 2.5 Pro, Claude Sonnet 4 and four open 27–32B models.
- The 7B conductor beat **GPT-5 solo**: AIME25 93.3 vs 90.8, GPQA-Diamond 87.5 vs 82.3,
  LiveCodeBench 83.9 vs 82.9 — at roughly **6× less cost** than heavyweight ensembles.
- Training was small: **~960 questions, 200 GRPO iterations, 2×H100** (~160 GB for full-precision
  7B GRPO at 64 rollouts per question). Their stated reason it works at 7B: leaning on powerful
  workers sidesteps the exploration problem that usually strangles small-model RL.
- Weights are not published; the productised version is their **Fugu** API. This repository is a
  replication of the *recipe*, from the paper's description.

> **Provenance caveat, kept deliberately visible.** The figures above entered this repo from a
> research summary written 2026-07-17, not from a close reading of the paper. They are load-bearing
> for the reward design, so **the paper must be re-read before the reward is frozen** (P3 gate in
> [ROADMAP.md](ROADMAP.md)). Anything below that turns out to differ gets corrected here first.

## What we add: a world model over workers

The paper's conductor learns worker competence *implicitly* — nothing in it predicts whether a given
worker will succeed at a given subtask. The intended contribution here is to make that explicit:

> **W(subtask, worker) → P(success), Ê[cost], Ê[latency]**

trained on routing telemetry, and used three ways:

1. **In-context** — the conductor sees predicted success alongside the static catalog, so routing is
   informed rather than only reinforced.
2. **Rollout pruning** — skip rollouts the world model is confident are dead, spending the
   concurrency budget where the gradient actually is. This matters far more on 4 concurrency units
   than on a datacentre.
3. **Reward shaping** (carefully, and last) — a routing decision that beats the world model's
   expectation is more interesting than one that merely got lucky.

Open questions, honestly open: does W generalise across subtask *phrasings* or only topics; is
subtask embedding + worker id enough, or is the question's own embedding required; does a learned W
beat the trivial baseline of per-worker marginal accuracy (it must, or it isn't worth the code).

## Reward design (phase 0)

Verifiable-answer domains only, to start — the reward must be free and unambiguous or the training
signal is a research project of its own:

- **GSM8K** (grade-school arithmetic; easy, dense, good for plumbing) and **MATH500** (harder,
  spread across difficulty). **AIME25 held out** and never trained on.
- Score = `1.0` if the extracted final answer is mathematically equivalent to gold, else `0.0`.
- **Equivalence, not string match.** `1/2`, `0.5`, `\frac{1}{2}` and `2\sqrt3` vs `2*sqrt(3)` must
  all compare correctly. Extraction-and-comparison bugs are the classic reason a GRPO run learns
  nothing while looking healthy, so `reward.py` ships with a table of adversarial pairs as tests.
- **Malformed workflow → 0.0**, with the failure reason recorded. See DESIGN.md for why this is not
  repaired.
- **Cost is recorded but not yet penalised.** The interesting version of this reward is
  accuracy-per-dollar, but a cost term added before accuracy works cannot be debugged. Phase 0
  measures cost so phase 3 can price it.

Deliberately excluded for now: LLM-judged rewards and agentic-dev tasks. A judge is a second
unvalidated model in the loop, and phase 0 exists to remove unvalidated things.

## Training plan (phase 2+, not built yet)

- **GRPO**, group-relative advantage over k rollouts of the *same* question: `A_i = (r_i - mean(r))
  / std(r)`. `rollout.group_advantages()` already emits exactly this, so the trainer plugs into the
  harness rather than reshaping it.
- **Policy sizing against 24 GB.** The paper used ~160 GB for full-precision 7B GRPO at 64
  rollouts/question. So: **Qwen2.5-1.5B-Instruct full GRPO first** (fits, iterates fast, and if
  routing signal exists it should appear at 1.5B), then **7B QLoRA via Unsloth** once 1.5B shows
  signal. The harness is model-agnostic; policy choice is config.
- **Workers = Featherless (flat rate) + local Ollama**, not frontier APIs. GRPO's appetite for
  rollouts makes per-token worker billing the dominant cost, which is exactly what a flat rate
  removes. The binding constraint becomes concurrency units, which is why the governor exists.
- **Rollouts per question** will be far below the paper's 64 at first — 4 concurrency units means k
  is bounded by patience. Worth measuring how low k can go before the advantage estimate is noise.

## Phase-0 hypothesis (the number this phase exists to produce)

> A prompted small conductor over a heterogeneous worker pool beats the **best single worker** in
> that pool on a fixed 100-question slice.

If true, training has a validated target. If false, the finding is more valuable than a training
run would have been, and it constrains the reward: either the pool is too homogeneous for routing to
pay, the catalog gives the conductor too little to route on, or the gains only appear on harder
questions than the slice contains. Each of those is testable next, and none of them is discoverable
after a night of GRPO.
