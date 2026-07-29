# Coryphaeus — research notes

## The result being replicated

**"Learning to Orchestrate Agents in Natural Language with the Conductor"** —
[arXiv:2512.04388](https://arxiv.org/abs/2512.04388), Stefan Nielsen, Edoardo Cetin, Peter
Schwendeman, Qi Sun, Jinglue Xu, Yujin Tang (Sakana AI; ICLR 2026,
[blog](https://sakana.ai/learning-to-orchestrate/)).

**Confirmed from the paper's own abstract** (verified 2026-07-29):

- A **7B Conductor** trained with **reinforcement learning** that coordinates other LLMs rather than
  answering itself. It learns two things at once: *targeted communication topologies* for
  agent-to-agent collaboration, and *the instructions it writes* for each worker — "prompt
  engineer focused instructions to the LLMs to maximally leverage their individual capabilities."
- It "achieves significant performance gains **beyond any individual worker**", state of the art on
  LiveCodeBench and GPQA. That phrasing is exactly the phase-0 hypothesis below, and the reason the
  baseline arm here is *best single worker* rather than an average.
- **Letting the Conductor select itself as a worker produces recursive topologies** — described as a
  new form of dynamic test-time scaling through online iterative adaptation. Self-assignment is
  therefore load-bearing, not a curiosity, and is supported from the first commit.
- **Training uses randomised agent pools**, which is how it adapts to arbitrary open/closed worker
  sets. Design implication taken up in phase 1: pool composition and the catalog text are
  experimental variables, and a conductor evaluated only on one fixed pool has learned that pool.

**Still second-hand** — from a research summary written 2026-07-17 rather than the full text. Not
contradicted by the abstract, but not verified either, and load-bearing for the reward:

- GRPO specifically as the RL algorithm, and Qwen2.5-7B as the base.
- The ≤5-step cap on workflows (adopted here as `MAX_STEPS`).
- AIME25 93.3 vs GPT-5's 90.8, GPQA-Diamond 87.5 vs 82.3, LiveCodeBench 83.9 vs 82.9; ~6× less cost
  than heavyweight ensembles.
- ~960 training questions, 200 GRPO iterations, 2×H100 (~160 GB at 64 rollouts/question).
- Workers: GPT-5, Gemini 2.5 Pro, Claude Sonnet 4, plus four open 27–32B models.
- The claim that leaning on powerful workers sidesteps small-model RL's exploration problem.

> **Read the full text before the reward is frozen** (gate 0.11 in [ROADMAP.md](ROADMAP.md)).
> Anything above that turns out to differ gets corrected here first.

Weights are not published; the productised version is their **Fugu** API. This repository replicates
the *recipe* as described, using no code or weights from that work.

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
