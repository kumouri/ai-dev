# Coryphaeus — roadmap & status

Legend: ✅ done · 🚧 in progress · ⬜ not started

## Phase 0 — harness + baseline (no GPU, no training)

Everything the trainer will reuse, plus the number that justifies training at all.

| # | Item | Status |
|---|---|---|
| 0.1 | Repo scaffold: workspace, license, docs, CI | ✅ |
| 0.2 | `schema.py` + `workflow.py` — strict parse, named failure reasons | ✅ |
| 0.3 | Worker layer: protocol, `FakeWorkerPool`, Ollama, Featherless | ✅ |
| 0.4 | Registry + unit-aware concurrency governor with per-worker 429 backoff | ✅ |
| 0.5 | `orchestrate.py` — dep-threaded execution, failure paths | ✅ |
| 0.6 | `reward.py` — math equivalence + adversarial test table | ✅ |
| 0.7 | `telemetry.py` — versioned JSONL; the world-model dataset | ✅ |
| 0.8 | `rollout.py` — k-rollout groups + GRPO advantages | ✅ |
| 0.9 | Dataset loaders (GSM8K, MATH500) + bundled offline fixture | ✅ |
| 0.10 | `scripts/`: `smoke_workers`, `run_baseline`, `report` | ✅ |
| 0.11 | **Re-read arXiv:2512.04388 and correct RESEARCH.md before freezing the reward** | ⬜ |
| 0.12 | Run the 100-question slice; record the baseline number here | ⬜ |

**Exit gate:** a report showing prompted-conductor vs best-single-worker accuracy, cost, latency and
tokens on a fixed slice — plus the answer to whether routing pays at all.

### Result

_Pending 0.12. The number goes here, whichever way it lands._

## Phase 1 — worker pool breadth ⬜

Featherless workers wired and smoke-tested against the real unit throttle; a pool deliberately
heterogeneous in size and speciality (routing cannot pay in a pool of near-identical workers). Fix
the catalog text the conductor reads, and measure how much its wording moves the baseline — a
prompt-sensitivity number worth having *before* the reward is frozen.

## Phase 2 — training environment ⬜

Fresh WSL2 Ubuntu-24.04 (CUDA passthrough; the existing WSL1 distro cannot), distro and Hugging Face
cache on a roomy volume rather than the system drive. torch + TRL, `nvidia-smi` and `torch.cuda`
verified inside the distro, then an overfit-one-batch sanity run before any real training.

## Phase 3 — GRPO on a small policy ⬜

Qwen2.5-1.5B-Instruct, full GRPO, workers from phase 1, reward from phase 0. Watch the parse-failure
reason codes as first-class metrics: a policy that stops emitting valid workflows is the first thing
that goes wrong. Price the cost term into the reward once accuracy moves.

## Phase 4 — scale the policy ⬜

7B QLoRA via Unsloth, only if 1.5B showed signal. Compare against phase 3 at equal rollout budget.

## Phase 5 — the world model over workers ⬜

Train `W(subtask, worker) → P(success)` on phase 0–4 telemetry. First check it beats the trivial
per-worker marginal-accuracy baseline; then feed it into the conductor's context, then use it to
prune rollouts. This is the contribution the paper does not cover — see
[RESEARCH.md](RESEARCH.md#what-we-add-a-world-model-over-workers).

## Deferred / not doing yet

- **LLM-judge rewards** and **agentic-dev tasks** — a second unvalidated model in the loop.
- **Frontier-API workers** — metered per rollout, which is what the flat rate exists to avoid.
- **Routing across architectures** (attention vs post-attention/RWKV-converted models) — a genuinely
  interesting use of a conductor: learning when lossy linear-state memory is good enough. Belongs to
  a sibling workspace member, not here.
