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
| 0.11 | Verify arXiv:2512.04388 against RESEARCH.md — abstract done; full text before freezing the reward | 🚧 |
| 0.12 | Run the slice; record the baseline number here | 🚧 |

**Exit gate:** a report showing prompted-conductor vs best-single-worker accuracy, cost, latency and
tokens on a fixed slice — plus the answer to whether routing pays at all.

### Run parameters, and what was cut

Stated plainly because the headline number means nothing without them:

- **MATH500, first 40 questions.** Not the 100 originally planned. Local inference on one 4090 runs
  ~15–20 s per question per arm; four arms over 100 questions is several hours, and a smaller
  *complete* comparison beats a larger partial one. The `--chunk` blocking means the number can be
  re-read at any block boundary and the slice extended later without re-running what is done.
- **Pool: `q2`, `q4`, `q9`** (2B / 4B / 9B). `q27` and `g12` were **left out**: together the full
  five-model pool is ~35 GB against 24 GB of VRAM, and `g12` alone took 287 s on a cold load during
  the smoke test. Excluding them makes the pool *less* heterogeneous, which if anything understates
  routing's value — a strong-but-slow worker is exactly what a router should learn to save for hard
  questions. Phase 1 puts them back.
- **Conductor: `q4`, k=1, temperature 0.8, `max_tokens` 1024.** k=1 means no advantage signal; this
  arm measures routing quality, not GRPO readiness.
- **Verifier: the module's own sympy comparison**, with `math_verify` off (see `reward.py` for why —
  it reports `16 == 16*pi` on a bare non-LaTeX gold).

### Result

_Pending 0.12. The number goes here, whichever way it lands._

## Phase 1 — worker pool breadth ⬜

Featherless workers wired and smoke-tested against the real unit throttle; `q27` and `g12` back in;
a pool deliberately heterogeneous in size and speciality, because routing cannot pay in a pool of
near-identical workers. Measure how much the catalog's *wording* moves the baseline — a
prompt-sensitivity number worth having before the reward is frozen.

**Randomised pools, from the paper's abstract.** It trains over randomised agent pools, which is how
its conductor generalises to arbitrary worker sets. So a conductor measured on one fixed pool has
learned that pool, not routing — evaluation needs held-out pool compositions, not just held-out
questions. That is a phase-1 change to the harness (sample the registry per rollout), and it is
cheap to add now and expensive to retrofit after training starts.

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
