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

### Result — the prompted conductor loses, and the pool is most of the reason

**On the common 150-question set** (all arms scored on identical questions):

| arm | accuracy | unparsed | accuracy on *parsed* | latency/q |
|---|---|---|---|---|
| `q9` (9B) | **58.7%** | — | 58.7% | 32.4 s |
| `q4` (4B) | 55.3% | — | 55.3% | 25.2 s |
| `q2` (2B) | 38.0% | — | 38.0% | 22.9 s |
| **→ conductor** (`q4`) | 47.3% | 14.7% | **55.5%** | **19.0 s** |

**Verdict: −11.3 points against the best single worker.** Held between −9 and −12 from n=50 onward.

**Stopped early, and stated plainly:** the run was cut at 150–175 questions of a planned 400 to free
the GPU for training. Early stopping is only corrupting when it is *favourable* — stopping because
you like what you see. Here the constraint was hardware and the result runs **against** the
conductor, so there is no selection pressure in the number's favour. The pre-registered n was 400;
this is n=150. Arms had drifted to unequal counts (150/151/175/175) by the cut, hence the common-set
comparison rather than a headline built from different slices.

### The finding that matters more than the verdict

Scoring every question against every worker gives an **oracle ceiling** — what a perfect router would
score:

- **51 of 150 questions (34%) were solved by no worker in the pool.** Ceiling = 99/150 = **66.0%**.
- `q9` alone reaches **58.7%**. So the *entire* headroom available to perfect routing is **+7.3
  points**.
- Of the 99 solvable questions the conductor routed 70 to a worker that solved it (**70.7%**) and 29
  to one that did not.

So the negative result decomposes into three separate things, only one of which is about routing
being hard:

1. **Formatting (14.7%).** Unparsed rollouts score zero. On parsed rollouts it is 55.5% — level with
   `q4`. This is what GRPO pressures directly.
2. **Routing (worth ≤ +7.3 points here).** It split 63/63 between `q4` and `q9` when `q9` is the
   better worker — shuffling rather than choosing. Real, and learnable.
3. **The pool (worth far more).** A ceiling only 7.3 points above the best single worker means these
   three same-family Qwen models are *too correlated for routing to pay*. No conductor, trained or
   not, can win much here.

(3) is the load-bearing one, and it is exactly what phase 1 addresses: the remote pool spans 7B–72B
across two lineages **and includes a math specialist**, which should lift the ceiling substantially.
Re-measuring the oracle ceiling on that pool is the first thing worth doing — a routing experiment on
a pool with no headroom cannot succeed, and would look like a conductor failure.

The conductor was also the **fastest** arm (19.0 s vs `q9`'s 32.4 s) by pushing work to cheaper
models. The cost-efficiency half of the paper's result appears even where the accuracy half does not.

## Phase 1 — worker pool breadth 🚧

| # | Item | Status |
|---|---|---|
| 1.1 | `WorkerRegistry.subset()` sharing worker objects | ✅ |
| 1.2 | Deterministic `sample_pool()` + `sample_pool_split()` (held-out **compositions**) | ✅ |
| 1.3 | Catalogue snapshot → pinned manifest with provider-reported units | ✅ |
| 1.4 | Remote pool smoke-tested live | ✅ |
| 1.5 | Catalog-wording sensitivity A/B | ⬜ |
| 1.6 | `q27`/`g12` back in the local pool for a local heterogeneous arm | ⬜ |
| 1.7 | Two-provider pool: six pinned OpenRouter seats + token-spend ledger | ✅ |

**Randomised pools, from the paper's abstract.** It trains over randomised agent pools, which is how
its conductor generalises to arbitrary worker sets. A conductor measured on one fixed pool has
learned that pool, not routing — so evaluation needs held-out **pool compositions**, not only
held-out questions. `sample_pool_split()` provides them, and refuses rather than silently returning
fewer when the registry is too small to supply distinct ones (quietly returning fewer would make a
leak look like a pass).

### What the live catalogue taught us (2026-07-29)

Probing the account beat guessing, three times over:

1. **Units are published per model** (`concurrency_cost`), and the real distribution has four tiers.
   The 24–32B band costs **2**, where `units_for_params` guessed 4 — over-reserving on exactly the
   mid-size workers a router most wants. The heuristic is now a documented fallback only.
2. **Reasoning models are in the remote pool too.** `Qwen/Qwen3-32B` spent 256 tokens thinking and
   answered **25 instead of 42** — a truncated, plausible, *wrong* answer, which is worse than an
   error because it scores as incompetence rather than misconfiguration.
   `chat_template_kwargs={"enable_thinking": false}` fixes it: 6 tokens, correct.
3. **A 4xx can be transient.** The provider returns `400 completion_error` for a generation that
   failed on its side, and `503 capacity_exhausted` while a model spins up. Both must be retried,
   not scored — see below.

### What the July 2026 per-token market taught us (1.7)

The Featherless account's **4 concurrency units were the system-wide throughput ceiling** — during
cloud training the GPU idled 50–70% waiting on workflow execution, and the provider's higher tiers
were not purchasable. The fix was pool diversification: six OpenRouter seats (8B–70B across four
lineages), each **pinned to one upstream** (`allow_fallbacks: false`) because OpenRouter is itself
a router and an unpinned model id can change host and quantization between calls. Probing the live
account beat guessing, again:

1. **The Qwen2.5 mid-band is dead per-token.** Nobody on OpenRouter serves 14B/32B Instruct
   anymore; Featherless keeps those exact legacy checkpoints, so its seats became boutique:
   `qwen25-14b`, `qwen25-32b`, the 72B eval arm, and the **math specialist nobody hosts per-token**.
2. **Qwen3-32B is fp8-only across every OpenRouter upstream.** So the pool carries it twice on
   purpose — `fl-qwen3-32b` (Featherless) vs `or-qwen3-32b` (DeepInfra fp8) — a free
   served-system A/B on quantization that the calibration comparison measures directly.
3. **Per-token workers move the scarce resource from concurrency to money.** OpenRouter seats cost
   1 unit per call and the upstreams take hundreds in flight; what needs governing is spend, and
   with auto-top-up enabled the **token-spend ledger** (`coryphaeus/spend.py`, same reserve/settle
   discipline as the cloud ledger, own $50/mo ceiling) is the only refusal in the pipeline.

### Open design question: infrastructure noise in the reward

Under GRPO a failed rollout scores zero, and zero is how the policy learns *"routing there was a bad
choice."* So a provider hiccup is indistinguishable, to the gradient, from a genuinely poor routing
decision. Transient failures are now retried rather than scored, which handles the common case — but
across three consecutive smoke runs a *different* worker each time hit transient capacity, so on this
provider these are frequent, not rare, and retries can still be exhausted.

The honest options, none of them yet chosen:

- **Raise `max_retries` for training runs** (cheap, partial).
- **Record `infra_failed` in telemetry and measure the rate** before deciding anything — the right
  first step, since the size of the problem is currently unknown.
- **Resample the question rather than score it** when a rollout failed infrastructurally. Correct in
  principle, but it changes what a training batch *is*, so it is a research decision and not a
  silent fix.

**Deliberately not decided here.** Measure first.

## Phase 2 — training environment ✅

Fresh WSL2 **Ubuntu-24.04** on a roomy volume (196 GB VHDX, 185 GB free), leaving the pre-existing
WSL1 distro untouched — WSL1 cannot do CUDA passthrough. Verified inside the distro:

```
kernel      6.18.33.2-microsoft-standard-WSL2      /dev/dxg present
torch       2.13.0+cu130   cuda.is_available() True
device      RTX 4090, capability (8, 9), driver 610.47
bf16 matmul ok
```

**The CUDA toolkit was not needed** — a step this plan originally included and shouldn't have. The
torch wheel bundles its own CUDA 13.0 runtime and reaches the driver through WSL's
`/usr/lib/wsl/lib` passthrough, so `apt install cuda-toolkit` would have been ~3 GB for nothing *and*
carried the real risk it exists to warn about: installing a **driver** package inside WSL clobbers
those passthrough stubs. Add the toolkit only if something genuinely needs `nvcc` (compiling
flash-attn, some Unsloth paths) — not on principle.

**But `python3-dev` *is* needed, which the above nearly hid.** The first training attempt died in
Triton:

```
subprocess.CalledProcessError: ['/usr/bin/gcc', '/tmp/.../cuda_utils.c', ... -I/usr/include/python3.12]
```

Triton JIT-compiles a small C shim at first use, so it needs a C compiler **and Python's development
headers**. Ubuntu ships `python3.12` without them. The misleading part is where it points: the command
line is full of CUDA flags, so it reads as a CUDA/driver problem. It isn't — linking
`-l:libcuda.so.1 -L/usr/lib/wsl/lib` by hand succeeds fine. The missing file is `Python.h`.

So the honest environment list is: **`build-essential` + `python3-dev`**, no CUDA toolkit. Checking
`torch.cuda.is_available()` does *not* catch this — that passes long before anything asks Triton to
compile, which is why the failure landed at step 0 of training rather than during verification.

Caches live inside the distro's ext4 (`~/.cache/huggingface`, `~/.cache/uv`), not on a `/mnt` drvfs
mount: drvfs is slow and breaks the hardlinks the HF cache relies on. The VHDX is already on the
roomy volume, so the space is there either way.

**Not yet verified:** `mem_get_info` with the GPU idle — the phase-0 baseline was holding ~14 GB at
the time, so the reading was 9.4 of 24 GiB free. Re-check once the card is quiet.

## Phase 3 — GRPO on a small policy 🚧

| # | Item | Status |
|---|---|---|
| 3.1 | `train/dataset.py` — rows carrying prompt + gold + **the pool that prompt advertised** | ✅ |
| 3.2 | `train/bridge.py` — async→sync reward seam, one `gather` per batch, explicit timeout | ✅ |
| 3.3 | `train/grpo.py` + `scripts/train_grpo.py` — verified against **TRL 1.9.2** | ✅ |
| 3.4 | 0.5B smoke: advantages non-degenerate, no NaNs, checkpoint written | ✅ |
| 3.4b | Fix 100% completion clipping before the real run | ⬜ |
| 3.5 | Qwen2.5-1.5B real run — r4 (calibrated questions, hardened) | 🚧 |
| 3.6 | Evaluate with `run_baseline.py` unchanged, against the phase-0 table | ⬜ |

### 3.5 post-mortem of run 1 — killed at step 3 of 1000, three separate causes

The first 1.5B run was stopped after 2.7 hours with steps taking 9 → 22 → 35 *minutes* (ETA 415
hours). The GPU held memory but sat idle — "deloaded" to the eye. Telemetry separated three causes:

1. **VRAM squeeze at launch → WDDM spill.** The trainer needs ~12 GB; at launch only 13.7 GB was
   truly free (desktop ambient + lingering allocations). Windows lets CUDA oversubscribe into shared
   system memory instead of failing, so the model *loaded* — into a silent 10–30× slowdown. Step 1
   was already 9 minutes; the run was born spilled, not degraded later. **Lesson: check free VRAM
   against the trainer's need immediately before launch, and treat a slow step 1 as a kill signal,
   not a warm-up.** r2 launched with 20.5 GB free and a step-time gate on the monitor.
2. **The prompt advertised `self` but training wires no self-worker** — TRL owns the policy, so the
   reward function cannot invoke it. 7/16 rollouts died `self_unavailable`: the policy was punished
   for believing its own prompt. Fixed: training prompts render with `include_self=False`; the
   prompted baseline keeps self-assignment (it *does* wire a self-worker, and recursion is the
   paper's mechanism).
3. **The math specialist was capacity-dead for the whole run** — 8/8 calls exhausted retries over
   2.6 h, and Mistral-7B went to capacity when probed as a replacement. At this hour, only the
   mainline Qwen instructs are dependably warm on this provider. The pinned pool is now the four
   reliable workers (14B / 32B / qwen3-32B / 72B-eval-only); specialists come back when they can be
   smoked warm, because **a dead worker feeds infra noise straight into the reward** — every rollout
   that routes to it scores zero for reasons that have nothing to do with routing.

Also sized honestly: even healthy, 1000 optimizer steps at ~1.5–2.5 min/step is 25–40 h. r2 runs
**200 steps** — the paper's own iteration count — which is an overnight run.

### 3.5 post-mortem of runs 2–3, and what r4 changes

**r2** (stopped by hand): born into a squeezed card again — the torch cache high-water grew
13.2 → 23.5 GB, pegged the card, and WDDM began paging (shared 0.77 → 2.02 GB). Step 3 spent
~10 minutes on ~1 minute of GPU work while its rollout group's network wall was 53 s. Telemetry,
not the progress bar, made that attribution possible.

**r3** (the gc experiment): `garbage_collection_threshold:0.8` was added to fight exactly that
cache high-water. The run was *healthy* for 17 steps — then died on
`CUDA error: an illegal memory access` inside generation, with **no checkpoint** (`save_steps=50`
meant 2 h of exposure). Prime suspect is the **combination** `expandable_segments` ×
`garbage_collection_threshold`: this stack ran ~20+ steps across r1/r2/smokes with
expandable_segments alone and never faulted; GC unmapping cached pages an unsynchronized kernel
still references fits the fault, and CUDA reports async faults at sync points — which is why the
traceback pointed at the sampling loop's stopping check rather than the true site. A 5–10-step
discriminator run cannot separate the suspects (the fault needed 17 steps to fire), so r4 simply
**drops gc_threshold** and controls pressure by eviction + headroom instead; if it faults again
without gc, the verdict flips to the cu130 stack and the next move is a cu126 downgrade.

**r3's more valuable finding: 40% of steps taught nothing.** `reward_std: 0` on 4/10 surviving
steps — every rollout in the group scored identically, so the GRPO advantage was zero. The reward
is binary and honest; the *questions* were the problem: one every worker solves (or none solves)
contains no routing decision. TRL 1.9.2 has no dynamic-sampling knob (verified by introspection —
`zero_std` appears in its source once, as the metric), so the fix is at the data layer:
**`scripts/calibrate_questions.py`** probes each candidate with a weak and a strong worker and
keeps the disagreements — the questions where *who you ask changes the outcome* — plus a small
deterministic fraction of both-wrong ones as headroom. The probe rows double as world-model
training data. Expected effect: unanimity from ~40% toward the low teens at unchanged k.

**Latency tail, measured (n=146 calls):** median 7.6 s, p95 27 s, max 88.8 s — a 63× spread, and a
step waits on the slowest of its 4 rollouts, so step time tracks the *max*. r4 caps per-call time
at 60 s (~2.2× p95 — not p95 itself, which would triple retry pressure on a 4-unit budget) and
classifies timeouts and mid-response connection drops as **transient** (`WorkerBusy` → governor
retry), never as scored failures: a stopwatch must not hand out zeros. Median step ~80 s says the
tail, not hardware, is the pacing lever.

### The v2 refine's numbers, and what GSM8K turned out to be

The 6-sample re-probe of the 153 unattempted v1 questions (2026-07-30):

- **79 of 153 (52%) sit at ≥5/6 weak-worker pass rate** — *within the v1 "discriminative" set*.
  Extrapolated to the full 800: the genuinely contested middle of GSM8K for this pool is roughly
  **7%**. GSM8K is substantially saturated for 14B+ workers; r3/r4's dead steps were downstream of
  that fact. Future training sets should draw from MATH-tier difficulty, where the middle is wide.
- These pass rates are the **world model's first labels** — `P(success | worker, question)`
  measured at n=6. The refine now persists every probed rate to a `.rates.jsonl` sidecar (kept or
  not), so refilters never cost a re-probe again. The v2 pass discarded its excluded rates and
  taught us that the hard way.
- **A fencepost shipped in v2:** the band `(0.17, 0.83)` claimed "1–5 of 6" but excluded both
  boundary shells (1/6 = 0.1667, 5/6 = 0.8333). The on-disk v2 set is therefore "2–4 of 6" plus the
  33 measured-mixed — 59 questions total, *more* concentrated than intended rather than broken, and
  the 5/6 exclusion is arguably correct (near-easy is the r4 failure mode). The corrected default
  band is `(0.15, 0.70)` — "1–4 of 6", skewed hard on purpose — with regression tests pinning both
  shells. The r5 gate arbitrates whether the 59 suffice; a FAIL triggers a refilter from the
  sidecar (free) rather than a re-probe.

**Final status (2026-07-30): the corpus is exhausted.** Three gate runs, three fails — **33%,
then 20%, then 22%** zero-variance groups against the 20% bar (the gate is strict-less-than, so
20% does not pass; a tie goes to not shipping). Each refilter between them was free (sidecar
rates, no re-probe) and each moved the number a little; none could clear the bar, because the
extrapolation above was right: GSM8K's genuinely contested middle for a 14B+ pool is ~7%, and no
band over measured rates can manufacture contest that is not in the corpus. Verdict: stop
filtering, change corpus. **MATH-tier is the next training set, and standing it up is the first
cloud workload — see phase 4.**

### Phase 4 validation night (2026-07-31) — the pipeline works, $0.13 of tuition

Six takes against live providers, each failing exactly one layer deeper, every lesson now a pinned
test:

1. RunPod's create-pod 400s without `disk` — mandatory in practice, optional in the schema.
2. (RunPod account 402'd on balance despite auto-top-up — flagged to the account holder; rerouted
   to Vast, which is why two providers exist.)
3. Vast answers non-JSON in ways that must not crash a poll loop (`body_json` everywhere).
4. Vast's bare paths 301 to trailing-slash canonicals; its v0 list endpoint is formally dead
   (410 → v1). Ten minutes of UNKNOWN, $0.02.
5. "Running" ≠ reachable: sshd wakes and keys install after the container starts, and /workspace
   is a RunPod convention a Vast image does not have — the launcher now knocks (retrying
   readiness probe that also mkdir -p's every push destination).
6. Container env does not reach SSH sessions — Vast's canonical `env | grep _ >>
   /etc/environment` onstart line, which our own docstring had been quoting without performing.
7. Disk arithmetic: image + venv + model + ~6 GB/checkpoint ≈ 43 GB against a 30 GB disk. Take 6
   completed 19 cloud training steps and died writing checkpoint-20. Default volume is now 60 GB.

**End-to-end proof:** take 6 provisioned in 33 s, knocked, pushed, cloned at the latest develop,
synced, trained 20 GRPO steps at 88–127 s/step on a $0.12/hr 3090, and its telemetry was pulled
home before terminate — where the gate verdict was computed locally: 6/20 zero-variance = 30%,
FAIL. GSM8K's fourth and final confirmation, this time measured on rented silicon. The corpus
chapter is closed; MATH-tier is the first real cloud workload.

**Checkpointing is now load-bearing:** `save_steps=10` (~≤80 min exposure), `save_total_limit=3`,
`--resume [checkpoint]` wired to TRL's `resume_from_checkpoint`, and `--checkpoint-dir` pointed at
native ext4 — 6–9 GB checkpoints over drvfs/9P are their own slow-motion incident. Dense
checkpoints convert step count from a commitment into a preference: run toward 200, read the
reward curve at 50/100, stop at any checkpoint without loss.

The reward path is **fully testable offline before a GPU is involved**: `train/` imports without torch
or TRL, and the fake pool covers batch ordering, per-item pools, malformed completions, and the
timeout. The pool column is load-bearing — the prompt advertises a catalogue, so scoring must honour
*that* catalogue or routing to a worker the conductor was never shown would be silently accepted.

### The two-arm night (2026-07-31) — first trained OR-pool conductor, and a third corpus verdict

The pool diversification (1.7) was immediately promoted into a **two-arm training experiment**:
the same 500 MATH L3–5 questions, calibrated per pool, each arm gated and trained on its own kept
set. Outcomes:

- **OpenRouter arm: the first conductor ever trained on the OR pool.** v1 single-sample
  calibration gate-FAILED at 25%; the 6-sample refine (115 kept) brought the gate to **15%
  PASS**, and the full 200-step run trained to completion (~79 s/step, network-bound). Checkpoint
  home; MATH500 eval pending.
- **Featherless arm: corpus verdict, not a checkpoint.** Its v1 calibration (155 kept) gated at
  **exactly 20% — FAIL** (a tie goes to not shipping), and the 6-sample refine then showed why:
  **81% of its v1-kept questions are 5/6–6/6 for qwen25-14b** — only 33 genuinely contested
  questions remain. Training 200 steps on 33 questions would be ~24 epochs of memorization, so
  the arm was deliberately not relaunched. **MATH L3–5 is exhausted for the Featherless pool the
  same way GSM8K was** — while the *same questions under the same filter* keep 115 for the OR
  pool. That asymmetry is the served-system comparison's first headline.
- **Single-sample probes overestimate disagreement, measured twice:** v1 said 62% contested for
  the OR pair and 29% for the Featherless pair; 6-sample refinement converged both to ~30% and
  ~7% respectively. Filter noise, not pool truth — the r5 lesson, now with numbers.

The night's baseline work (same MATH500-400 slice as phase 0, all ledgered):

- **phi-4@deepinfra bf16 is the pool flagship:** 70.5% solo at $0.03/400q, beating the 70B
  (64.8%) at 2.5× cheaper — reproduced exactly across two runs.
- **A token cap can manufacture incompetence:** both qwen3 seats burned ~1.1k tokens of
  undisableable reasoning and scored 4.8%/12.2% under a 1024 cap; the per-seat
  `max_tokens_floor` (2048) recovered them to 37.2%/41.8% — still truncating their hardest
  chains, so more capability sits behind a higher floor. Easy smokes cannot catch this; only a
  hard corpus can.
- **Definitive oracle ceiling, floored: 78.5% vs phi-4's 70.5% = +8.0 routing headroom** —
  thin, but better than GSM8K ever offered, and the pool's shape makes the routing question
  concrete: *know when phi-4 fails*, plus the phase-2 cost term.
- A prompted `llama31-8b` conductor routed 289/400 calls to phi-4 from the catalog text alone
  (34.0% — parse failures and its own overhead eat the rest).

Marketplace tuition, all pinned as code the same night (PRs #28–30): host forensics in
`provisioned` events + `CORYPHAEUS_VAST_EXCLUDE` (a host that accepts rentals and never boots
keeps a healthy reliability score); a `cuda_max_good` driver floor (a stale-driver host bills a
full bootstrap before torch refuses it); and the provision timeout raised to 2700 s after seven
"junk hosts" turned out to be honest cold image pulls guillotined at exactly the old 1500 s mark.

**The post-payload pull hang, twice observed — now fixed.** Take 6's launcher hung after its
payload finished (killed by hand, orphan-swept), and the OR arm's pull wedged for six hours
*after transferring everything* — the wall-clock hard kill terminated the box, correctly but
expensively (~$0.45 of idle), and the wrapper's rc=1 then triggered a redundant relaunch that
had to be interrupted by hand (the SIGINT path terminated + settled both ledgers cleanly, which
was good to see proven live). The pull now has its own per-attempt window
(`CORYPHAEUS_PULL_TIMEOUT_S`, default 90 min — the *healthiest* pull of the night took 78) with
one retry (rsync resumes, so a near-end wedge completes in minutes), and after a second timeout
**the local disk gets the last word**: artifacts present → the run proceeds, with a
`pull_timeout_partial` event naming exactly what landed; nothing present on a successful
payload → a failed run, honestly. The transfer's exit status is only a claim; what is on disk
is the ground truth — which is precisely what the six-hour wedge demonstrated.

### What reading the installed TRL actually caught

The plan said to pin TRL and read its real signature rather than trust a remembered API. Two things
turned up that a hardcoded config would have hit an hour into a real run:

1. **`max_prompt_length` does not exist in TRL 1.9.2.** Passing it raises on construction. (Prompts
   here run 1141–1502 chars against a 32k context, so nothing needs truncating anyway.)
2. **The generation batch must be a whole number of groups.**
   `per_device_train_batch_size * gradient_accumulation_steps` must be divisible by
   `num_generations`, or:
   `ValueError: generation_batch_size (4) must be divisible by num_generations (3)`.
   Hit for real with `--k 3`. `gradient_accumulation_steps` is now *derived* from `num_generations`
   by default so any `k` is valid by construction, and an explicit conflicting value is **rejected
   rather than quietly adjusted** — silently changing a number somebody chose is its own bug.

Also worth knowing for later: TRL 1.9's `GRPOTrainer` exposes `rollout_func` and
`environment_factory`. If owning generation as well as scoring turns out cleaner than the
reward-function seam, that is the door.

`--dry-run` now does everything except `trainer.train()` — including constructing the real
`GRPOTrainer` — because construction is exactly where an API change surfaces. Current status against
TRL 1.9.2: trainer builds, **no config setting dropped**.

### 3.4 — the smoke run passed, and found the next problem

Qwen2.5-0.5B-Instruct, 32 GSM8K questions, k=4, 5 steps, ~321 s total.

**Exit gate met.** `reward: 0.25`, `reward_std: 0.5`, **`frac_reward_zero_std: 0`** — every group had
reward variance, so advantages are non-degenerate and there is something to learn from. No NaNs
(`loss ≈ -1.5e-08`, expected for a policy-gradient surrogate at beta=0). Checkpoint written.

**Throughput, as the plan asked:** ~20 s/step early, spiking to 151 s on a later step. The spikes are
provider retries, not compute — confirming the prediction that with remote workers the **network, not
backprop, is the ceiling**. Budget the real run on that basis.

**The problem it exposed: `completions/clipped_ratio: 1`.** Every single completion hit the 512-token
cap, and `completions/mean_terminated_length: 0` — *none* terminated naturally. The policy emits its
JSON block and then rambles to the cap without ever producing EOS. Two consequences:

- The parse successes were **incidental** — a workflow that happened to appear before the truncation.
- Rambling **cost the policy nothing**, so there is no gradient pressure toward terminating.

**`stop_strings` does not fix it in TRL 1.9.2** (tried, not assumed):

```
ValueError: There are one or more stop strings ... but we could not locate a tokenizer.
When generating with stop strings, you must pass the model's tokenizer to `generate`.
```

TRL does not forward its processing class into `generate`, so the knob is unusable from here. Left
**off** rather than shipped as a default that breaks training. Next thing to try: `eos_token_id` in
`generation_kwargs` — token *ids* need no tokenizer at generation time, only at setup. Worth fixing
before 3.5, because a 100% clip rate means the reward signal is measuring the wrong thing.

Watch the parse-failure reason codes as first-class metrics: a policy that stops emitting valid
workflows is the first thing that goes wrong, and phase 0 showed the prompted conductor's entire
deficit was formatting rather than routing. Price the cost term into the reward only once accuracy
moves.

## Phase 4 — cloud training 🚧

GSM8K's exhaustion (phase 3, above) sets the next corpus — MATH-tier questions — and phase 3's
three runs of fighting the desktop's own ambient VRAM (WDDM spill, 13.7 GB "free" that wasn't)
set the venue: the policy moves to a rented box, and the desktop goes back to being free for
calibration and evaluation.

**The decision (July 2026 pricing).** Two independent research passes converged on the same
number: a community-cloud RTX 4090 runs ~**$0.34/hr**. The GRPO reward is network-bound — the
GPU idles 50–70% of every step waiting on worker calls (r2 measured ~1 minute of GPU work inside
a ~10-minute step) — so the *cheapest* card that fits the policy wins, not the fastest, and an
overnight 200-step run is ≈ **$2.70**. Rent-don't-buy falls out of the same utilisation: a
$4,329 RTX 5090 against a rented A100 (~$0.69/hr) breaks even after ~**12 years** at 10 h/week.

**The architecture.** Policy + trainer on the rented box; workers stay on the flat-rate provider
exactly as before (the box's network is now part of the reward path — hence the Vast network
floors in [CLOUD.md](CLOUD.md)); calibration and evaluation stay local and free. The training
code does not change: the box is a location, not a design.

**Four locked decisions:**

1. **Two-provider abstraction.** RunPod and Vast behind the five-verb `CloudProvider` protocol,
   plus a fake backend so everything above the contract tests offline. One provider is a
   dependency; two is a market.
2. **A $50/month enforced ceiling.** Append-only receipts, reservation-before-provision, settle
   after terminate, fail-closed: a crashed launcher over-counts, never under.
   `CORYPHAEUS_CLOUD_BUDGET_USD` raises it deliberately.
3. **Full-FT 1.5B now, LoRA at 7B.** The 1.5B full-finetune fits a 24 GB card; the 7B step keeps
   the old phase-4 gate unchanged — only if 1.5B shows signal, compared at equal rollout
   budget — and goes LoRA on a 48 GB A6000 at essentially the same hourly price.
4. **Validate-then-MATH.** The first paid run is the ≈$1 GSM8K `validate` chain — a corpus whose
   behaviour on this stack is thoroughly known — so the first MATH run debugs MATH, not the
   plumbing.

| # | Item | Status |
|---|---|---|
| 4.1 | `cloud/` scaffold: five-verb provider contract, append-only budget ledger | ✅ |
| 4.2 | Notifications: env-pluggable Telegram, `NullNotifier` default, never raises into a run | ✅ |
| 4.3 | RunPod + Vast backends behind the contract (Vast with network floors) | 🚧 |
| 4.4 | Launcher + `cloud_run.py` chains (`validate` / `probe` / `full`), `--terminate-orphans` | 🚧 |
| 4.5 | Runbook ([CLOUD.md](CLOUD.md)), env documented, gotchas in project memory | ✅ |
| 4.6 | The ≈$1 `validate` run on a real box | ⬜ |
| 4.7 | MATH-tier calibration + the first cloud training run | ⬜ |

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
