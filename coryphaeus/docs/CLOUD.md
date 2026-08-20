# Coryphaeus — cloud training runbook

How to rent a GPU, run the training chain, and get told when the box stops billing — from a fresh
clone, assuming nothing beyond two provider accounts. Phase context lives in
[ROADMAP.md](ROADMAP.md) (phase 4); this document is the operating manual.

All prices in this document are **July 2026** readings. They will drift; the reasoning holds
longer than the numbers, and the dates are here precisely so staleness is visible.

## Why cloud, and why the cheapest card wins

The GRPO reward here is **network-bound**: every rollout is scored by calling remote worker
models, so the GPU idles 50–70% of every optimizer step waiting on the network (measured in
phase 3 — one step spent ~10 minutes wall-clock on ~1 minute of GPU work while its rollout
group's network wall was 53 s). A card that idles most of the time should be the *cheapest* one
that fits the policy, not the fastest one money buys — which makes the right rental a
community-cloud RTX 4090 at ~**$0.34/hr** (July 2026), and an overnight 200-step run roughly
8 h × $0.34 ≈ **$2.70**. The rented box holds only the policy and trainer; workers stay on the
flat-rate API provider, and calibration/evaluation stay local and free.

## What this does not do

Stated up front, in the same spirit as the roadmap's negative results:

- **No multi-GPU.** The policy is a 1.5B full-finetune now, a 7B LoRA later — one card is the
  point. Nothing here schedules across boxes.
- **No spot/interruptible bidding.** On-demand only. At these run lengths (~$3), a preemption
  costs more in restart friction than interruptible pricing saves.
- **No immunity to capacity droughts.** An offer list is a snapshot of a marketplace, not a
  promise. Whole GPU classes go dry at peak hours — the same lesson the worker provider taught
  in phase 3 — and the fix is patience, a looser price cap, or the other provider.
- **No worker management.** Featherless is its own account, key, and (flat-rate) budget; this
  machinery never touches it.

## Accounts and keys

Two providers, deliberately — one is a dependency, two is a market:

1. **RunPod** — create an account, add a little credit, generate an API key in the console, and
   register your SSH **public** key with the account. Community cloud is where the $0.34/hr
   offers live.
2. **Vast.ai** — same shape: account, credit, API key, SSH public key.

**Reliability is a pay-per-use knob, not a third provider.** RunPod's Secure tier
(`CORYPHAEUS_RUNPOD_CLOUD=SECURE`) rents datacenter hosts at roughly 2× the community $/hr —
one env var, no new accounts. The intended pattern is *escalation*: retry wrappers try the
cheap community lottery first and export the tier mid-loop after repeated marketplace failures,
paying double for one night's reliability instead of failing it. A genuinely separate third
provider (Lambda-class) only earns its integration cost when runs get long enough that an
interruption loses real money — revisit at 7B.

Keys go in `.env` at the repo root (see [`.env.example`](../../.env.example)) and **nowhere
else** — never in code, never in a tracked file, never in a shell history you might paste:

```
RUNPOD_API_KEY=...        # required for --provider runpod
VAST_API_KEY=...          # required for --provider vast
CORYPHAEUS_SSH_KEY=~/.ssh/id_ed25519   # optional; this is the default
```

`--provider fake` needs no account at all — it exists so everything above the provider contract
tests offline.

## The budget ledger

Every provisioning attempt passes through a hard monthly gate, defaulting to **$50/month**
(`CORYPHAEUS_CLOUD_BUDGET_USD` raises it — deliberately, in env, not by editing code or the
ledger). The mechanics matter because they decide what a crash costs:

- The ledger is **append-only JSONL receipts**, not a mutable counter. Billing disputes are
  settled by receipts, and a crashed launcher must never be able to *lose* a spend record.
- A **reservation is written before provisioning** at the run's worst case (max runtime × price
  + volume), and **settled after termination** at the actual cost.
- An unsettled reservation keeps counting at its estimate. So a launcher that dies between the
  two **over-counts, never under-counts** — fail-closed, in money's favor. The failure mode
  costs vigilance, not dollars.

Do not hand-edit the ledger to "free up" budget. If the ceiling is genuinely wrong, raise
`CORYPHAEUS_CLOUD_BUDGET_USD`; if a reservation looks stale after a crash, see
[the orphan belt](#auto-terminate-and-the-orphan-belt) below.

**Two ledgers when per-token workers are involved.** GPU rental settles on the cloud ledger
above; rollouts routed to per-token seats (OpenRouter) spend money the box itself cannot meter,
so the *launcher* brackets them on the separate token-spend ledger
(`CORYPHAEUS_TOKEN_BUDGET_USD`, see `coryphaeus/spend.py`): worst case reserved before
provisioning, actuals settled after the pull from the telemetry's own `cost_usd` receipts. Same
append-only, over-count-on-crash rules. Both ledgers are **machine-local files** — launch all
runs that share a month's budget from one machine, or count both ledgers when reading spend.

## The three chains

`scripts/cloud_run.py` runs one of three presets end to end — provision, train, sync, terminate,
settle, notify. Common flags: `--provider runpod|vast|fake`, `--min-vram`, `--max-price`,
`--max-hours`, `--volume-gb`, `--dry-run` — and `--worker-providers` (default `featherless`),
which decides in one place both **which worker API keys ride the provider's env injection** and
the `--providers` filter every train stage runs under. The box gets exactly the selected
providers' keys: it cannot leak a secret it never held, and it cannot quietly train on a wider
pool than the experiment declared.

### `--chain validate` — ≈ $1, and the first thing to run

A short GSM8K training run: a corpus whose behavior on this stack is thoroughly known (see the
roadmap's phase 3), used purely to prove the plumbing — provision → environment → data → train →
checkpoint on the volume → telemetry back → terminate → settle. Roughly $1 at July 2026 prices.
Run this before pointing real money at anything: the first MATH run should debug MATH, not SSH.

```bash
uv run python coryphaeus/scripts/cloud_run.py --provider runpod --chain validate \
    --min-vram 24 --max-price 0.40 --max-hours 4 --volume-gb 30
```

### `--chain probe` — measure before committing

The ~20-step probe run whose telemetry feeds `scripts/gate_zero_std.py`: it measures what
fraction of optimizer steps have zero reward variance (all k rollouts scored identically — steps
that teach nothing) on the *current* question set with the *current* policy. Each launch probes
its own label-derived question window (`--seed` is derived from the run label): before that,
train_grpo's seed defaulted to 0 and every probe graded the *same* 20-question sample — two
gates in a row measured one window twice while the rest of the set went unseen.

### `--chain full` — probe, gate, then the real run

Runs the probe first, applies the gate to its telemetry, and continues into the full run **only
on a PASS**. Gate semantics (from `gate_zero_std.py`, unchanged here):

- **PASS** = *question-dead* zero-variance fraction strictly under **20%** (`--threshold 0.2`;
  exactly 20% is a FAIL — a tie goes to not shipping).
- **Question-dead** counts only groups the policy actually engaged: all k scores identical AND at
  least two rollouts parsed. A zero-variance group with ≤1 parsed rollout is a **parse-storm** —
  the policy failing to emit valid workflows, k times — reported beside the fraction but not in
  it. Measured 2026-08-01/20: the probe policy parse-fails ~half its rollouts, so iid chance
  alone manufactures ~24% zero-at-zero groups on a sound set; two gates FAILed in a row on
  exactly that before the accounting learned to tell the two apart. Unanimous-*correct* groups
  are parsed by construction, so the r4 disease (too-easy questions) stays fully counted.
- **FAIL** = the question set is mis-calibrated: fix the filter, not the trainer, and re-run the
  probe. r4 spent 56% of its GPU time on zero-variance groups; the gate exists so a paid box
  never repeats that.
- **No verdict** (exit 2) when fewer than 12 complete groups exist — a verdict from n=3 is noise
  wearing a badge.

The probe's checkpoints resume into the full run, so a passed gate costs nothing extra.

`--dry-run` works on every chain: it lists current offers, the one it would take, the estimated
worst-case cost, and the budget check — and provisions nothing.

## Auto-terminate, and the orphan belt

A rented GPU that outlives its job is a billing leak, so termination is **structural, not
polite**: every path out of a run — success, failure, timeout, the launcher crashing — ends at
`terminate()`, and the provider contract requires terminate to be **idempotent** (terminating a
terminated instance is a no-op, not an error) precisely because crash handlers call it too.

Structural still isn't absolute — a laptop dying mid-run takes the crash handlers with it. The
belt on top of the suspenders:

```bash
uv run python coryphaeus/scripts/cloud_run.py --provider runpod --terminate-orphans
```

sweeps the account for instances carrying this project's label (`coryphaeus-*`) and terminates
the ones the local ledger does not vouch for: instances the ledger has never heard of, and
instances whose reservation was already settled (a terminate that did not stick). An instance
whose reservation is still **open** is listed but skipped by default — an open reservation may
be a live launcher on another terminal — and swept with `--include-reserved`, which is the flag
to reach for after a crash. Sweeping an open reservation also settles it, at the
provider-reported cost or, when the provider cannot say, at the reservation estimate (the
ledger's rule: over-count, never under-count). The sweep is idempotent and free to run; run it
after any crash, network drop, or moment of doubt, once per provider, and a stale-looking budget
usually rights itself here.

### The artifact pull stalls (or seems to)

The pull is bounded twice per attempt. The hard ceiling is `CORYPHAEUS_PULL_TIMEOUT_S` (default
5400 s — sized so a legitimately slow multi-GB checkpoint pull over a marketplace uplink fits;
the reference healthy pull took 78 minutes). Under it sits the **wedge detector**: if the local
artifact footprint grows by zero bytes for `pull_stall_window_s` (180 s), the attempt is cut
with a `pull_stalled` event and retried — a dead channel is not a slow transfer. Measured
2026-08-20: a wedged ssh channel sat 1h43m moving nothing (and sailed 760 s past the ceiling,
because the only thing watching it was the transfer it was watching), then the retry moved all
27.6 GB in 13.7 s. rsync resumes, so a cut attempt keeps its partial progress. After a second
cut the launcher checks the local artifact dir: files present → the run proceeds and
`pull_timeout_partial` records exactly how many files and bytes landed (verify completeness
before trusting a partial pull); nothing present after a successful payload → the run fails
with `ArtifactPullError`, because the deliverable is lost. Either way the box is terminated and
settled — a stuck transfer costs minutes of patience, never the remaining `max_hours` window.

**The pull is also verdict-aware** (2026-08-20): after a FAILED payload — a gate FAIL included —
it excludes `checkpoints/` and brings home telemetry only. The evidence is a failed run's whole
deliverable; its shards are dead spend (v5 hauled 27.6 GB off a run its own gate had refused —
64% of that run's cost, after the verdict). On a successful payload everything comes home,
because a passed probe's checkpoint is the resume seed. The scp fallback cannot exclude and
says so out loud rather than silently narrowing.

## Checkpoints and resume

The trainer checkpoints every 10 steps (`save_steps=10`, ~≤80 min of exposure at observed step
times, `save_total_limit=3`), and `--resume` picks up from the newest checkpoint. On the cloud
this only helps if checkpoints **outlive the instance**: they are written to the persistent
volume (`--volume-gb`), so a run that dies costs one save interval, not the run — re-provision
and resume rather than restart.

One honesty note on "persistent" (July 2026): on RunPod a network volume survives the pod and
attaches to a new one in the same datacenter; on Vast, storage is tied to the host machine, so
resume-after-reprovision is only as durable as that host's availability. Sync anything you
cannot afford to lose (final checkpoints, telemetry) off the box — RunPod egress is free.

## Notifications

Optional, and built so that having none costs nothing. With `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_ID` in env, the launcher pushes run events (started / finished / failed /
budget-refused) to a Telegram chat; without them it is silently a no-op. Setup is two minutes:
create a bot with @BotFather (free) for the token, message your bot once, and read your chat id
out of the `getUpdates` response.

Two contract points, load-bearing enough to be tested: a notification failure is **logged, never
raised** — it must never kill a training run, because `send()` shares the exit path with
`terminate()` — and the bot token never appears in a log line.

## Costs (July 2026)

| Item | Price | Notes |
|---|---|---|
| RTX 4090 24 GB (RunPod community) | ~$0.34/hr | the 1.5B full-FT phase; ≈$2.70 per 8 h run |
| RTX A6000 48 GB | ~$0.33/hr | the 7B LoRA phase — double the VRAM, same money |
| Persistent volume | ~$0.07/GB/mo | checkpoints live here; 30 GB ≈ $2.10/mo — delete when a phase ends |
| Egress (RunPod) | $0 | syncing checkpoints and telemetry home is free |
| Egress (Vast) | host-set | marketplace hosts price their own network — check the offer |

The Vast caveat is bigger than its egress line: hosts also differ wildly in network quality, and
**this workload's reward is network-bound** — every rollout calls the worker API from the rented
box. A cheap box with bad network is a slow box *here specifically*, so the Vast backend filters
offers on reliability and bandwidth floors, not price alone. A too-good-to-be-true Vast price
usually failed those floors.

## Troubleshooting

### Provisioning times out or the offer vanishes

Offers are marketplace snapshots; between listing and provisioning, someone else can take the
box, and whole GPU classes go dry at peak hours. The launcher now absorbs the common case
itself: a provision refused **for capacity** ("no instances currently available") falls through
to the next-cheapest qualifying offer, up to three tries, each skip recorded as an
`offer_fallback` event — the reservation is priced at the `--max-price` cap, so any qualifying
fallback stays inside it. Schema and auth refusals do NOT fall through; they repeat identically
on every offer and must surface as themselves. If all tried offers are dry: retry (the launcher
re-lists), loosen `--max-price` by a few cents, drop `--min-vram` if you were above the
policy's need, try the other provider, or wait for off-peak. This is a capacity drought, not a
launcher bug — the same weather phase 3 observed on the worker provider.

**Boots are bimodal — so the window's job is failing fast, not patience.** Measured over 30
rentals (2026-08-01): **16 hosts reached `ssh_ready` in 40–90 seconds; 14 never came up at
all.** Nothing lands in between. A host that is silent at five minutes is not slow, it is
dead, and waiting on it buys nothing but wall clock — which is the scarce resource whenever a
run has to finish by a deadline. The default window is 900s: an order of magnitude above every
observed success, tight enough to re-roll ~4 times an hour.

*(An earlier default raised this to 2700s on the theory that ~10 GB cold image pulls were being
guillotined at 1500s — seven same-night failures had all died at exactly that mark. The 2700s
window then produced the same failures 45 minutes later instead of 25, which disproved it: the
deaths clustered at the timeout because that is when we stopped waiting, not because the pull
was nearly done. Worth remembering as a shape of mistake — "every failure happened at exactly
my threshold" reads as a guillotine, but it is equally the signature of a population that
never finishes.)*

**"NVIDIA driver too old" on the very first training step.** The image's CUDA toolkit is not
the constraint — the *host driver* is, and marketplace hosts run whatever driver they run. A
host below the pinned torch's CUDA build passes every other floor, bills a full bootstrap, and
then refuses in the first step — **both providers sold us one the same night** (a 12.8 Vast
host, then a 12.4 RunPod host). The floor is torch's, shared across backends
(`CORYPHAEUS_MIN_CUDA`, default 12.9; the older `CORYPHAEUS_VAST_MIN_CUDA` spelling is honored
as an alias): Vast filters offers on `cuda_max_good`, and RunPod sends `allowedCudaVersions` on
create — which is the entire reason the RunPod backend's pod lifecycle speaks the documented
`rest.runpod.io/v1` dialect (the older `api.runpod.io/v2` pods endpoint 422s the field by
name; unset means "any CUDA version is acceptable" — the docs' words, and the trap). If this
error still appears, the floor has drifted behind the torch pin — raise it in lockstep, not on
a hunch.

**`cudaErrorNoKernelImageForDevice` at the very first kernel launch.** The driver floor's blind
spot: it measures *software*, and a museum-piece GPU behind a freshly-updated driver passes it
clean. Observed live 2026-08-20 — a Tesla P40 (sm_61, Pascal) reporting `cuda_max_good` 13.0
was the cheapest qualifying offer on the whole market, and torch 2.13.0+cu130 ships kernels for
sm_75..sm_120 only, so training died at step 0 (a $0.016 lesson, thanks to the pull-terminate-
settle path working). The *architecture* floor covers it: Vast filters offers on the
marketplace's `compute_cap` field (`CORYPHAEUS_MIN_COMPUTE_CAP`, default 750 = sm_75), enforced
server-side in the query and re-checked client-side like every other floor, missing-is-refused.
750 also refuses the bf16-less sm_70 V100 band sitting just above Pascal in the price ladder.
Like the driver floor, keep it in lockstep with the pinned torch's kernel list — the wheel's
own warning names the sm list it ships.

**A repeat offender in the cheap tier.** A host can *accept* the rental and never boot the
image — pending until the provision timeout fires, sometimes for hours of retries in one night
(2026-07-31: five straight failures, every one on a healthy-scoring host, because a reliability
score never sees "accepted and did nothing"). Every `provisioned` event in
`launcher-events.jsonl` records the offender's identity (`host_id`, `machine_id`); put either id
into `CORYPHAEUS_VAST_EXCLUDE` (comma-separated) and relaunch — the offers query refuses it for
the session. The list is deliberately not persisted: tonight's broken host may be next month's
fine one. And do **not** dodge upward with a higher `--min-vram` instead — the next price tier
up is Tesla V100 territory, and V100s have no bf16, so the trainer fails differently and worse.

### The budget gate refused

The refusal is complete by design. Anatomy of the message:

```
provisioning ~$3.20 would pass the monthly ceiling: $46.30 settled + $2.10 reserved of
$50.00 for 2026-07; resets 2026-08-01. Raise CORYPHAEUS_CLOUD_BUDGET_USD deliberately
if this is intended.
```

- **settled** — terminated runs, at actual cost.
- **reserved** — live runs (and crashed launchers that never settled), at worst-case estimate.
- **resets** — the first of next month; the ledger is monthly.

If *reserved* looks too high after a crash, run `--terminate-orphans` (terminating settles). If
the ceiling itself is the problem, raise the env var — that is what it is for. Never hand-edit
the ledger; it is receipts, and receipts that can be edited are not receipts.

### SSH: first connect

Three distinct failures that all look like "can't get in":

- **Host key prompt / mismatch.** Every provision is a brand-new host with a brand-new host key;
  a strict `known_hosts` policy will balk. This is expected for freshly rented machines, not an
  attack indicator in this context.
- **`Connection refused` right after the instance turns RUNNING.** `sshd` often comes up tens of
  seconds after the provider flips the state. Wait and retry before diagnosing anything.
- **`Permission denied (publickey)`.** The provider injects the **public** key registered in
  your account; the launcher authenticates with the private key at `CORYPHAEUS_SSH_KEY`
  (default `~/.ssh/id_ed25519`). This error means the two do not correspond — key not registered
  with that provider, or the env var points at the wrong file.
