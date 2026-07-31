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
that teach nothing) on the *current* question set with the *current* policy.

### `--chain full` — probe, gate, then the real run

Runs the probe first, applies the gate to its telemetry, and continues into the full run **only
on a PASS**. Gate semantics (from `gate_zero_std.py`, unchanged here):

- **PASS** = zero-variance fraction strictly under **20%** (`--threshold 0.2`; exactly 20% is a
  FAIL — a tie goes to not shipping).
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
box, and whole GPU classes go dry at peak hours. In order: retry (the launcher re-lists),
loosen `--max-price` by a few cents, drop `--min-vram` if you were above the policy's need, try
the other provider, or wait for off-peak. This is a capacity drought, not a launcher bug — the
same weather phase 3 observed on the worker provider.

**Before blaming hosts, check whether every death happened at exactly your timeout.** Seven
"junk hosts" in one night (2026-07-31) all died pending at precisely the then-default 1500s —
which is not what broken hosts look like, it is what honest cold image pulls look like when
guillotined at 96%: ~10 GB at peak-hour registry speeds needs more than 25 minutes, and the
"good" hosts were merely warm ones. A pending box costs pennies per extra ten minutes, while a
re-roll repeats the cold pull from zero on a different cold host. The default is now 2700s;
raise it before concluding the market is broken.

**"NVIDIA driver too old" on the very first training step.** The image's CUDA toolkit is not
the constraint — the *host driver* is, and marketplace hosts run whatever driver they run. A
host below the pinned torch's CUDA build passes every network/reliability floor, bills a full
bootstrap (~50 minutes at rental rates, observed 2026-07-31), and then refuses in the first
step. `offers()` now filters on Vast's `cuda_max_good` (floor `CORYPHAEUS_VAST_MIN_CUDA`,
default 12.9); if this error still appears, the floor has drifted behind the torch pin — raise
it in lockstep, not on a hunch.

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
