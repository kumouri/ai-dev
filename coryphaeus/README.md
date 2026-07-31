# Coryphaeus

> *κορυφαῖος* — the leader of the Greek chorus: the one who directs it and speaks for it, without
> being the chorus.

A **small conductor language model**. It never answers a question itself. It reads the question and
emits a *workflow* — up to five steps, each a natural-language subtask assigned to a named worker
model, each declaring which earlier results it gets to see. Workers do the answering. The conductor
learns only how to route.

This is an independent replication of the recipe published in **"Learning to Orchestrate Agents in
Natural Language with the Conductor"** (Sakana AI, ICLR 2026 — [arXiv:2512.04388](https://arxiv.org/abs/2512.04388)),
in which a 7B policy trained with GRPO outscored GPT-5 solo on AIME25, GPQA-Diamond and
LiveCodeBench at a fraction of the cost of heavyweight ensembles. **No weights or code from that
work are used here** — the published weights are not available. This repository re-implements the
described recipe from scratch, and then departs from it (see below).

## Where this departs from the paper

In the paper the conductor learns which-worker-for-what *implicitly*, as a side effect of the policy
gradient. There is no explicit model of worker competence. The intended contribution here is a
**world model over workers**: a predictor of *this worker will succeed on this subtask*, trained on
routing telemetry, used to inform the conductor's context and to prune rollouts. That is why every
step of every rollout is logged as a `(subtask, worker, outcome, cost, latency)` tuple from the very
first run — the dataset is a by-product of the harness, not a later data-collection project.

## Status — phase 0

Harness and baseline, deliberately before any training. A GRPO run optimises whatever the reward
says; until a *prompted* conductor demonstrably beats the best single worker on a real evaluation
slice, a GPU-hour is aimed at an unvalidated target. Phase 0 builds every component the trainer
will reuse and produces that number. Phases and status: [docs/ROADMAP.md](docs/ROADMAP.md).

## Design in one diagram

```
question
   │
   ▼
ConductorPolicy ──emits──▶ raw text ──strict parse──▶ Workflow(≤5 Steps)   [parse failure = reward 0]
                                                          │
                                    ┌─────────────────────┴─────────────────────┐
                                    ▼                                           ▼
                              WorkerRegistry                              Telemetry (JSONL)
                          (catalog + capability tags)                 one row per executed step
                                    │                                  = the world-model dataset
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              FakeWorkerPool   Ollama (local)   Featherless (remote)
                (offline)        :11434         unit-budget governed
                                    │
                                    ▼
                          Orchestrator (steps in index order,
                           each fed only its declared deps)
                                    │
                                    ▼
                          final answer ──▶ Reward (verified math equivalence)
```

Systems detail: [docs/DESIGN.md](docs/DESIGN.md). ML detail: [docs/RESEARCH.md](docs/RESEARCH.md).

## Quickstart

```bash
uv sync                    # from the repo root
uv run pytest              # offline: FakeWorkerPool, bundled fixtures, no network/GPU/key

# Local worker pool (expects Ollama; default http://127.0.0.1:11434, override with OLLAMA_BASE_URL)
uv run python coryphaeus/scripts/smoke_workers.py --local

# Remote worker pool (needs FEATHERLESS_API_KEY in .env — see ../.env.example)
uv run python coryphaeus/scripts/smoke_workers.py --featherless

# The phase-0 experiment: solo arms vs the prompted conductor
uv run python coryphaeus/scripts/run_baseline.py --limit 5 --local-only    # smoke
uv run python coryphaeus/scripts/run_baseline.py --limit 100 --local-only  # the real slice
uv run python coryphaeus/scripts/report.py runs/<stamp>
```

## Cloud training

The GRPO phase rents its GPU: the reward is network-bound, so the cheapest 24 GB card wins —
~$0.34/hr, ≈$2.70 per overnight run (July 2026). Spend is capped by an append-only budget ledger
(`CORYPHAEUS_CLOUD_BUDGET_USD`, default $50/month, fail-closed), and every exit path terminates
the instance. Full runbook — accounts, chains, costs, troubleshooting:
[docs/CLOUD.md](docs/CLOUD.md).

```bash
# what would be rented, at what price, against what budget — provisions nothing
uv run python coryphaeus/scripts/cloud_run.py --provider runpod --chain validate --dry-run

# the ≈$1 end-to-end sanity run: provision → short GSM8K train → checkpoint → terminate
uv run python coryphaeus/scripts/cloud_run.py --provider runpod --chain validate \
    --min-vram 24 --max-price 0.40 --max-hours 4 --volume-gb 30

# the belt on top of auto-terminate: kill anything of ours still running on the account
uv run python coryphaeus/scripts/cloud_run.py --provider runpod --terminate-orphans
```

## Configuration

Everything environment-specific is read from the environment with a documented default; nothing is
hardcoded to a machine. See [`../.env.example`](../.env.example) for the full set.

## License

Apache License 2.0 — see [../LICENSE](../LICENSE). Copyright 2026 Ceryce Armstrong.
