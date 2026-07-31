# Coryphaeus — systems design

Scope: the harness. The ML side (reward shape, GRPO, the world model) is in
[RESEARCH.md](RESEARCH.md); phase status is in [ROADMAP.md](ROADMAP.md).

## The one-sentence contract

A **policy** emits text; a **strict parser** turns that text into a `Workflow` or a named failure;
an **orchestrator** executes the workflow against a **worker registry** under a **concurrency
governor**, writing one **telemetry** row per step; a **reward** scores the final answer.

Every one of those is a seam. The policy can be a prompted model today and a GRPO-trained
checkpoint tomorrow without the rest moving.

## Modules

| Module | Responsibility | Deliberately not its job |
|---|---|---|
| `schema.py` | `Step` / `Workflow` dataclasses + all validation rules | parsing, execution |
| `workflow.py` | text → `Workflow` \| `ParseFailure(reason)`; one deterministic repair pass | validation semantics (delegates to `schema`) |
| `workers/base.py` | `Worker` protocol, `WorkerSpec`, `WorkerResult` | knowing about providers |
| `workers/registry.py` | name → worker, the catalog text the conductor reads, the governor | invoking |
| `workers/{fake,ollama,featherless}.py` | one provider each | retry policy (governor owns it) |
| `policy/` | question + catalog → raw workflow text (k samples) | validity, execution |
| `orchestrate.py` | run a workflow, thread deps, aggregate | scoring |
| `reward.py` | final text + gold → score | anything about workflows |
| `rollout.py` | k rollouts per question → GRPO-shaped group + advantages | training |
| `telemetry.py` | append-only JSONL, stable schema | analysis |

## Why the workflow is a DAG that needs no topological sort

A step's `deps` may only reference **strictly earlier** step indices. Cycles are therefore
impossible by construction, and execution in index order *is* a valid topological order. This is a
validation rule doing the work an algorithm would otherwise do, and it removes a whole class of
"the model emitted a cycle" runtime failure.

`deps` omitted defaults to **all prior steps** (configurable: `all` / `prev` / `none`). Rationale:
the common final step is an aggregation, and a chain default would silently starve it of context.
The trained policy's prompt will require explicit `deps`, because *which* results a step sees is
part of what we want it to learn.

## Strict parsing, and why we do not repair with a model

Parsing does exactly one deterministic repair pass — fenced-block extraction, trailing-comma
removal, single→double quote normalisation, stray-prose stripping — and then **fails with a
reason**. It never asks a model to fix the workflow.

A malformed workflow is **signal**. Under GRPO the policy learns to emit parseable workflows only if
unparseable ones cost it reward; a repair step launders exactly the error the policy needs to feel.
So failures are recorded (`json_decode`, `too_many_steps`, `unknown_worker`, `forward_dep`,
`empty_subtask`, …) and scored zero. Every reason code is a metric worth watching during training.

## The concurrency governor

Providers throttle differently and the difference is load-bearing for GRPO, which wants many
parallel rollouts:

- **Featherless** bills concurrency in *units*: a sub-16B model costs 1, a 70B+ model costs 4. A
  Premium account holds **4 units total** and returns HTTP 429 beyond that. So a single 70B call
  saturates the account.
- **Ollama** is local: the real limit is VRAM (24 GB here), not a quota. A model larger than VRAM
  spills to system RAM and becomes very slow, so it gets a unit cost that keeps it effectively
  serial.
- **OpenRouter** bills per *token*, not per concurrent request — the pinned upstreams take
  hundreds in flight. Its unit budget is our politeness bound and the blast radius of a runaway
  loop, not a provider quota. The scarce resource is money instead: every result carries
  `cost_usd` (returned usage × the manifest's pinned prices), and paid runs bracket themselves in
  the **token-spend ledger** (`coryphaeus/spend.py`) — worst case reserved before the first call,
  actuals settled after, refusal past the monthly ceiling. With provider auto-top-up enabled,
  that ledger is the only thing that ever says no.

`UnitPool` is therefore an **N-unit counting semaphore** (an `asyncio.Condition`, because
`asyncio.Semaphore` cannot acquire N atomically — a naive loop of single acquires deadlocks two
concurrent 4-unit requests against each other). One pool per provider; the budget comes from env
(`FEATHERLESS_UNIT_BUDGET`, `OLLAMA_UNIT_BUDGET`, `OPENROUTER_UNIT_BUDGET`).

A 429 backs off **that worker** with exponential delay + jitter, never the whole batch — one
oversubscribed 70B worker must not stall a slice of 2B calls that would have fitted.

## One manifest, two providers — and every OpenRouter seat pins its upstream

The pool is pinned in `src/coryphaeus/manifests/worker_pool.json`, whatever the provider.
Featherless seats are refreshed by `scripts/featherless_catalog.py` (which merges: it replaces
only featherless entries, never another provider's). OpenRouter seats are hand-pinned from live
per-endpoint probes and carry three things a router-of-routers would otherwise hide:

- **`pin`** — the upstream provider the request is locked to (`provider.order` +
  `allow_fallbacks: false`). OpenRouter is itself a router; unpinned, the same model id can be
  served by different hosts with different quantizations on consecutive calls. A calibration
  measured over that is a property of nothing. The adapter also *verifies* provenance
  response-side and fails a mispinned answer by name, zeroing its text so nothing downstream can
  score it — while keeping tokens and cost, because the ledger reports what was spent.
- **`price_in_per_m` / `price_out_per_m`** — required, so a paid worker can never look free.
- **`quantization`** — recorded per seat (policy: bf16 preferred, fp8 allowed). Pass rates are
  properties of the *served system*, so a seat change is a recalibration, and it must be visible
  in this file's diff.
- **`think`** — the reasoning knob is seat data too: `false` sends the disable field; `null`
  omits it, because an upstream with no reasoning control under `require_parameters` 404s the
  entire seat ("No endpoints found"). And an endpoint can accept the knob yet ignore it,
  streaming billed reasoning to a separate field — metered per call (`meta.reasoning_chars`)
  and named as the failure when it starves content, instead of a mute "empty response".

The same principle in one sentence: **a worker is a served system, not a model name** — the
manifest exists to make every property that affects behaviour part of the pin.

## Fake-first

`FakeWorkerPool` is a first-class module, not a test fixture. It is deterministic (answers hashed
from name + prompt + seed), configurable per worker for competence and failure rate, and needs no
network, no key, no GPU. Consequences:

- The **entire** loop — parse → execute → score → group → advantages — is testable offline, so CI
  is fast and free and the default `pytest` run touches nothing external.
- Governor and failure-path behaviour can be tested deterministically, which is impossible against
  a live provider.
- Competence profiles are configurable, so "does routing help?" can be checked against a pool whose
  ground-truth answer is *known* before trusting the metric on real workers.

This mirrors the `FakeSpelunk` simulator in the sibling `lavadream` project, for the same reason:
research code that can only run against the real thing does not get tested.

## Telemetry is the product, not the logging

`runs/<UTC stamp>/` contains `meta.json`, `rollouts.jsonl`, `steps.jsonl`. A `steps.jsonl` row is
one `(question, rollout, step, worker, subtask, deps, tokens, latency, cost, ok, error)` tuple, and
the rollout row carries the score. That is precisely the supervised dataset the world-model lane
needs (features: subtask + worker spec; label: did it contribute to a correct answer).

Both files carry `schema_version`. Cost and latency are recorded even for flat-rate and local
workers, where the marginal dollar cost is zero — the paper's headline result is cost-efficiency,
and a number you did not record is a number you cannot report.
