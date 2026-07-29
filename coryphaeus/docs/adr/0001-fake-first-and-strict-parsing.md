# ADR-0001 — Fake-first worker pool, and parse failures as reward signal

- **Status:** accepted
- **Date:** 2026-07-29
- **Context:** phase 0 (harness before training)

## Context

The harness has to talk to two live providers (local Ollama, remote Featherless) and to a policy
whose output is unstructured text from a small model. Both are sources of nondeterminism, and both
cost something to exercise — VRAM and wall-clock locally, concurrency units remotely. Research code
that can only be run against live dependencies does not get tested, and untested harness code is how
a training run silently optimises the wrong thing.

Two decisions follow.

## Decision 1 — the fake worker pool is a first-class module

`workers/fake.py` ships in `src/`, not in `tests/`. It is deterministic (answers hashed from worker
name + prompt + seed), configurable per worker for competence and failure rate, and requires no
network, key or GPU.

**Consequences**

- The default `pytest` run is fully offline: parse → execute → score → group → advantages, end to
  end, with no external call. CI is fast, free and never flaky for network reasons.
- Governor behaviour (unit budgets, N-unit atomic acquisition, 429 backoff) and worker-failure paths
  become deterministically testable. They are not testable against a live provider.
- Because fake workers have *configured* competence, "does routing beat the best single worker?" can
  be validated against a pool whose correct answer is known, before the metric is trusted on real
  workers.
- Cost: two implementations of the `Worker` protocol to keep honest. Mitigated by the protocol being
  five lines and by the real adapters being thin.

Rejected alternative: recorded/replayed HTTP fixtures. They pin behaviour to whatever the provider
did on the day of recording, cannot express competence profiles, and go stale silently.

## Decision 2 — a malformed workflow is scored, not repaired

Parsing runs exactly one *deterministic* repair pass (fenced-block extraction, trailing commas,
quote normalisation, prose stripping) and then fails with a named reason. It never calls a model to
fix a workflow.

**Consequences**

- Under GRPO the policy only learns to emit parseable workflows if unparseable ones cost it reward.
  A model-repair step would launder precisely the error the policy needs to feel — the harness would
  be quietly doing the policy's job, and the trained conductor would degrade the moment the repair
  step were removed.
- Failure reasons (`json_decode`, `too_many_steps`, `unknown_worker`, `forward_dep`,
  `empty_subtask`, …) become first-class training metrics. Rising `unknown_worker` means the catalog
  text is failing; rising `too_many_steps` means the step cap is not reaching the model.
- Cost: the phase-0 baseline will look worse than it could, because a prompted small model emits
  malformed workflows more often than a trained one. That is the honest number, and inflating it
  would corrupt the very comparison the phase exists to make.
- The deterministic repair pass is a deliberate middle ground: it fixes *formatting noise* that is
  an artefact of text generation, not *reasoning errors* about routing.

## Decision 3 — dependencies may only point backwards

A step's `deps` may reference strictly earlier step indices only. Cycles become impossible by
construction, index order is a valid execution order, and a whole class of runtime failure
disappears into a validation rule.

**Consequence:** the conductor cannot express iterative refinement between two steps. It can express
recursion by assigning itself as a worker, which is the paper's mechanism, so nothing in the
replication is lost. Revisit only if a real routing pattern turns out to need a back edge.
