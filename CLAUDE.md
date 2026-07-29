# ai-dev — Claude project memory

## What this is

A **public** monorepo of AI engineering experiments, one `uv` workspace. First and currently only
member: **`coryphaeus/`** — a small conductor LM that routes subtasks to a worker pool rather than
answering anything itself. See `coryphaeus/docs/` for its design, research notes, and roadmap.

## Public repo — the hygiene rules are structural

This repo is public. Two consequences that bite if forgotten:

- **No machine-specific or personal data in tracked files.** No absolute user paths, no account
  ids, no API keys, no personal context. Anything environment-specific is **config + env with a
  documented default** (see `.env.example`) — which is better design anyway, so there is never a
  reason to hardcode.
- **Every future workspace member inherits the public default.** Adding a member means its
  contents are public from the first push. Check before adding one that touches anything private.

## Conventions

- **Git Flow:** `main` (release) ← `develop` (integration) ← `feat/*`. PRs target `develop`.
- **Merges are merge commits** (`gh pr merge --merge`) — never squash, never rebase.
- **Never merge on red or pending CI.** No exceptions, including failures that look unrelated.
- **Python 3.12** via uv (the host's system Python may be newer than torch supports — do not use it).
- **Ruff** line-length 100, `E,F,W,I,UP,B`. **Pytest** with markers `live` and `data`; CI runs
  neither, so the default suite must stay fully offline.
- **Docs stay in sync with code in the same change.** Stale docs are a bug.

## Layout

```
pyproject.toml            # virtual workspace root: members, shared ruff/pytest config, dev group
coryphaeus/
  src/coryphaeus/         # the package (src layout)
  scripts/                # CLI entry points (smoke_workers, run_baseline, report)
  tests/                  # offline only — FakeWorkerPool, bundled fixtures
  docs/                   # DESIGN (systems), RESEARCH (ML), ROADMAP (phases), adr/
```

## CI

`.github/workflows/ci.yml` runs on PRs into `develop` and `main` (no push triggers — Git Flow) and
does `uv sync --locked --dev` → `ruff check` → `ruff format --check` → `pytest -m "not live and not
data"`. Reproduce it locally with those four commands. `--locked` means a dependency change with a
stale `uv.lock` fails CI rather than resolving differently on the runner.

## Key commands

```bash
uv sync                                                   # workspace venv
uv run pytest                                             # offline suite
uv run pytest -m live                                     # requires a live provider
uv run ruff check . && uv run ruff format --check .
uv run python coryphaeus/scripts/smoke_workers.py --local  # ping the local Ollama pool
uv run python coryphaeus/scripts/run_baseline.py --help
```

## Gotchas

- **Ollama defaults to `127.0.0.1:11434`** in code; if this machine serves it elsewhere, that lives
  in the local `.env` (`OLLAMA_BASE_URL`), never in a tracked default. `smoke_workers.py` prints the
  URL it resolved, so a wrong one is one line of output rather than a mystery.
- **Concurrency units come from the provider, not from a size guess.** Featherless publishes
  `concurrency_cost` per model; the pinned manifest
  (`src/coryphaeus/manifests/featherless_pool.json`, refreshed by `scripts/featherless_catalog.py`)
  carries it. `units_for_params` is a **fallback only** — the 24–32B band costs 2, not 4. A 4-unit
  worker consumes a 4-unit account outright and serializes every other rollout.
- **Reasoning models return reasoning instead of an answer.** Both adapters default to thinking
  *off* (`think=False` → Ollama's `think` field, Featherless's
  `chat_template_kwargs={"enable_thinking": false}`). Left on with a modest token budget, a model
  burns the budget thinking and returns a truncated, plausible, **wrong** answer — which scores as
  incompetence rather than misconfiguration. Observed both locally and remotely.
- **Transient failures must never be scored.** Under GRPO a failed rollout scores zero, and zero
  teaches the policy "that worker was a bad choice" — so a provider hiccup would be laundered into a
  routing lesson. `WorkerBusy` covers 429/502/503/504 **and** the provider's transient error codes,
  which arrive on a `400` (`completion_error`). Permanent failures (403 gated, 404 bad id) are
  returned as outcomes and not retried. See `docs/ROADMAP.md` → "infrastructure noise in the reward".
- **`.gitignore` ignores any directory named `data/`**, at any depth. That is why the pinned manifest
  lives in `manifests/` — a `data/` directory inside the package would silently not be committed.
- **A malformed workflow is a reward signal, not an exception.** Parse failures are recorded with a
  reason and scored zero. Never "helpfully" repair a workflow with a second model call — that
  launders the very error the policy needs to learn from.
