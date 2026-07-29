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
- **Featherless throttles by concurrency units** (small model = 1, 70B+ = 4; a Premium account has
  4 total, then HTTP 429). The registry's governor enforces this; don't bypass it with raw clients.
- **A malformed workflow is a reward signal, not an exception.** Parse failures are recorded with a
  reason and scored zero. Never "helpfully" repair a workflow with a second model call — that
  launders the very error the policy needs to learn from.
