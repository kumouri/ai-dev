# ai-dev

A monorepo of AI engineering experiments — one `uv` workspace, one lint/test config, one CI.

| Member | What it is | Status |
|---|---|---|
| [`coryphaeus/`](coryphaeus/) | A **small conductor language model** that routes subtasks to a pool of workers instead of answering anything itself. Independent replication of the published Sakana Conductor recipe, plus a world model over worker success. | 🚧 phase 0 — harness + baseline |

## Why a monorepo

These experiments share a spine: worker pools, routing, evaluation harnesses, verified rewards.
Keeping them in one workspace means one virtualenv, one ruff config, one CI pipeline, and code
that can be lifted between experiments without a package release.

## Quickstart

```bash
uv sync                                  # workspace venv, all members
uv run pytest                            # offline test suite — no network, no GPU, no API key
uv run ruff check . && uv run ruff format --check .
```

Each member has its own README with its own commands.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Copyright 2026 Ceryce Armstrong.
