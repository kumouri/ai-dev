#!/usr/bin/env bash
# What runs ON the rented box: sync the public repo to a pinned ref, sync the env, exec the chain.
#
#   REPO_URL=https://…/ai-dev.git REPO_REF=<sha-or-branch> bootstrap.sh <command> [args…]
#
# The launcher composes the command (see scripts/cloud_run.py); this script stays deliberately
# dumb — clone-or-pull, uv sync, exec — so the *same* file works whether the box booted the
# project Docker image (repo pre-baked), a provider stock template (bash+git+curl only), or a
# volume-mounted box whose /workspace shadows the baked copy.
#
# set -euo pipefail: any failed step must kill the run so the launcher sees a nonzero exit and
# tears the box down — a half-bootstrapped box that keeps billing is the worst outcome available.
set -euo pipefail

# REPO_URL is required, not defaulted: this public repo carries no account names, its own
# canonical URL included. The launcher derives it from the local checkout's `origin` at runtime.
REPO_URL="${REPO_URL:?REPO_URL is required (public https clone URL of the ai-dev repo)}"
REPO_REF="${REPO_REF:-develop}"
REPO_DIR="${REPO_DIR:-/workspace/ai-dev}"

# The uv cache on the volume: a restarted box re-syncs torch from cache in seconds instead of
# re-downloading gigabytes. Harmless when /workspace is ephemeral — it is just a cache.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/uv-cache}"

# Stock provider templates ship bash+git+curl but not uv.
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --tags origin
# Detached checkout treats branch names, tags, and SHAs identically — REPO_REF may be any of
# them. Branch names need the origin/ fallback: a fresh clone has no local tracking branch yet.
git -C "$REPO_DIR" checkout --detach "$REPO_REF" 2>/dev/null \
    || git -C "$REPO_DIR" checkout --detach "origin/$REPO_REF"

# Sync from the member directory (the extras are coryphaeus's; the virtual root has none) with
# --locked, so the box resolves exactly what CI resolved rather than whatever is newest today.
cd "$REPO_DIR/coryphaeus"
uv sync --locked --extra train --extra data --extra verify
cd "$REPO_DIR"

echo "[bootstrap] $(date -u +%FT%TZ) at $(git -C "$REPO_DIR" rev-parse --short HEAD), exec: $*"
# exec, not a subshell: the chain inherits PID and signals, so an SSH disconnect or provider
# stop reaches the trainer directly instead of orphaning it behind a dead wrapper.
exec "$@"
