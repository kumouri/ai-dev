# Training image for rented GPU boxes (RunPod / Vast / anything that boots a Docker image).
#
# Build from the REPO ROOT — the workspace lockfile lives there:
#
#   docker build -f coryphaeus/cloud/Dockerfile -t <your-registry>/coryphaeus-train:latest .
#
# Push to a registry *you* control and point CORYPHAEUS_CLOUD_IMAGE at it. No canonical public
# tag is baked anywhere in this repo on purpose: the repo carries no account names.
#
# `runtime`, not `devel`, on purpose: torch wheels bundle their own CUDA libraries, so the image
# only needs a driver-compatible CUDA runtime, and the devel image is ~4 GB of nvcc nobody calls.
# 12.8 rather than 12.9+ because marketplace hosts lag on drivers, and a lower runtime floor
# rents more of the market at the same price.
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# build-essential + python3-dev, NOT the CUDA toolkit: Triton JIT-compiles a C shim at first use,
# and a missing compiler/Python.h fails the very first training step with a wall of CUDA flags
# that reads like a driver problem (see CLAUDE.md gotchas / docs/ROADMAP.md phase 2).
# rsync: the launcher pulls artifacts with rsync, which needs the binary on BOTH ends.
# openssh-server: marketplace "direct SSH" runs sshd inside the container (the provider injects
# keys and starts it); proxy-SSH providers simply ignore it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates \
        build-essential python3-dev \
        rsync openssh-server \
    && rm -rf /var/lib/apt/lists/*

# uv, from its official distroless image — the supported copy-in install.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Hardlinks fail across container overlay layers; copy mode silences the warning wall.
ENV UV_LINK_MODE=copy
# Python 3.12 pinned via uv, not the base image's python: same rule as local dev — torch support
# trails new CPython releases, so the interpreter is chosen deliberately.
RUN uv python install 3.12

# The repo bakes at the path bootstrap.sh also targets. If the provider mounts a volume over
# /workspace (RunPod's default), this baked copy is shadowed and bootstrap.sh re-clones onto the
# volume — the bake then only serves volume-less boxes and local `docker run` smoke tests.
WORKDIR /workspace/ai-dev
COPY . .

# Sync from the member directory: the extras belong to coryphaeus, and the virtual workspace
# root has no extras of its own. --locked so the box resolves exactly what CI resolved.
WORKDIR /workspace/ai-dev/coryphaeus
RUN uv sync --locked --extra train --extra data --extra verify

WORKDIR /workspace/ai-dev
# The workspace venv on PATH so `python` and the chain scripts resolve without ceremony.
ENV PATH="/workspace/ai-dev/.venv/bin:${PATH}"

# bash, not a chain script: the launcher drives the box over SSH, and providers commonly replace
# the entrypoint with their own init anyway. The image is an environment, not a job.
ENTRYPOINT ["/bin/bash"]
