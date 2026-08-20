"""How the launcher reaches a rented box: pure command builders + thin async runners.

The builders return argv lists and touch nothing, so tests assert exact SSH/rsync/scp argument
construction with no network and no ssh binary. The runners are deliberately thin — spawn, stream,
return the exit code — because everything with decision content (what to run, what an exit code
means) lives in the launcher, where it is tested against fakes.

Host-key policy, documented once because it looks wrong until the fleet shape is considered:
``StrictHostKeyChecking=accept-new``. These boxes are ephemeral — every run is a first connect to
a host that did not exist an hour ago, so strict checking would prompt on 100% of runs (and
BatchMode turns that prompt into a failure). The tempting alternative ``=no`` is strictly worse:
it also silently accepts a *changed* key for a host already seen, which is the one signal that
actually indicates a man-in-the-middle. Pre-pinning keys is impossible for a box that lives one
run and is gone. ``accept-new`` trusts first contact and still hard-fails if a known host's key
changes mid-run. The launcher additionally points ``UserKnownHostsFile`` at a per-run file so a
provider recycling an IP (with a fresh host key) can never collide with a key recorded by a
*previous* run — and the user's own known_hosts is never polluted with one-run hosts.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
from collections.abc import Sequence
from pathlib import Path

from .providers.base import Instance

#: Env var naming the SSH private-key *path* whose public half is registered with the GPU
#: provider. A path, never key material — the key itself must not exist anywhere this public
#: repo could reach. See docs/CLOUD.md.
SSH_KEYFILE_ENV = "CORYPHAEUS_SSH_KEY"


def default_keyfile() -> Path | None:
    """Keyfile path from the environment, or None to let ssh use its normal resolution
    (agent first, then the default identities — ~/.ssh/id_ed25519 included)."""
    raw = os.environ.get(SSH_KEYFILE_ENV, "").strip()
    return Path(raw).expanduser() if raw else None


def _ssh_options(*, keyfile: Path | None, known_hosts: Path | None) -> list[str]:
    """The ``-o``/``-i`` options shared by ssh, scp, and rsync's transport.

    Port is *not* here on purpose: ssh spells it ``-p`` and scp spells it ``-P``, so each builder
    adds its own.
    """
    options = [
        # Unattended launcher: a password prompt must fail instantly, never hang the run.
        "-o", "BatchMode=yes",
        # See the module docstring for why accept-new and not yes/no.
        "-o", "StrictHostKeyChecking=accept-new",
        # A black-holed TCP connect should surface in seconds, not the kernel's minutes.
        "-o", "ConnectTimeout=30",
        # An 8-hour run over a NAT that silently drops the mapping would otherwise "run" until
        # the wall-clock hard kill. 4 missed probes at 30s = dead in ~2 minutes; the chain's
        # --resume makes failing fast strictly cheaper than hanging until the axe.
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=4",
    ]  # fmt: skip
    if known_hosts is not None:
        options += ["-o", f"UserKnownHostsFile={known_hosts}"]
    if keyfile is not None:
        # IdentitiesOnly: an agent loaded with several keys otherwise offers them all and trips
        # the server's max-auth-tries before reaching the right one.
        options += ["-i", str(keyfile), "-o", "IdentitiesOnly=yes"]
    return options


def build_ssh_cmd(
    host: str,
    command: str,
    *,
    port: int | None = None,
    user: str = "root",
    keyfile: Path | None = None,
    known_hosts: Path | None = None,
) -> list[str]:
    """argv for running ``command`` on the box. ``command`` is one string — ssh joins multiple
    args with spaces anyway, so pre-composing it keeps quoting in exactly one place (the CLI)."""
    argv = ["ssh", *_ssh_options(keyfile=keyfile, known_hosts=known_hosts)]
    if port is not None:
        argv += ["-p", str(port)]
    argv += [f"{user}@{host}", command]
    return argv


def build_rsync_cmd(
    host: str,
    remote_dir: str,
    local_dir: Path,
    *,
    port: int | None = None,
    user: str = "root",
    keyfile: Path | None = None,
    known_hosts: Path | None = None,
    excludes: Sequence[str] = (),
) -> list[str]:
    """argv for pulling ``remote_dir``'s *contents* into ``local_dir``.

    The trailing slash on the source is load-bearing rsync semantics: with it, contents land
    directly in ``local_dir``; without it, an extra directory level appears. ``--partial`` because
    the pull may race a wall-clock kill — a truncated 6 GB checkpoint that can resume beats a
    deleted one. ``excludes`` are rsync patterns skipped by the transfer — how a failed run's
    pull brings home the telemetry without the dead run's 27.6 GB of shards (2026-08-20).
    """
    transport = ["ssh", *_ssh_options(keyfile=keyfile, known_hosts=known_hosts)]
    if port is not None:
        transport += ["-p", str(port)]
    source = f"{user}@{host}:{remote_dir.rstrip('/')}/"
    exclude_args: list[str] = []
    for pattern in excludes:
        exclude_args += ["--exclude", pattern]
    return [
        "rsync",
        "-az",
        "--partial",
        *exclude_args,
        # rsync re-splits the -e string itself, honoring shell-style quotes — so quote each token
        # and a keyfile path containing spaces survives.
        "-e", " ".join(shlex.quote(token) for token in transport),
        source,
        str(local_dir),
    ]  # fmt: skip


def build_scp_cmd(
    sources: Sequence[str],
    dest: str,
    *,
    port: int | None = None,
    keyfile: Path | None = None,
    known_hosts: Path | None = None,
) -> list[str]:
    """argv for scp in either direction; each source/dest is a local path or ``user@host:path``.

    Exists because rsync is not a given on the launcher side (Windows ships an OpenSSH client
    with scp, not rsync) — the runner falls back to this. scp copies a directory *itself* into
    dest rather than its contents; artifacts end up one level deeper than under rsync, which is
    the acceptable cost of a universal fallback.
    """
    argv = ["scp", "-r", "-C", *_ssh_options(keyfile=keyfile, known_hosts=known_hosts)]
    if port is not None:
        argv += ["-P", str(port)]  # capital -P: the classic scp-vs-ssh trap
    argv += [*sources, dest]
    return argv


async def run_streaming(argv: Sequence[str], *, prefix: str = "[remote] ") -> int:
    """Run argv, mirroring its output to stdout line by line, and return the exit code.

    stderr is merged into stdout: one chronological stream is worth more to 7am forensics than
    two neatly separated ones. On cancellation (the launcher's wall-clock hard kill) the local
    child is killed too — otherwise a dead run leaves an ssh client holding the pipe forever.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        assert proc.stdout is not None  # guaranteed by PIPE above; narrows the type
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            print(prefix + line.decode("utf-8", errors="replace").rstrip(), flush=True)
        return await proc.wait()
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()  # reap; instant after kill()


def make_ssh_exec(
    *,
    user: str = "root",
    keyfile: Path | None = None,
    known_hosts: Path | None = None,
):
    """The launcher's remote-exec seam, real edition: run a command on the instance over SSH."""

    async def _exec(instance: Instance, command: str) -> int:
        if not instance.ssh_host:
            raise RuntimeError(f"instance {instance.instance_id} has no ssh endpoint yet")
        argv = build_ssh_cmd(
            instance.ssh_host,
            command,
            port=instance.ssh_port,
            user=user,
            keyfile=keyfile,
            known_hosts=known_hosts,
        )
        return await run_streaming(argv)

    return _exec


def make_rsync_pull(
    *,
    user: str = "root",
    keyfile: Path | None = None,
    known_hosts: Path | None = None,
):
    """The artifact-pull seam, real edition: rsync when available, scp otherwise."""

    async def _pull(
        instance: Instance,
        remote_dir: str,
        local_dir: Path,
        *,
        excludes: Sequence[str] = (),
    ) -> int:
        if not instance.ssh_host:
            raise RuntimeError(f"instance {instance.instance_id} has no ssh endpoint yet")
        local_dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("rsync"):
            argv = build_rsync_cmd(
                instance.ssh_host,
                remote_dir,
                local_dir,
                port=instance.ssh_port,
                user=user,
                keyfile=keyfile,
                known_hosts=known_hosts,
                excludes=excludes,
            )
        else:
            if excludes:
                # scp cannot exclude; the universal fallback pulls everything. Say so rather
                # than silently widening the transfer the caller asked to narrow.
                print(f"[pull] scp fallback cannot honor excludes {list(excludes)}", flush=True)
            argv = build_scp_cmd(
                [f"{user}@{instance.ssh_host}:{remote_dir}"],
                str(local_dir),
                port=instance.ssh_port,
                keyfile=keyfile,
                known_hosts=known_hosts,
            )
        return await run_streaming(argv, prefix="[pull] ")

    return _pull


def make_scp_push(
    *,
    user: str = "root",
    keyfile: Path | None = None,
    known_hosts: Path | None = None,
):
    """The push seam: copy a local file up before the payload runs.

    Exists because run artifacts like calibrated question files live under the gitignored
    ``runs/`` tree — the box cannot clone what the repo does not track, so the launcher carries
    them up.
    """

    async def _push(instance: Instance, local_path: Path, remote_path: str) -> int:
        if not instance.ssh_host:
            raise RuntimeError(f"instance {instance.instance_id} has no ssh endpoint yet")
        argv = build_scp_cmd(
            [str(local_path)],
            f"{user}@{instance.ssh_host}:{remote_path}",
            port=instance.ssh_port,
            keyfile=keyfile,
            known_hosts=known_hosts,
        )
        return await run_streaming(argv, prefix="[push] ")

    return _push
