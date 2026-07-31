"""Rent a GPU box, run the training chain on it, pull artifacts, terminate. The CLI.

    # the first-run choice: 20-step probe + zero-variance gate, then the box comes down
    uv run python coryphaeus/scripts/cloud_run.py --provider runpod --chain validate \\
        --questions-file coryphaeus/runs/calibrated-gsm8k.jsonl

    # look before renting: cheapest qualifying offer + what it would reserve, then refuse
    uv run python coryphaeus/scripts/cloud_run.py --provider vast --dry-run

    # the belt for the auto-terminate suspenders
    uv run python coryphaeus/scripts/cloud_run.py --provider runpod --terminate-orphans

Money flows through the budget ledger (a hard monthly gate), the box is terminated from every
exit path, and the whole run leaves a forensic events trail — see cloud/launcher.py.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from coryphaeus.cloud.budget import BudgetExceeded, BudgetLedger
from coryphaeus.cloud.launcher import (
    LauncherError,
    LaunchSpec,
    NoOfferError,
    launch,
    pick_offer,
    worst_case_usd,
)
from coryphaeus.cloud.providers.base import CloudProvider, Instance
from coryphaeus.config import REPO_ROOT, settings
from coryphaeus.pools import load_manifest
from coryphaeus.spend import EST_TOKENS_IN, sum_cost_usd, token_ledger

#: Where things live ON the box. /workspace because that is where the target providers mount the
#: persistent volume — code, caches, and artifacts all survive a container restart there.
REMOTE_REPO_DIR = "/workspace/ai-dev"
REMOTE_RUNS_DIR = "/workspace/runs"
REMOTE_QUESTIONS = "/workspace/questions.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", required=True, choices=("runpod", "vast", "fake"))
    parser.add_argument("--min-vram", type=int, default=24, help="GB; the policy needs 24")
    parser.add_argument("--max-price", type=float, default=0.60, help="$/hr ceiling for offers")
    parser.add_argument(
        "--max-hours",
        type=float,
        default=8.0,
        help="drives BOTH the budget reservation and the wall-clock hard kill",
    )
    # 60, not 30: the disk holds ~20 GB of unpacked image + ~8 GB venv + ~3 GB model cache +
    # ~6 GB PER CHECKPOINT (1.5B with optimizer state, save_total_limit=3). Take 6 finished 19
    # training steps and died writing checkpoint-20 at 30 GB: "No space left on device".
    parser.add_argument("--volume-gb", type=int, default=60)
    parser.add_argument(
        "--provision-timeout",
        type=float,
        default=2700.0,
        help="seconds to wait for the box to reach RUNNING before terminating. Size it to the "
        "image: ~10 GB over a marketplace link needs 15-25 min cold — and at peak hours more. "
        "2026-07-31: seven 'junk hosts' in one night all died at exactly the old 1500s mark, "
        "i.e. honest cold pulls guillotined at 96%%; the two hosts that 'worked' were merely "
        "warm. A pending box costs pennies per extra 10 minutes; a re-roll repeats the cold "
        "pull from zero on a different host.",
    )
    parser.add_argument(
        "--chain",
        choices=("probe", "full", "validate"),
        default="validate",
        help="probe = 20-step probe only; validate = probe + zero-variance gate (the first-run "
        "choice); full = probe + gate + deadline-bounded full run resuming the probe's checkpoint",
    )
    parser.add_argument(
        "--questions-file",
        default="",
        help="local calibrated-questions JSONL (scripts/calibrate_questions.py output). It lives "
        "under gitignored runs/, so the launcher pushes it up rather than expecting a clone to "
        "have it. Without it the chain trains on raw GSM8K — see that script for why not to.",
    )
    parser.add_argument("--k", type=int, default=4, help="rollouts per question")
    parser.add_argument("--steps", type=int, default=200, help="full-run optimizer steps")
    parser.add_argument("--gate-threshold", type=float, default=0.2)
    parser.add_argument(
        "--worker-providers",
        default="featherless",
        help="comma-separated worker providers the box will use (featherless, openrouter). The "
        "launcher ships exactly these providers' API keys and passes the same list to every "
        "train stage — the pool a box may use is exactly the pool it can authenticate to. "
        "Per-token providers additionally get a worst-case token-ledger reservation, settled "
        "from the pulled telemetry's own cost receipts.",
    )
    parser.add_argument(
        "--image",
        default="",
        help="Docker image for the box; defaults to $CORYPHAEUS_CLOUD_IMAGE. Build "
        "coryphaeus/cloud/Dockerfile and push it to a registry you control — any image with "
        "bash, git, and curl also works (bootstrap.sh installs uv and clones the repo).",
    )
    parser.add_argument(
        "--repo-url",
        default="",
        help="public https clone URL the box fetches; defaults to $CORYPHAEUS_REPO_URL, then "
        "this checkout's origin remote. Never hardcoded — this repo carries no account names.",
    )
    parser.add_argument(
        "--repo-ref", default="", help="ref the box trains from; defaults to $REPO_REF or develop"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the offer it would take and the money it would reserve, then refuse",
    )
    parser.add_argument(
        "--terminate-orphans",
        action="store_true",
        help="list provider instances labeled coryphaeus-* and terminate the ones the local "
        "ledger does not vouch for — the belt for the auto-terminate suspenders",
    )
    parser.add_argument(
        "--include-reserved",
        action="store_true",
        help="with --terminate-orphans: also sweep instances whose reservation is still open "
        "(default skips them — an open reservation may be a live launcher on another terminal)",
    )
    return parser.parse_args(argv)


def load_provider(name: str) -> CloudProvider:
    """Import a backend lazily and find its entry point without hardcoding its API.

    Lazy because backends may pull provider SDK bits this CLI should not require for --help, and
    tolerant (build() factory, else a no-arg *Provider class) because the backends land
    independently of this launcher.
    """
    module = importlib.import_module(f"coryphaeus.cloud.providers.{name}")
    build = getattr(module, "build", None)
    provider: object | None = build() if callable(build) else None
    if provider is None:
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and value.__module__ == module.__name__
                and value.__name__.lower().endswith("provider")
            ):
                try:
                    candidate = value()
                except TypeError:
                    continue  # needs constructor args we cannot guess; keep looking
                if isinstance(candidate, CloudProvider):
                    provider = candidate
                    break
    if provider is None or not isinstance(provider, CloudProvider):
        raise SystemExit(
            f"coryphaeus.cloud.providers.{name} exposes neither a build() factory nor a no-arg "
            "*Provider class implementing the CloudProvider protocol"
        )
    return provider


def resolve_repo_url(explicit: str) -> str:
    """--repo-url, else env, else this checkout's origin (normalized to anonymous https)."""
    url = explicit.strip() or os.environ.get("CORYPHAEUS_REPO_URL", "").strip()
    if not url:
        try:
            url = subprocess.run(
                ["git", "-C", str(REPO_ROOT), "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            url = ""
        # The box clones anonymously — an ssh remote would demand a key it must not have.
        if url.startswith("git@github.com:"):
            url = "https://github.com/" + url.removeprefix("git@github.com:")
    if not url:
        raise SystemExit("cannot determine the repo URL — pass --repo-url or CORYPHAEUS_REPO_URL")
    return url


def build_chain(args: argparse.Namespace, train_label: str) -> str:
    """The command sequence the box runs, joined with && so any failed stage stops the chain."""
    # --no-sync: bootstrap.sh already synced with the train extras; a bare `uv run` would re-sync
    # to the default (extra-less) set and strip torch right out from under the trainer.
    run = "uv run --no-sync python"
    questions = f" --questions-file {REMOTE_QUESTIONS}" if args.questions_file else ""
    checkpoints = f" --checkpoint-dir {REMOTE_RUNS_DIR}/checkpoints"
    # The pool a box may use is exactly the providers whose keys ride provision(env=...) — the
    # same --worker-providers value drives the pushed env (see worker_env) and this filter, so
    # they cannot drift apart. The manifest is multi-provider; an unfiltered registry build on
    # the box would demand keys it must not have.
    providers = f" --providers {args.worker_providers}"

    probe = (
        f"{run} coryphaeus/scripts/train_grpo.py --label {train_label} --k {args.k} "
        f"--max-steps 20{checkpoints}{questions}{providers}"
    )
    gate = (
        f"{run} coryphaeus/scripts/gate_zero_std.py --label {train_label} "
        f"--threshold {args.gate_threshold}"
    )
    # The trainer's polite deadline must undercut the launcher's hard kill by enough to save a
    # checkpoint and rsync it home — the probe, sync, and pull all live inside max_hours too.
    deadline = max(0.5, args.max_hours - 1.5)
    full = (
        f"{run} coryphaeus/scripts/train_grpo.py --label {train_label} --k {args.k} "
        f"--max-steps {args.steps} --deadline-hours {deadline:.2f}{checkpoints}{questions} "
        f"--resume{providers}"
    )
    return {
        "probe": probe,
        "validate": f"{probe} && {gate}",
        "full": f"{probe} && {gate} && {full}",
    }[args.chain]


#: Worker provider name → (env var, Settings attribute). The single map both the key check and
#: the pushed env draw from, so "which keys does the box get" has exactly one answer.
WORKER_KEYS = {
    "featherless": ("FEATHERLESS_API_KEY", "featherless_api_key"),
    "openrouter": ("OPENROUTER_API_KEY", "openrouter_api_key"),
}


def parse_worker_providers(raw: str) -> tuple[str, ...]:
    providers = tuple(p.strip() for p in raw.split(",") if p.strip())
    unknown = set(providers) - set(WORKER_KEYS)
    if not providers or unknown:
        raise SystemExit(
            f"--worker-providers must name at least one of {sorted(WORKER_KEYS)}; "
            f"got {raw!r}" + (f" (unknown: {sorted(unknown)})" if unknown else "")
        )
    return providers


def worker_env(cfg, providers: tuple[str, ...]) -> dict[str, str]:
    """The worker API keys the box gets — exactly the selected providers', nothing more.

    A missing key fails HERE, before a box is rented: a chain that would exit 2 at its first
    registry build wastes minutes and cents. And an unselected provider's key is never pushed —
    a box cannot leak a secret it never held.
    """
    env: dict[str, str] = {}
    for provider in providers:
        var, attr = WORKER_KEYS[provider]
        value = getattr(cfg, attr) or ""
        if not value:
            raise SystemExit(
                f"{var} is not set but --worker-providers includes {provider!r}. The chain "
                "routes workers remotely; renting a box that will exit 2 at its first step "
                "wastes minutes and cents."
            )
        env[var] = value
    return env


def token_reserve_usd(
    entries: list[dict], *, chain: str, steps: int, k: int, providers: tuple[str, ...]
) -> float:
    """Worst-case token spend for a training chain, from the manifest's own pinned prices.

    Every rollout is charged the maximum five workflow calls, each at the priciest selected
    per-token seat, for every step of every stage in the chain (the probe's 20 plus the full
    run's budget). Deliberately generous — the settle reports what the pulled telemetry says
    actually happened. Flat-rate-only selections cost 0.
    """
    per_call = max(
        (
            (float(e["price_in_per_m"]) * EST_TOKENS_IN + float(e["price_out_per_m"]) * 1024)
            / 1_000_000
            for e in entries
            if e.get("provider") in providers and e.get("price_out_per_m") is not None
        ),
        default=0.0,
    )
    steps_total = 20 + (steps if chain == "full" else 0)
    return round(per_call * steps_total * k * 5, 4)


def build_remote_command(*, repo_url: str, repo_ref: str, chain: str) -> str:
    """The single string handed to SSH.

    Bootstrap-from-nothing: clone if the repo is absent (a stock image has no bootstrap.sh to run
    yet — the clone is what delivers it), then hand off to bootstrap.sh, which pins the ref and
    syncs the env. `bash -l` so provider-injected env (the API key rides provision(env=...), and
    providers surface it via profile.d) reaches the chain. Secrets never appear in this string.
    """
    ensure = (
        '[ -d "$REPO_DIR/.git" ] || git clone "$REPO_URL" "$REPO_DIR"; '
        f'exec bash "$REPO_DIR/coryphaeus/cloud/bootstrap.sh" bash -c {shlex.quote(chain)}'
    )
    exports = (
        f"REPO_URL={shlex.quote(repo_url)} REPO_REF={shlex.quote(repo_ref)} "
        f"REPO_DIR={shlex.quote(REMOTE_REPO_DIR)} "
        f"CORYPHAEUS_RUNS_DIR={shlex.quote(REMOTE_RUNS_DIR)}"
    )
    return f"{exports} bash -lc {shlex.quote(ensure)}"


def make_notify():
    """Env-configured Telegram notify, or None. Pluggable and public-safe: two env vars, no
    private daemon plumbing, and the launcher treats notify failures as non-fatal."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not (token and chat_id):
        return None

    async def _notify(text: str) -> None:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={"chat_id": chat_id, "text": text},
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Sanitized on purpose: httpx error text embeds the request URL, and this URL embeds
            # the bot token — which must never reach a log line. `from None` keeps the original
            # (token-bearing) exception out of the chain entirely.
            raise RuntimeError(f"telegram sendMessage -> HTTP {exc.response.status_code}") from None
        except httpx.HTTPError as exc:
            raise RuntimeError(f"telegram sendMessage failed: {type(exc).__name__}") from None

    return _notify


def _instance_view(entry: object) -> tuple[str, str, str]:
    """(instance_id, label, state) from whatever shape a backend's list_instances() yields."""
    if isinstance(entry, Instance):
        raw = entry.raw if isinstance(entry.raw, dict) else {}
        return entry.instance_id, str(raw.get("label", "")), entry.state.value
    if isinstance(entry, dict):
        instance_id = str(entry.get("instance_id") or entry.get("id") or "")
        return instance_id, str(entry.get("label", "")), str(entry.get("state", "unknown"))
    return "", "", "unknown"


async def terminate_orphans(
    provider: CloudProvider, ledger: BudgetLedger, *, include_reserved: bool
) -> int:
    """Sweep coryphaeus-labeled instances the ledger does not vouch for.

    The launcher's try/finally is the suspenders; this is the belt, for the cases structure
    cannot reach — a kill -9'd launcher, a terminate() the provider acknowledged and ignored, a
    box provisioned from a machine whose ledger this is not.
    """
    lister = getattr(provider, "list_instances", None)
    if lister is None:
        print(
            f"{provider.name}: the backend exposes no list_instances() (the five-verb protocol "
            "has no listing op). Sweep orphans from the provider console instead.",
            file=sys.stderr,
        )
        return 2
    rows = ledger.rows()
    estimates = {
        r["reservation_id"]: float(r["estimated_usd"]) for r in rows if r["event"] == "reserve"
    }
    settled = {r["reservation_id"] for r in rows if r["event"] == "settle"}
    open_reservations = set(estimates) - settled

    swept = skipped = 0
    for entry in await lister():
        instance_id, label, state = _instance_view(entry)
        if not label.startswith("coryphaeus-") or not instance_id:
            continue  # not ours — never touch other tenants' (or other projects') boxes
        if state == "terminated":
            continue
        if label in open_reservations and not include_reserved:
            skipped += 1
            print(
                f"SKIP {instance_id} ({label}, {state}): reservation still open — a launcher may "
                "be live. Re-run with --include-reserved to sweep it anyway."
            )
            continue
        await provider.terminate(instance_id)
        swept += 1
        if label not in estimates:
            why = "unknown to the local ledger"
        elif label in settled:
            why = "settled but still alive — a terminate that did not stick"
        else:
            why = "open reservation, swept on request"
            cost = None
            try:
                cost = await provider.cost_so_far(instance_id)
            except Exception as exc:  # a sweep must not die on a cost query
                print(f"  cost_so_far failed ({exc!r}); settling at the reservation estimate")
            # Unknown cost settles at the reservation estimate, not zero: the ledger's rule is
            # over-count, never under-count.
            ledger.settle(
                label,
                float(cost) if cost is not None else estimates[label],
                note="orphan sweep",
            )
        print(f"terminated {instance_id} ({label}): {why}")
    print(f"\nswept {swept}, skipped {skipped} open reservation(s)")
    return 0


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = settings()
    cloud_root = cfg.runs_dir / "cloud"
    # The ledger lives beside the runs it paid for, not inside any single run directory — it must
    # survive every individual run being cleaned up.
    ledger = BudgetLedger(cloud_root / "budget.jsonl")
    provider = load_provider(args.provider)

    if args.terminate_orphans:
        return await terminate_orphans(provider, ledger, include_reserved=args.include_reserved)

    if args.dry_run:
        offers = list(
            await provider.offers(min_vram_gb=args.min_vram, max_price_per_hour=args.max_price)
        )
        try:
            offer = pick_offer(offers, min_vram_gb=args.min_vram, max_price_per_hour=args.max_price)
        except NoOfferError as exc:
            print(f"no qualifying offer: {exc}")
            return 1
        shown = sorted(
            (
                o
                for o in offers
                if o.vram_gb >= args.min_vram and o.price_per_hour <= args.max_price
            ),
            key=lambda o: o.price_per_hour,
        )[:5]
        print(f"qualifying offers (up to 5 shown, cheapest first, {len(offers)} listed):")
        for o in shown:
            marker = "  <- would take" if o.offer_id == offer.offer_id else ""
            print(
                f"  {o.gpu_name:<24} {o.vram_gb:>3} GB  ${o.price_per_hour:.3f}/hr"
                f"  [{o.offer_id}]{marker}"
            )
        reserve_at_cap = worst_case_usd(args.max_price, args.max_hours, args.volume_gb)
        expect_at_offer = worst_case_usd(offer.price_per_hour, args.max_hours, args.volume_gb)
        spend = ledger.month_spend()
        print(f"offer     : {offer.gpu_name} {offer.vram_gb} GB at ${offer.price_per_hour:.3f}/hr")
        print(f"            ({offer.provider} offer {offer.offer_id})")
        print(f"reserve   : ${reserve_at_cap:.2f} worst case ({args.max_hours:g}h at the "
              f"${args.max_price:.2f} cap + {args.volume_gb} GB volume)")  # fmt: skip
        print(f"expected  : ${expect_at_offer:.2f} at the offer's actual rate")
        print(f"budget    : ${spend.settled_usd:.2f} settled + ${spend.reserved_usd:.2f} reserved "
              f"of ${spend.ceiling_usd:.2f} for {spend.month} "
              f"(${spend.remaining_usd:.2f} remaining)")  # fmt: skip
        print("\ndry run: refusing to provision.")
        return 0

    worker_providers = parse_worker_providers(args.worker_providers)
    worker_keys = worker_env(cfg, worker_providers)
    image = args.image.strip() or os.environ.get("CORYPHAEUS_CLOUD_IMAGE", "").strip()
    if not image:
        print(
            "no image: pass --image or set CORYPHAEUS_CLOUD_IMAGE. Build coryphaeus/cloud/"
            "Dockerfile and push it to a registry you control; any image with bash, git, and "
            "curl also works (bootstrap.sh installs uv and clones the repo).",
            file=sys.stderr,
        )
        return 2

    push: tuple[tuple[Path, str], ...] = ()
    if args.questions_file:
        questions_path = Path(args.questions_file)
        if not questions_path.is_file():
            print(f"--questions-file {questions_path} does not exist", file=sys.stderr)
            return 2
        push = ((questions_path, REMOTE_QUESTIONS),)

    # The box must trust our key BEFORE the first SSH — authorized_keys cannot be delivered over
    # the channel it gates. The public half rides provision(env=...) as PUBLIC_KEY: RunPod's stock
    # templates consume exactly that name at boot, and the Vast backend translates it into its
    # onstart mechanism. Account-level provider key settings are deliberately not used — per-run
    # env keeps the whole pipeline free of account mutations.
    key_path = Path(
        os.environ.get("CORYPHAEUS_SSH_KEY", "").strip() or "~/.ssh/id_ed25519"
    ).expanduser()
    pub_path = key_path.with_suffix(key_path.suffix + ".pub")
    if not pub_path.is_file():
        print(
            f"no SSH public key at {pub_path} (from CORYPHAEUS_SSH_KEY={key_path}). Generate a "
            f'dedicated automation key: ssh-keygen -t ed25519 -N "" -f {key_path}',
            file=sys.stderr,
        )
        return 2
    ssh_public_key = pub_path.read_text(encoding="utf-8").strip()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    label = f"coryphaeus-{stamp}-{args.chain}"
    run_dir = cloud_root / label
    repo_url = resolve_repo_url(args.repo_url)
    repo_ref = args.repo_ref.strip() or os.environ.get("REPO_REF", "").strip() or "develop"
    chain = build_chain(args, train_label=f"cloud-{stamp}")

    spec = LaunchSpec(
        image=image,
        command=build_remote_command(repo_url=repo_url, repo_ref=repo_ref, chain=chain),
        label=label,
        # The secrets ride the provider's env injection — never the command line, never a file
        # in this repo, never the events log. Exactly the selected providers' keys, no more.
        env={
            **worker_keys,
            "CORYPHAEUS_RUNS_DIR": REMOTE_RUNS_DIR,
            "HF_HOME": "/workspace/hf",
            "PUBLIC_KEY": ssh_public_key,
        },
        min_vram_gb=args.min_vram,
        max_price_per_hour=args.max_price,
        max_hours=args.max_hours,
        volume_gb=args.volume_gb,
        provision_timeout_s=args.provision_timeout,
        remote_artifact_dir=REMOTE_RUNS_DIR,
        artifact_dir=run_dir / "artifacts",
        push=push,
    )

    notify = make_notify()

    # Rollouts on the box spend per-token money the box cannot meter (the ledgers are local
    # files). So the launcher brackets them: worst case reserved here, actuals settled below
    # from the pulled telemetry's own cost_usd receipts. A crash between the two over-counts —
    # the ledger's one allowed failure mode.
    token_reserve = token_reserve_usd(
        load_manifest(), chain=args.chain, steps=args.steps, k=args.k, providers=worker_providers
    )
    tokens = token_ledger(cfg.runs_dir)
    if token_reserve:
        try:
            tokens.reserve(label, token_reserve, note=f"cloud {args.chain} rollouts, worst case")
            print(f"token ledger: reserved ${token_reserve:.2f} worst-case as {label}")
        except BudgetExceeded as exc:
            print(f"\nlaunch refused (token budget): {exc}", file=sys.stderr)
            if notify is not None:
                with contextlib.suppress(Exception):
                    await notify(f"[coryphaeus] {label}: token-budget-refused — {exc}")
            return 1

    try:
        result = await launch(provider, spec, ledger, run_dir=run_dir, notify=notify)
    except (BudgetExceeded, LauncherError) as exc:
        # The launcher already terminated, settled, and wrote the forensic trail; the CLI's job
        # is a readable verdict and a nonzero exit.
        print(f"\nlaunch failed: {exc}", file=sys.stderr)
        print(f"events: {run_dir / 'launcher-events.jsonl'}", file=sys.stderr)
        if isinstance(exc, BudgetExceeded) and notify is not None:
            # A refusal happens before the launcher owns notifications — but an unattended
            # launch that silently declined is exactly what notifications exist to surface.
            with contextlib.suppress(Exception):
                await notify(f"[coryphaeus] {label}: budget-refused — {exc}")
        return 1
    finally:
        if token_reserve:
            pulled = (
                sorted(spec.artifact_dir.rglob("*.jsonl")) if spec.artifact_dir.is_dir() else []
            )
            spent = sum_cost_usd(pulled)
            tokens.settle(label, spent, note=f"summed cost_usd over {len(pulled)} pulled file(s)")
            print(f"token ledger: settled {label} at ${spent:.4f}")

    print(f"\ndone: exit {result.exit_code}, settled ${result.actual_usd:.2f}")
    print(f"artifacts: {spec.artifact_dir}")
    print(f"events   : {result.events_path}")
    print(json.dumps({"instance": result.instance_id, "offer": result.offer.offer_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
