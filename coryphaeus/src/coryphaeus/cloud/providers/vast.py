"""Vast.ai backend, on the console REST API (``https://console.vast.ai/api/v0``, verified
July 2026 — v0 is the current version; it is the same API the official CLI drives).

Vast is a spot marketplace of individually-owned machines, which buys the lowest prices on the
market and costs predictability: hosts differ wildly in network quality, and OUR TRAINING REWARD
IS NETWORK-BOUND — the GPU idles 50-70% of every step waiting on worker API calls. A cheap card
behind a residential DSL line is a slower step, not a cheaper one. ``offers()`` therefore
enforces two floors that are project policy, not provider defaults: measured ``inet_down`` >=
200 Mb/s and ``reliability2`` >= 0.98. It also surfaces the host's self-set egress price in
:attr:`GpuOffer.egress_per_gb`, because a marketplace host can make a chatty workload expensive
in a way the $/hr number never shows.

Shapes verified against ``docs.vast.ai/api-reference`` (July 2026):

* ``POST /bundles`` — search offers. Filters are operator objects (``{"gpu_ram": {"gte":
  24576}}``; operators ``eq|neq|gt|lt|gte|lte|in|notin``), plus ``type`` (``ondemand|bid|
  reserved``), ``order`` (``[["dph_total", "asc"]]``) and ``limit``. Response ``{"offers":
  [...]}`` where — units matter — ``gpu_ram`` is **MB** (the CLI displays GB; the REST API does
  not), ``dph_total``/``dph_base``/``min_bid`` are $/hr, ``inet_down``/``inet_up`` are Mb/s,
  ``inet_down_cost``/``inet_up_cost`` are $/GB, ``storage_cost`` is $/GB/month, and
  ``reliability2`` is a [0, 1] score.
* ``PUT /asks/{offer_id}`` — create an instance from an offer. Body: ``image``, ``env`` (flat
  dict), ``disk`` (GB), ``runtype``, ``label``, ``onstart``, ``target_state``; bid rentals add
  ``price`` ($/hr). Response ``{"success": true, "new_contract": <instance id>}`` — the id
  arrives under ``new_contract``, not ``id``.
* ``GET /instances/{id}`` — show. The single instance arrives wrapped as ``{"instances":
  {...}}`` — plural key, singular object; a real quirk of the API, not a typo here. State
  rides ``actual_status``; ``ssh_host``/``ssh_port`` appear once running; ``start_date`` is a
  unix timestamp (float seconds).
* ``DELETE /instances/{id}`` — destroy. 200 ``{"success": true}``, or 404 (``{"success":
  false, "error": "not_found"}``) once it no longer exists.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence

import httpx

from ...config import settings
from ...workers.base import TRANSIENT_STATUSES
from .base import GpuOffer, Instance, InstanceState, ProviderError, body_json


class MissingApiKey(RuntimeError):
    """No ``VAST_API_KEY`` available. Raised at construction, not mid-run."""


#: Measured-download floor, Mb/s. The reward loop streams worker traffic for the whole step;
#: below roughly this the network, not the GPU, sets the step time — which silently inverts the
#: entire cheapest-card-that-fits logic this package is built on.
MIN_INET_DOWN_MBPS = 200.0
#: Vast's own machine uptime score, [0, 1]. 0.98 still admits most of the marketplace but cuts
#: the hosts that flap mid-run; an interrupted run pays for re-provisioning AND re-warming the
#: trainer, so "slightly cheaper but occasionally vanishes" is a bad trade here.
MIN_RELIABILITY = 0.98

#: ``actual_status`` → normalized state. The docs' own guidance (July 2026): once
#: ``actual_status`` is "exited", "unknown" or "offline" the instance "will never reach
#: running" — so all three map to STOPPED, because the launcher's question is "keep waiting or
#: clean up?" and for all three the answer is clean up. Null means the container has not
#: reported yet, i.e. still coming up.
_VAST_STATES = {
    None: InstanceState.PENDING,
    "": InstanceState.PENDING,
    "loading": InstanceState.PENDING,
    "running": InstanceState.RUNNING,
    "exited": InstanceState.STOPPED,
    "offline": InstanceState.STOPPED,
    "unknown": InstanceState.STOPPED,
}


def _reason(response: httpx.Response) -> str:
    """Vast errors arrive as ``{"success": false, "error": <code>, "msg": <text>}`` (verified
    July 2026). Surface code and message — a refused rental must name its reason in the log."""
    try:
        body = response.json() or {}
    except ValueError:
        return response.text[:200]
    code = str(body.get("error") or "")
    msg = str(body.get("msg") or "")
    joined = f"{code}: {msg}" if code and msg else (code or msg)
    return joined[:300] or response.text[:200]


def _ssh_onstart(public_key: str) -> str:
    """Boot script granting the launcher SSH access on any Vast image; "" when there is no key.

    RunPod's stock templates consume a ``PUBLIC_KEY`` env var at boot; Vast images share no
    such convention, so the same contract is built here by hand: write the key into root's
    ``authorized_keys`` and make sure an sshd is up. Images vary (some have ``service``, some
    only an sshd binary, some already run one), so every arm tolerates failure — this script
    must never be the reason a box that would have worked did not come up. The key deliberately
    ALSO stays in ``env``: harmless duplication beats a missing key on an image that does
    happen to know the RunPod convention.

    ``onstart`` is the verified REST field name (create-instance reference, July 2026:
    "Commands to run when instance starts", example ``env | grep _ >> /etc/environment; ...``).
    The CLI exposes the same thing as its ``--onstart-cmd`` flag; that spelling is assumed to
    be flag-only and is not sent.
    """
    if not public_key:
        # No key → no script, rather than a benign empty one: an empty onstart still lands in
        # the instance record and reads as intent. Omission honestly says "image defaults rule".
        return ""
    # POSIX single-quote escaping. Real OpenSSH public keys never contain quotes, but this is
    # boot-time shell on a rented box — the one place to be paranoid about interpolation.
    quoted = public_key.replace("'", "'\"'\"'")
    return "\n".join(
        [
            # Container env does NOT reach SSH sessions: sshd spawns fresh environments, so the
            # provision-injected secrets (FEATHERLESS_API_KEY et al.) are invisible to the
            # payload shell unless persisted where PAM reads them. This is Vast's own canonical
            # onstart line — quoted as their docs' example above — and take 5 paid $0.012 to
            # learn why it exists: the chain bootstrapped perfectly and then exited 2 on a
            # "missing" key that was sitting in the container environment all along.
            "env | grep _ >> /etc/environment",
            "mkdir -p /root/.ssh",
            "chmod 700 /root/.ssh",
            # Append, never overwrite: an image (or a human mid-debug) may have installed keys
            # of its own, and clobbering them is an unrecoverable lockout on a rented box.
            f"printf '%s\\n' '{quoted}' >> /root/.ssh/authorized_keys",
            "chmod 600 /root/.ssh/authorized_keys",
            "service ssh start 2>/dev/null || service sshd start 2>/dev/null || true",
            "pgrep -x sshd >/dev/null || /usr/sbin/sshd 2>/dev/null || sshd 2>/dev/null || true",
        ]
    )


class VastProvider:
    """GPU rental on the Vast.ai marketplace, with network floors suited to this project."""

    name = "vast"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
        attempts: int = 4,
        base_delay: float = 2.0,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        """Args:
        client: injected for MockTransport tests, exactly like the worker adapters.
        attempts / base_delay / sleeper: bounded transient-retry knobs; no governor exists
            above this layer, so the loop lives here and ``sleeper`` keeps tests sleep-free.
        now: unix-time source for :meth:`cost_so_far`, injectable so tests are wall-clock-free.
        """
        cfg = settings()
        key = api_key or cfg.vast_api_key
        if not key:
            raise MissingApiKey(
                "VAST_API_KEY is not set. Put it in .env (see .env.example); the offline test "
                "suite and every other provider work without it."
            )
        self._api_key = key
        self.base_url = (base_url or cfg.vast_base_url).rstrip("/")
        self._client = client
        self._timeout = timeout
        self._attempts = max(1, attempts)
        self._base_delay = base_delay
        self._sleep = sleeper or asyncio.sleep
        self._now = now or time.time

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _send(self, method: str, url: str, *, json_body: dict | None) -> httpx.Response:
        # follow_redirects: the API 301s bare paths to their trailing-slash canonicals with an
        # HTML body a JSON client cannot use. Canonical paths avoid the round-trip; following is
        # the belt for any path we got wrong. GETs only in practice — provision PUTs land direct.
        if self._client is not None:
            return await self._client.request(
                method, url, json=json_body, headers=self._headers, follow_redirects=True
            )
        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            return await client.request(method, url, json=json_body, headers=self._headers)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        idempotent: bool = True,
    ) -> httpx.Response:
        """One API call with bounded transient retries — same split as the worker adapters.

        429/502/503/504 mean "ask again"; other 4xx mean the request is wrong. The cloud
        wrinkle: a call that died IN FLIGHT is retried only when idempotent. ``PUT /asks/{id}``
        is *not* idempotent despite the verb — each accepted call opens a rental contract — so
        a timed-out provision raises :class:`ProviderError` telling the caller to reconcile
        (list instances) rather than risk renting the offer twice. A transient status is safe
        everywhere: the server answered, so no contract was opened.
        """
        return await self._request_absolute(
            method, f"{self.base_url}{path}", json_body=json_body, idempotent=idempotent
        )

    async def _request_absolute(
        self,
        method: str,
        url: str,
        *,
        json_body: dict | None = None,
        idempotent: bool = True,
    ) -> httpx.Response:
        """Same retry loop, taking a full URL — the seam that lets ``list_instances`` reach the
        v1 API while everything else stays on the configured v0 base."""
        failure = ""
        for attempt in range(self._attempts):
            if attempt:
                await self._sleep(self._base_delay * 2 ** (attempt - 1))
            try:
                response = await self._send(method, url, json_body=json_body)
            except (httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                if not idempotent:
                    raise ProviderError(
                        f"{self.name}: {method} {url} died in flight "
                        f"({type(exc).__name__}) — a contract may exist on the far side; "
                        "list instances and reconcile before retrying"
                    ) from exc
                failure = f"transport {type(exc).__name__}"
                continue
            except httpx.HTTPError as exc:
                # A refused connection is not a timeout — do not launder every transport error
                # into a retry (same reasoning as the featherless adapter).
                raise ProviderError(
                    f"{self.name}: {method} {url} failed — {type(exc).__name__}: {exc}"
                ) from exc
            if response.status_code in TRANSIENT_STATUSES:
                failure = f"http {response.status_code}: {_reason(response)}"
                continue
            return response
        raise ProviderError(
            f"{self.name}: {method} {url} still failing after {self._attempts} attempts — {failure}"
        )

    async def offers(self, *, min_vram_gb: int, max_price_per_hour: float) -> Sequence[GpuOffer]:
        """On-demand offers passing the VRAM/price floors AND the network floors, cheapest first.

        Every floor is enforced twice: once server-side in the query (cheap, less traffic) and
        once client-side on the returned rows. The double-check is deliberate — Vast has two
        reliability field spellings (``reliability``/``reliability2``) and the CLI has aliased
        between them, so a silently ignored server-side filter would rent exactly the flaky
        hosts this module exists to refuse. The client-side check is the guarantee; the query
        is an optimization.
        """
        query: dict = {
            "type": "ondemand",
            "rentable": {"eq": True},
            # Verified machines have passed Vast's own hardware/network checks; unverified ones
            # are cheaper but exactly where the mismeasured-bandwidth stories come from.
            "verified": {"eq": True},
            # The policy + trainer fit on one card; multi-GPU rigs price worse per useful hour.
            "num_gpus": {"eq": 1},
            # REST API units are MB (the CLI displays GB). Hosts report *usable* VRAM, a hair
            # under nameplate — a 24 GB 4090 shows ~24564 MB — so an exact 24*1024 floor would
            # exclude every 4090 from a 24 GB ask. Half a GB of tolerance forgives reporting
            # slop without admitting the next card class down.
            "gpu_ram": {"gte": int(min_vram_gb) * 1024 - 512},
            "dph_total": {"lte": max_price_per_hour},
            "inet_down": {"gte": MIN_INET_DOWN_MBPS},
            "reliability2": {"gte": MIN_RELIABILITY},
            "order": [["dph_total", "asc"]],
            "limit": 64,
        }
        response = await self._request("POST", "/bundles", json_body=query)
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name}: offer search failed — "
                f"http {response.status_code}: {_reason(response)}"
            )
        data = body_json(response)
        if data is None:
            raise ProviderError(
                f"{self.name}: offer search returned non-JSON "
                f"(http {response.status_code}): {response.text[:120]!r}"
            )
        rentable: list[GpuOffer] = []
        for raw in (data or {}).get("offers") or []:
            # round(), not floor: 24564 MB is a 24 GB card, and flooring it to 23 would fail
            # the very min_vram_gb comparison the caller asked for.
            vram_gb = round(float(raw.get("gpu_ram") or 0) / 1024)
            price = float(raw.get("dph_total") or 0.0)
            inet_down = float(raw.get("inet_down") or 0.0)
            reliability = float(raw.get("reliability2") or raw.get("reliability") or 0.0)
            if vram_gb < min_vram_gb or price > max_price_per_hour:
                continue
            if inet_down < MIN_INET_DOWN_MBPS or reliability < MIN_RELIABILITY:
                continue
            min_bid = raw.get("min_bid")
            rentable.append(
                GpuOffer(
                    provider=self.name,
                    offer_id=str(raw.get("id") or ""),
                    gpu_name=str(raw.get("gpu_name") or ""),
                    vram_gb=vram_gb,
                    price_per_hour=price,
                    interruptible_price_per_hour=(float(min_bid) if min_bid is not None else None),
                    # Egress is traffic LEAVING the box, i.e. the host's upload price. The
                    # reward loop's requests ride it constantly; hosts set it themselves and
                    # some set it silly, so it must be visible before renting.
                    egress_per_gb=float(raw.get("inet_up_cost") or 0.0),
                    raw=raw,
                )
            )
        return sorted(rentable, key=lambda offer: offer.price_per_hour)

    async def provision(
        self,
        offer: GpuOffer,
        *,
        image: str,
        env: dict[str, str],
        volume_gb: int,
        label: str,
    ) -> Instance:
        payload: dict = {
            "image": image,
            "env": dict(env),
            # "ssh" lets Vast pick direct vs proxied SSH per host capability; describe() reads
            # whichever endpoint materialized out of ssh_host/ssh_port.
            "runtype": "ssh",
            "label": label,
            "target_state": "running",
        }
        if volume_gb:
            payload["disk"] = volume_gb  # omitted → provider default (8 GB, verified July 2026)
        onstart = _ssh_onstart(env.get("PUBLIC_KEY", ""))
        if onstart:
            payload["onstart"] = onstart
        response = await self._request(
            "PUT", f"/asks/{offer.offer_id}", json_body=payload, idempotent=False
        )
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name}: provision of offer {offer.offer_id} refused — "
                f"http {response.status_code}: {_reason(response)}"
            )
        body = body_json(response)
        if body is None:
            raise ProviderError(
                f"{self.name}: provision of offer {offer.offer_id} returned http "
                f"{response.status_code} with a non-JSON body — a contract MAY exist; "
                "reconcile with list_instances before retrying"
            )
        if not body.get("success") or body.get("new_contract") is None:
            # An offer can be snatched between search and rent — the marketplace's race, not
            # ours. Surface the API's own message so the caller can move to the next offer.
            raise ProviderError(f"{self.name}: provision returned no contract — {body}")
        return Instance(
            provider=self.name,
            instance_id=str(body["new_contract"]),
            state=InstanceState.PENDING,
            gpu_name=offer.gpu_name,
            price_per_hour=offer.price_per_hour,
            raw=body,
        )

    async def describe(self, instance_id: str) -> Instance:
        """Current state; never raises — a poll failure is UNKNOWN and the caller decides.

        404 maps to TERMINATED: a destroyed contract has no record, so "nothing to show" is
        what successful destruction looks like from outside, and the watch loop needs that to
        read as "stopped billing".
        """
        try:
            # Trailing slash is CANONICAL: the bare path answers 301 with an HTML body, which a
            # non-following client reads as ten minutes of UNKNOWN (lived it, 2026-07-31, $0.02).
            response = await self._request("GET", f"/instances/{instance_id}/")
        except ProviderError as exc:
            return self._opaque(instance_id, InstanceState.UNKNOWN, str(exc))
        if response.status_code == 404:
            return self._opaque(instance_id, InstanceState.TERMINATED, _reason(response))
        if response.status_code >= 400:
            return self._opaque(instance_id, InstanceState.UNKNOWN, _reason(response))
        # The single instance arrives under a PLURAL key — {"instances": {...}} — verified
        # July 2026. Reading it as a list here would be the bug.
        data = body_json(response)
        if data is None:
            # Observed live 2026-07-31: a just-created contract briefly answers 2xx with an
            # EMPTY body. That is "not ready to say", i.e. UNKNOWN — the poll keeps polling.
            return self._opaque(
                instance_id, InstanceState.UNKNOWN, "non-JSON body (fresh contract?)"
            )
        return self._to_instance((data or {}).get("instances") or {}, instance_id=instance_id)

    async def list_instances(self) -> Sequence[Instance]:
        """Every rental contract on the account, any state — the orphan sweep's raw material.

        Verified (docs, July 2026): the list endpoint returns ``{"instances": [...]}`` — a real
        array this time, where the singular endpoint wraps one *object* under the same plural
        key — and each element carries ``label`` (string or null) plus the same fields as
        show-instance. Assumed: the docs spell the list's path ``/api/v1/instances`` while every
        other endpoint this backend drives is documented under ``/api/v0``; this call follows
        the backend's configured base (v0), the prefix the official CLI has always driven for
        listing, rather than mixing API versions inside one client. If Vast ever retires the v0
        spelling, this is the first call that will notice.

        A failed list RAISES rather than returning ``[]``: to an orphan sweep, "could not look"
        and "nothing there" are opposite answers, and the wrong one ends the search while a
        rental keeps billing.
        """
        # The v0 list is DEAD — it answers 410 `deprecated_endpoint` pointing at v1 (observed
        # live 2026-07-31, resolving this module's own earlier v0-vs-v1 uncertainty). The v1
        # shape is verified: {"success": true, "instances": [...], "next_token": ...} — a real
        # array under the same plural key show-instance overloads for a single object.
        v1_base = self.base_url.replace("/api/v0", "/api/v1")
        response = await self._request_absolute("GET", f"{v1_base}/instances/")
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name}: listing instances failed — "
                f"http {response.status_code}: {_reason(response)}"
            )
        data = body_json(response)
        if data is None:
            raise ProviderError(
                f"{self.name}: instance list returned non-JSON "
                f"(http {response.status_code}) — could-not-look must not read as nothing-there"
            )
        return [self._to_instance(row) for row in (data or {}).get("instances") or []]

    def _to_instance(self, body: dict, *, instance_id: str | None = None) -> Instance:
        status = body.get("actual_status")
        state = _VAST_STATES.get(
            status if status is None else str(status).lower(), InstanceState.UNKNOWN
        )
        ssh_port = body.get("ssh_port")
        return Instance(
            provider=self.name,
            instance_id=instance_id if instance_id is not None else str(body.get("id") or ""),
            state=state,
            gpu_name=str(body.get("gpu_name") or ""),
            price_per_hour=float(body.get("dph_total") or 0.0),
            ssh_host=body.get("ssh_host"),
            ssh_port=int(ssh_port) if ssh_port else None,
            # Vast's field is already called "label" but is nullable; the orphan sweep joins on
            # raw["label"] across providers, so the key is guaranteed non-null here.
            raw={**body, "label": str(body.get("label") or "")},
        )

    async def terminate(self, instance_id: str) -> None:
        """Destroy the contract. Idempotent: 404 (already destroyed) is success, not an error.

        The launcher calls this from every exit path including crash handlers; the second call
        raising would mask the original crash while a rental keeps billing elsewhere.
        """
        response = await self._request("DELETE", f"/instances/{instance_id}")
        if response.status_code == 404:
            return
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name}: destroy {instance_id} refused — "
                f"http {response.status_code}: {_reason(response)}"
            )
        try:
            body = response.json() or {}
        except ValueError:
            return
        if body.get("success") is False and str(body.get("error") or "") != "not_found":
            raise ProviderError(f"{self.name}: destroy {instance_id} refused — {_reason(response)}")

    async def cost_so_far(self, instance_id: str) -> float | None:
        """(now - start_date) x the instance's own rate; None when that is not knowable.

        Vast exposes no per-instance accrued-dollars field (billing lives on invoice
        endpoints), but ``start_date`` is the provider's billing clock and survives launcher
        restarts — strictly better than the ledger's own wall-clock fallback, which starts
        whenever the launcher happened to notice. Excludes storage and egress charges, which
        only invoices show; the ledger treats this as a floor, not gospel.
        """
        try:
            response = await self._request("GET", f"/instances/{instance_id}/")
        except ProviderError:
            return None
        if response.status_code >= 400:
            return None
        data = body_json(response)
        if data is None:
            return None
        body = (data or {}).get("instances") or {}
        start = body.get("start_date")
        rate = body.get("dph_total")
        if start is None or rate is None:
            return None
        return max(0.0, (self._now() - float(start)) / 3600.0) * float(rate)

    def _opaque(self, instance_id: str, state: InstanceState, error: str) -> Instance:
        return Instance(
            provider=self.name,
            instance_id=instance_id,
            state=state,
            gpu_name="",
            price_per_hour=0.0,
            raw={"error": error},
        )
