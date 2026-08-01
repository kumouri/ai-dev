"""RunPod backend, on REST API v2 (``https://api.runpod.io/v2``, verified July 2026).

RunPod has had three API generations: the original GraphQL API (``api.runpod.io/graphql``), REST
v1 (``rest.runpod.io/v1``), and REST v2. As of July 2026 the v1 docs carry an explicit
retirement notice ("deprecated and will be retired in the near future") and point new
integrations at v2 — so this backend speaks v2 only. Every path, field name and enum below was
read from the live OpenAPI spec (``https://api.runpod.io/v2/openapi.json``) and the v2 reference
at ``docs.runpod.io/api-reference-v2`` (verified July 2026):

* ``GET /catalog/gpus`` → ``{"gpus": [{"id", "name", "memory", "price": {"secure",
  "community"}, "availability"}]}``. ``memory`` is VRAM in GB, prices are $/hr, and
  ``availability`` (``NONE|LOW|MEDIUM|HIGH``) appears only when ``include=AVAILABILITY``.
* ``POST /pods`` → 201 + the Pod object. Request: ``name``, ``image``, ``gpu: {id, count}``,
  ``env`` (flat dict), ``cloud`` (``SECURE|COMMUNITY``), ``ports`` (``["22/tcp"]``),
  ``mounts: {"persistent": {"size", "path"}}``.
* ``GET /pods/{id}`` → Pod: ``status`` in ``PROVISIONING|STARTING|RUNNING|EXITED|ERROR|
  TERMINATED``, ``cost`` ($/hr), and ``runtime`` (null until RUNNING) carrying ``uptime``
  seconds and ``ports: [{"private", "public", "type", "ip"}]``.
* ``DELETE /pods/{id}`` → 204, or 404 once the pod no longer exists.
* Errors are problem-details-shaped: ``{"title", "status", "detail", "errors": [...]}``.

Community cloud is the default on purpose: the training reward is network-bound and the GPU
idles 50-70% of every step, so the right rental is the cheapest card that fits the policy — and
community pricing is the cheap half of the catalog (a 4090 listed ~$0.34/hr community vs ~$0.69
secure when checked, July 2026). The tier is a pay-per-use reliability knob, not a constant:
``CORYPHAEUS_RUNPOD_CLOUD=SECURE`` flips the catalog query, the quoted price, and the provision
payload together (see ``_cloud_tier``), so retry wrappers can escalate to datacenter hosts
mid-loop after community-lottery failures.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable, Sequence

import httpx

from ...config import settings
from ...workers.base import TRANSIENT_STATUSES
from .base import GpuOffer, Instance, InstanceState, ProviderError, body_json, min_cuda


def _cloud_tier() -> str:
    """``COMMUNITY`` (default) or ``SECURE`` — reliability as a pay-per-use knob.

    Community is the cheap half of the marketplace and carries marketplace weather (old
    drivers, sold-out cards); Secure is RunPod's datacenter tier at roughly 2× the $/hr with
    overwhelmingly modern hosts. Env-driven (``CORYPHAEUS_RUNPOD_CLOUD``) rather than a
    constructor arg so retry wrappers can ESCALATE mid-loop: try the cheap lottery first, then
    pay double for a datacenter host instead of failing the night. Read per call, not cached.
    """
    raw = os.environ.get("CORYPHAEUS_RUNPOD_CLOUD", "").strip().upper()
    return raw if raw in ("COMMUNITY", "SECURE") else "COMMUNITY"


#: The CUDA versions RunPod's create schema accepts (docs, August 2026). The shared driver
#: floor selects the acceptable suffix of this list.
_CUDA_VERSIONS = (
    "11.8", "12.0", "12.1", "12.2", "12.3", "12.4",
    "12.5", "12.6", "12.7", "12.8", "12.9", "13.0",
)  # fmt: skip


def _allowed_cuda_versions() -> list[str]:
    floor = min_cuda()
    allowed = [v for v in _CUDA_VERSIONS if float(v) >= floor]
    # A floor above the schema's ceiling would otherwise send [] — which the docs define as
    # "anything goes", the exact opposite of what a high floor means. Ask for the newest.
    return allowed or [_CUDA_VERSIONS[-1]]


class MissingApiKey(RuntimeError):
    """No ``RUNPOD_API_KEY`` available. Raised at construction, not mid-run."""


#: Pod ``status`` → normalized state. EXITED and ERROR both mean "the container died and will
#: not come back on its own", which for the launcher is the same decision: stop waiting, clean
#: up — so both map to STOPPED rather than pretending one is more recoverable than the other.
_POD_STATES = {
    "PROVISIONING": InstanceState.PENDING,
    "STARTING": InstanceState.PENDING,
    "RUNNING": InstanceState.RUNNING,
    "EXITED": InstanceState.STOPPED,
    "ERROR": InstanceState.STOPPED,
    "TERMINATED": InstanceState.TERMINATED,
}


def _reason(response: httpx.Response) -> str:
    """Pull RunPod's own title/detail out of a problem-details body, for a legible log line.

    A refused provision must be explainable to a human reading a log at 7am; "http 422" alone
    tells them nothing, the API's ``detail`` usually tells them everything.
    """
    try:
        body = response.json() or {}
    except ValueError:
        return response.text[:200]
    title = str(body.get("title") or "")
    detail = str(body.get("detail") or "")
    extra = "; ".join(str(e) for e in (body.get("errors") or []))
    joined = ": ".join(part for part in (title, detail) if part)
    if extra:
        joined = f"{joined} ({extra})" if joined else extra
    return joined[:300] or response.text[:200]


class RunPodProvider:
    """GPU rental on RunPod's community (and secure) cloud, via REST v2."""

    name = "runpod"

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
    ) -> None:
        """Args:
        client: injected for MockTransport tests, exactly like the worker adapters. Left None,
            each call opens a short-lived client — fine at control-plane call rates.
        attempts / base_delay / sleeper: the transient-retry knobs. Control-plane calls get the
            same transient/permanent split the worker adapters use, but there is no governor
            above this layer, so the bounded retry loop lives here; ``sleeper`` exists so tests
            never actually sleep.
        """
        cfg = settings()
        key = api_key or cfg.runpod_api_key
        if not key:
            raise MissingApiKey(
                "RUNPOD_API_KEY is not set. Put it in .env (see .env.example); the offline test "
                "suite and every other provider work without it."
            )
        self._api_key = key
        self.base_url = (base_url or cfg.runpod_base_url).rstrip("/")
        self._client = client
        self._timeout = timeout
        self._attempts = max(1, attempts)
        self._base_delay = base_delay
        self._sleep = sleeper or asyncio.sleep

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _send(
        self, method: str, url: str, *, json_body: dict | None, params: dict | None
    ) -> httpx.Response:
        if self._client is not None:
            return await self._client.request(
                method, url, json=json_body, params=params, headers=self._headers
            )
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.request(
                method, url, json=json_body, params=params, headers=self._headers
            )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        params: dict | None = None,
        idempotent: bool = True,
    ) -> httpx.Response:
        """One API call with bounded transient retries.

        Same classification the worker adapters use — 429/502/503/504 mean "ask again",
        anything else 4xx means the request itself is wrong — with one cloud-specific wrinkle:
        a call that died IN FLIGHT (timeout, dropped connection) is retried only when the verb
        is idempotent. A provision that timed out may have created a pod on the far side, and
        blindly re-sending it would rent a second GPU; that ambiguity is raised as
        :class:`ProviderError` telling the caller to reconcile (list pods) before retrying.
        A transient *status* is different: a 429/503 response proves the server refused before
        creating anything, so those retry even on provision.
        """
        url = f"{self.base_url}{path}"
        failure = ""
        for attempt in range(self._attempts):
            if attempt:
                await self._sleep(self._base_delay * 2 ** (attempt - 1))
            try:
                response = await self._send(method, url, json_body=json_body, params=params)
            except (httpx.TimeoutException, httpx.RemoteProtocolError) as exc:
                if not idempotent:
                    raise ProviderError(
                        f"{self.name}: {method} {path} died in flight "
                        f"({type(exc).__name__}) — a pod may exist on the far side; "
                        "list pods and reconcile before retrying"
                    ) from exc
                failure = f"transport {type(exc).__name__}"
                continue
            except httpx.HTTPError as exc:
                # A refused connection is not a timeout — do not launder every transport error
                # into a retry (same reasoning as the featherless adapter).
                raise ProviderError(
                    f"{self.name}: {method} {path} failed — {type(exc).__name__}: {exc}"
                ) from exc
            if response.status_code in TRANSIENT_STATUSES:
                failure = f"http {response.status_code}: {_reason(response)}"
                continue
            return response
        raise ProviderError(
            f"{self.name}: {method} {path} still failing after {self._attempts} attempts "
            f"— {failure}"
        )

    async def offers(self, *, min_vram_gb: int, max_price_per_hour: float) -> Sequence[GpuOffer]:
        """GPU types on the selected cloud tier meeting the floors, cheapest first.

        Cards whose ``availability`` is ``NONE`` are dropped — the contract promises *currently
        rentable* offers, and a sold-out card would just make provision() fail slower. Cards
        with no price on the selected tier are dropped for the same reason. The tier drives the
        query, the price field read, AND the provision payload from one source (`_cloud_tier`) —
        picking on the community price while paying the secure one would corrupt the
        reservation math.
        """
        tier = _cloud_tier()
        response = await self._request(
            "GET",
            "/catalog/gpus",
            params={"include": "AVAILABILITY", "cloud": tier, "product": "POD"},
        )
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name}: listing gpus failed — "
                f"http {response.status_code}: {_reason(response)}"
            )
        data = body_json(response)
        if data is None:
            raise ProviderError(
                f"{self.name}: gpu catalog returned non-JSON "
                f"(http {response.status_code}): {response.text[:120]!r}"
            )
        rentable: list[GpuOffer] = []
        for gpu in (data or {}).get("gpus") or []:
            # Both keys verified against the v2 OpenAPI spec (July 2026): $/hr rides
            # price.community / price.secure. A card with no price on the selected tier
            # (e.g. secure-only hardware seen from COMMUNITY) is not rentable there.
            tier_price = (gpu.get("price") or {}).get(tier.lower())
            if tier_price is None or gpu.get("availability") == "NONE":
                continue
            vram_gb = int(gpu.get("memory") or 0)
            price = float(tier_price)
            if vram_gb < min_vram_gb or price > max_price_per_hour:
                continue
            rentable.append(
                GpuOffer(
                    provider=self.name,
                    offer_id=str(gpu.get("id") or ""),
                    gpu_name=str(gpu.get("name") or gpu.get("id") or ""),
                    vram_gb=vram_gb,
                    price_per_hour=price,
                    raw=gpu,
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
            "name": label,
            "image": image,
            "gpu": {"id": offer.offer_id, "count": 1},
            "env": dict(env),
            "cloud": _cloud_tier(),
            # NO CUDA floor on this dialect, and not for lack of trying: the DOCUMENTED RunPod
            # API (rest.runpod.io PodCreateInput) accepts allowedCudaVersions, but this endpoint
            # — the live-bisected api.runpod.io/v2 dialect — 422s the field by name
            # ("additional properties 'allowedCudaVersions' not allowed", 2026-08-01). Until the
            # backend migrates to the documented dialect, an old-driver draw here is a cheap
            # fast failure (~$0.008, ~3 min: torch refuses, launcher terminates + settles), and
            # retries re-roll the host. _allowed_cuda_versions() stays for the migration.
            # SSH is how the launcher reaches the box; declaring 22/tcp is what makes a public
            # mapping appear in runtime.ports for describe() to read back.
            "ports": ["22/tcp"],
            # Container disk is MANDATORY in practice though not in the schema: the handler
            # treats a body without `disk` as "no pod configuration parameters" and 400s the
            # whole request (bisected live, 2026-07-31 — same payload with disk advances).
            # 20 GB holds the image layers plus uv caches; training artifacts live on the
            # persistent /workspace mount, not here.
            "disk": 20,
        }
        if volume_gb:
            # A persistent volume at RunPod's conventional /workspace: checkpoints survive a
            # container restart, which on community cloud is a when, not an if.
            payload["mounts"] = {"persistent": {"size": volume_gb, "path": "/workspace"}}
        response = await self._request("POST", "/pods", json_body=payload, idempotent=False)
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name}: provision of {offer.offer_id!r} refused — "
                f"http {response.status_code}: {_reason(response)}"
            )
        created = body_json(response)
        if created is None:
            raise ProviderError(
                f"{self.name}: create pod returned http {response.status_code} with a non-JSON "
                "body — a pod MAY exist; reconcile with list_instances before retrying"
            )
        return self._to_instance(created)

    async def describe(self, instance_id: str) -> Instance:
        """Current pod state; never raises — a poll failure is UNKNOWN, the caller decides.

        A 404 maps to TERMINATED rather than UNKNOWN: once a pod is deleted the record is gone,
        so "nothing to describe" is exactly what successful termination looks like from outside
        — and the launcher's watch loop needs that to read as "stopped billing", not "mystery".
        """
        try:
            response = await self._request("GET", f"/pods/{instance_id}")
        except ProviderError as exc:
            return self._opaque(instance_id, InstanceState.UNKNOWN, str(exc))
        if response.status_code == 404:
            return self._opaque(instance_id, InstanceState.TERMINATED, _reason(response))
        if response.status_code >= 400:
            return self._opaque(instance_id, InstanceState.UNKNOWN, _reason(response))
        pod = body_json(response)
        if pod is None:
            # "Spoke, but not in JSON" is a poll answer, not a crash: UNKNOWN, poll again.
            return self._opaque(instance_id, InstanceState.UNKNOWN, "non-JSON body")
        return self._to_instance(pod)

    async def list_instances(self) -> Sequence[Instance]:
        """Every pod on the account, any state — the billing-leak backstop's raw material.

        ``GET /pods`` → ``{"pods": [...]}`` (``ListPodsResponse``, verified July 2026 against
        the v2 OpenAPI spec). Each pod runs through the same mapper as describe, so its ``name``
        lands in ``raw["label"]`` for the sweep to join on. A failed list RAISES rather than
        returning ``[]``: to an orphan sweep, "could not look" and "nothing there" are opposite
        answers, and the wrong one ends the search while a GPU keeps billing.
        """
        response = await self._request("GET", "/pods")
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name}: listing pods failed — "
                f"http {response.status_code}: {_reason(response)}"
            )
        data = body_json(response)
        if data is None:
            raise ProviderError(
                f"{self.name}: pod list returned non-JSON "
                f"(http {response.status_code}) — could-not-look must not read as nothing-there"
            )
        return [self._to_instance(pod) for pod in (data or {}).get("pods") or []]

    async def terminate(self, instance_id: str) -> None:
        """DELETE the pod. Idempotent by construction: 404 means already gone, which is success.

        The launcher calls this from every exit path including crash handlers; an exception on
        the second call would mask the original crash and could abort sibling cleanup while a
        GPU keeps billing.
        """
        response = await self._request("DELETE", f"/pods/{instance_id}")
        if response.status_code == 404 or response.status_code < 400:
            return
        raise ProviderError(
            f"{self.name}: terminate {instance_id} refused — "
            f"http {response.status_code}: {_reason(response)}"
        )

    async def cost_so_far(self, instance_id: str) -> float | None:
        """Provider-reported uptime x the pod's own rate; None when that is not knowable.

        The Pod object carries no accrued-dollars field (billing is a separate aggregate
        endpoint, ``/billing/pods``), but ``runtime.uptime`` is the provider's own clock — more
        trustworthy than the launcher's wall clock, which lies after a restart. ``runtime`` is
        null unless the pod is RUNNING, so a stopped or vanished pod returns None and the
        ledger falls back to its own estimate.
        """
        try:
            response = await self._request("GET", f"/pods/{instance_id}")
        except ProviderError:
            return None
        if response.status_code >= 400:
            return None
        pod = body_json(response)
        if pod is None:
            return None
        uptime = ((pod or {}).get("runtime") or {}).get("uptime")
        if uptime is None:
            return None
        return float(uptime) / 3600.0 * float(pod.get("cost") or 0.0)

    def _opaque(self, instance_id: str, state: InstanceState, error: str) -> Instance:
        return Instance(
            provider=self.name,
            instance_id=instance_id,
            state=state,
            gpu_name="",
            price_per_hour=0.0,
            raw={"error": error},
        )

    def _to_instance(self, pod: dict) -> Instance:
        ssh_host: str | None = None
        ssh_port: int | None = None
        for port in (pod.get("runtime") or {}).get("ports") or []:
            if port.get("private") == 22 and port.get("type") == "tcp":
                ssh_host = port.get("ip")
                ssh_port = int(port["public"]) if port.get("public") else None
                break
        return Instance(
            provider=self.name,
            instance_id=str(pod.get("id") or ""),
            state=_POD_STATES.get(str(pod.get("status") or "").upper(), InstanceState.UNKNOWN),
            gpu_name=str((pod.get("gpu") or {}).get("id") or ""),
            price_per_hour=float(pod.get("cost") or 0.0),
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            # RunPod calls it "name"; the orphan sweep joins on raw["label"] across providers,
            # so the normalized key is guaranteed here (empty string when absent).
            raw={**pod, "label": str(pod.get("name") or "")},
        )
