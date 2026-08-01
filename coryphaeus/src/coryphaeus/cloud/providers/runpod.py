"""RunPod backend — deliberately HYBRID across RunPod's two live REST dialects.

* **Catalog** (``offers()``): ``api.runpod.io/v2`` — ``GET /catalog/gpus`` → ``{"gpus": [{"id",
  "name", "memory", "price": {"secure", "community"}, "availability"}]}``. Verified July 2026
  against the v2 OpenAPI spec; tier-priced, works, kept.
* **Pod lifecycle** (provision/describe/list/terminate): ``rest.runpod.io/v1`` — the DOCUMENTED
  ``PodCreateInput`` dialect (docs.runpod.io api-reference), live-verified 2026-08-01. This is
  the only dialect that accepts ``allowedCudaVersions``, the field that stops the marketplace
  selling us a driver our torch refuses (the v2 pods endpoint 422s it by name). Both dialects
  read the SAME pods — proven live: a v2-created pod appears, identically shaped, in the v1
  list.

The v1 shapes, live-verified 2026-08-01:

* ``POST /pods`` → **201 + the Pod object**. Request: ``name``, ``imageName``,
  ``gpuTypeIds: [id]``, ``gpuCount``, ``cloudType`` (``SECURE|COMMUNITY``), ``env``,
  ``ports: ["22/tcp"]``, ``containerDiskInGb``, ``volumeInGb`` + ``volumeMountPath``,
  ``allowedCudaVersions``. **EVERY field has a server-side default** — an empty body 201s and
  rents a $0.69/hr Secure 4090 (paid ~$0.09 to learn this; the pod was terminated in minutes).
  Never send a partial body expecting a validation error.
* ``GET /pods/{id}`` / ``GET /pods`` (bare ARRAY, not an envelope) → Pod: ``desiredStatus`` in
  ``RUNNING|EXITED|TERMINATED`` (no PROVISIONING/STARTING class — a pod is "RUNNING" by desire
  from the moment of creation), ``publicIp`` (nullable) + ``portMappings`` (``{"22": 12345}``)
  arriving only once the container is actually up, ``costPerHr``, ``machineId`` + ``machine``
  (host forensics), ``lastStartedAt``. **List bodies carry each pod's full ``env`` — secrets —
  so raw list output must never be logged.**
* ``DELETE /pods/{id}`` → 204, or ``{"error": "pod not found", "status": 404}`` once gone.
* v2 errors are problem-details-shaped (``{"title", "status", "detail", "errors"}``); v1 errors
  are ``{"error", "status"}`` or a bare ``[{"error": ...}]`` array — ``_reason`` reads all.

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
from datetime import UTC, datetime

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


def _parse_pod_timestamp(raw: str) -> datetime | None:
    """Parse the v1 dialect's timestamp shape: ``2026-08-01 04:14:06.75 +0000 UTC``.

    A Go-style stamp — fractional seconds of varying width, a numeric offset, AND a trailing
    zone name. Tolerant on purpose (fraction may be absent); anything unparseable is None, and
    the caller falls back to the ledger's own estimate rather than guessing.
    """
    cleaned = raw.strip().removesuffix("UTC").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


class MissingApiKey(RuntimeError):
    """No ``RUNPOD_API_KEY`` available. Raised at construction, not mid-run."""


#: Pod ``desiredStatus`` → normalized state. The v1 dialect has no PROVISIONING/STARTING class
#: — a pod is "RUNNING" by desire from the moment of creation — so the PENDING distinction is
#: synthesized in ``_to_instance`` from the absence of an SSH endpoint. EXITED and ERROR both
#: mean "the container died and will not come back on its own": STOPPED either way. The v2-era
#: keys are kept as harmless aliases (both dialects read the same pods).
_POD_STATES = {
    "PROVISIONING": InstanceState.PENDING,
    "STARTING": InstanceState.PENDING,
    "CREATED": InstanceState.PENDING,
    "RUNNING": InstanceState.RUNNING,
    "EXITED": InstanceState.STOPPED,
    "ERROR": InstanceState.STOPPED,
    "TERMINATED": InstanceState.TERMINATED,
}


def _reason(response: httpx.Response) -> str:
    """Pull RunPod's own words out of an error body, for a legible log line.

    A refused provision must be explainable to a human reading a log at 7am; "http 422" alone
    tells them nothing. Handles all three observed error shapes: v2 problem-details
    (``{"title", "detail", "errors"}``), v1 (``{"error", "status"}``), and v1's bare
    ``[{"error": ...}]`` array.
    """
    try:
        body = response.json() or {}
    except ValueError:
        return response.text[:200]
    if isinstance(body, list):
        joined = "; ".join(str((e or {}).get("error") or e) for e in body)
        return joined[:300] or response.text[:200]
    title = str(body.get("title") or "")
    detail = str(body.get("detail") or body.get("error") or "")
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
        rest_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
        attempts: int = 4,
        base_delay: float = 2.0,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """Args:
        base_url: the v2 CATALOG dialect (offers/pricing).
        rest_url: the documented v1 POD-LIFECYCLE dialect — the one that speaks
            ``allowedCudaVersions``. See the module docstring for why the split exists.
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
        self.rest_url = (rest_url or cfg.runpod_rest_url).rstrip("/")
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
        base: str | None = None,
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
        url = f"{base or self.base_url}{path}"
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
            "imageName": image,
            "gpuTypeIds": [offer.offer_id],
            "gpuCount": 1,
            "env": dict(env),
            "cloudType": _cloud_tier(),
            # The reason this backend speaks the documented dialect at all: unset means "any
            # CUDA version is acceptable" — how a 12.4-driver relic billed a full bootstrap
            # before torch refused it (2026-08-01). The floor is torch's, shared across
            # backends (base.min_cuda).
            "allowedCudaVersions": _allowed_cuda_versions(),
            # SSH is how the launcher reaches the box; declaring 22/tcp is what makes
            # publicIp + portMappings["22"] appear for describe() to read back.
            "ports": ["22/tcp"],
            # Explicit even though this dialect would default it (to 50): every field left to a
            # default is a field the server chooses — the empty-body probe 201'd a $0.69/hr
            # Secure 4090 out of pure defaults. 20 GB holds image layers plus uv caches;
            # training artifacts live on the persistent /workspace volume, not here.
            "containerDiskInGb": 20,
        }
        if volume_gb:
            # A persistent volume at RunPod's conventional /workspace: checkpoints survive a
            # container restart, which on community cloud is a when, not an if.
            payload["volumeInGb"] = volume_gb
            payload["volumeMountPath"] = "/workspace"
        response = await self._request(
            "POST", "/pods", json_body=payload, idempotent=False, base=self.rest_url
        )
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
            response = await self._request("GET", f"/pods/{instance_id}", base=self.rest_url)
        except ProviderError as exc:
            return self._opaque(instance_id, InstanceState.UNKNOWN, str(exc))
        if response.status_code == 404:
            return self._opaque(instance_id, InstanceState.TERMINATED, _reason(response))
        if response.status_code >= 400:
            return self._opaque(instance_id, InstanceState.UNKNOWN, _reason(response))
        pod = body_json(response)
        if not isinstance(pod, dict):
            # "Spoke, but not in JSON" is a poll answer, not a crash: UNKNOWN, poll again.
            return self._opaque(instance_id, InstanceState.UNKNOWN, "non-JSON body")
        return self._to_instance(pod)

    async def list_instances(self) -> Sequence[Instance]:
        """Every pod on the account, any state — the billing-leak backstop's raw material.

        ``GET /pods`` on the v1 dialect returns a **bare array** (live-verified 2026-08-01; the
        v2 envelope shape is tolerated for the override case). Each pod runs through the same
        mapper as describe, so its ``name`` lands in ``raw["label"]`` for the sweep to join on.
        A failed list RAISES rather than returning ``[]``: to an orphan sweep, "could not look"
        and "nothing there" are opposite answers, and the wrong one ends the search while a GPU
        keeps billing. NOTE: list bodies carry each pod's full ``env`` — secrets — so neither
        this response nor ``Instance.raw`` from it may ever be logged wholesale.
        """
        response = await self._request("GET", "/pods", base=self.rest_url)
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
        pods = data if isinstance(data, list) else (data or {}).get("pods") or []
        return [self._to_instance(pod) for pod in pods]

    async def terminate(self, instance_id: str) -> None:
        """DELETE the pod. Idempotent by construction: 404 means already gone, which is success.

        The launcher calls this from every exit path including crash handlers; an exception on
        the second call would mask the original crash and could abort sibling cleanup while a
        GPU keeps billing.
        """
        response = await self._request("DELETE", f"/pods/{instance_id}", base=self.rest_url)
        if response.status_code == 404 or response.status_code < 400:
            return
        raise ProviderError(
            f"{self.name}: terminate {instance_id} refused — "
            f"http {response.status_code}: {_reason(response)}"
        )

    async def cost_so_far(self, instance_id: str) -> float | None:
        """Provider-clock uptime x the pod's own rate; None when that is not knowable.

        The v1 Pod carries no accrued-dollars or uptime-seconds field, but ``lastStartedAt`` is
        the provider's own clock — more trustworthy than the launcher's wall clock, which lies
        after a restart. A pod not desired-RUNNING, or with an unparseable timestamp, returns
        None and the ledger falls back to its own estimate (over-count, never under-count).
        """
        try:
            response = await self._request("GET", f"/pods/{instance_id}", base=self.rest_url)
        except ProviderError:
            return None
        if response.status_code >= 400:
            return None
        pod = body_json(response)
        if not isinstance(pod, dict):
            return None
        if str(pod.get("desiredStatus") or "").upper() != "RUNNING":
            return None
        started = _parse_pod_timestamp(str(pod.get("lastStartedAt") or ""))
        if started is None:
            return None
        hours = max(0.0, (datetime.now(UTC) - started).total_seconds() / 3600.0)
        return hours * float(pod.get("costPerHr") or 0.0)

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
        ssh_host = str(pod.get("publicIp") or "") or None
        ssh_port: int | None = None
        mapped = (pod.get("portMappings") or {}).get("22")
        if mapped:
            ssh_port = int(mapped)
        desired = str(pod.get("desiredStatus") or pod.get("status") or "").upper()
        state = _POD_STATES.get(desired, InstanceState.UNKNOWN)
        if state is InstanceState.RUNNING and not (ssh_host and ssh_port):
            # The v1 dialect's missing PENDING class, synthesized: "RUNNING" is a desire from
            # the moment of creation; the container is actually up only once it has an address.
            state = InstanceState.PENDING
        gpu_ids = pod.get("gpuTypeIds") or []
        gpu_name = str((pod.get("machine") or {}).get("gpuTypeId") or "") or str(
            gpu_ids[0] if gpu_ids else ""
        )
        return Instance(
            provider=self.name,
            instance_id=str(pod.get("id") or ""),
            state=state,
            gpu_name=gpu_name,
            price_per_hour=float(pod.get("costPerHr") or pod.get("cost") or 0.0),
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            # RunPod calls it "name"; the orphan sweep joins on raw["label"] across providers,
            # so the normalized key is guaranteed here (empty string when absent). raw carries
            # the pod's env — secrets — so it must never be logged wholesale.
            raw={**pod, "label": str(pod.get("name") or "")},
        )
