"""Cloud provider backends, against stubbed transports.

Same central question as the worker-adapter tests — transient vs permanent — plus the one that
is cloud-specific: **termination must be unconditionally safe.** The launcher calls terminate()
from every exit path including crash handlers, so a repeat terminate, or a terminate for an
instance that already vanished (404), must be a quiet success. An exception there would mask
the original crash and could abort sibling cleanup while a GPU keeps billing.

Response bodies mirror the real APIs as documented July 2026 (RunPod REST v2 via its published
OpenAPI spec; Vast.ai's console API reference) so the parsers meet the shapes they will meet in
production — including Vast's single-instance-under-a-plural-key quirk and its MB-denominated
``gpu_ram``.
"""

from __future__ import annotations

import json

import httpx
import pytest

from coryphaeus.cloud.providers import (
    CloudProvider,
    FakeCloudProvider,
    GpuOffer,
    InstanceState,
    ProviderError,
    RunPodProvider,
    VastProvider,
)
from coryphaeus.cloud.providers.runpod import MissingApiKey as RunPodMissingKey
from coryphaeus.cloud.providers.vast import MissingApiKey as VastMissingKey

ENV = {"CORYPHAEUS_RUN": "r6", "WORKER_POOL": "featherless"}
IMAGE = "ghcr.io/example/coryphaeus-trainer:cu124"
#: Clearly-fake key material — the tests assert plumbing, never validity.
SSH_PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1FakeKeyMaterialForOfflineTests coryphaeus-launcher"

RUNPOD_4090 = GpuOffer(
    provider="runpod",
    offer_id="NVIDIA GeForce RTX 4090",
    gpu_name="RTX 4090",
    vram_gb=24,
    price_per_hour=0.34,
)
VAST_4090 = GpuOffer(
    provider="vast", offer_id="18077244", gpu_name="RTX 4090", vram_gb=24, price_per_hour=0.31
)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def _no_sleep(_seconds: float) -> None:
    return None


def _runpod(handler, **kwargs) -> tuple[RunPodProvider, httpx.AsyncClient]:
    client = _client(handler)
    provider = RunPodProvider(
        api_key="test-key",
        base_url="https://api.runpod.io/v2",
        client=client,
        base_delay=0.0,
        sleeper=_no_sleep,
        **kwargs,
    )
    return provider, client


def _vast(handler, **kwargs) -> tuple[VastProvider, httpx.AsyncClient]:
    client = _client(handler)
    provider = VastProvider(
        api_key="test-key",
        base_url="https://console.vast.ai/api/v0",
        client=client,
        base_delay=0.0,
        sleeper=_no_sleep,
        **kwargs,
    )
    return provider, client


# --- realistic bodies ---------------------------------------------------------------------------
# RunPod: GET /v2/catalog/gpus?include=AVAILABILITY → {"gpus": [...]}; memory in GB, prices in
# $/hr under price.secure / price.community (verified July 2026 against the v2 OpenAPI spec).

RUNPOD_CATALOG = {
    "gpus": [
        {
            "id": "NVIDIA GeForce RTX 4090",
            "name": "RTX 4090",
            "memory": 24,
            "price": {"secure": 0.69, "community": 0.34},
            "availability": "HIGH",
        },
        {  # listed but sold out — must not be offered as "currently rentable"
            "id": "NVIDIA RTX A5000",
            "name": "RTX A5000",
            "memory": 24,
            "price": {"secure": 0.36, "community": 0.16},
            "availability": "NONE",
        },
        {
            "id": "NVIDIA A100 80GB PCIe",
            "name": "A100 80GB PCIe",
            "memory": 80,
            "price": {"secure": 1.64, "community": 1.19},
            "availability": "MEDIUM",
        },
        {  # secure-only card: community price is null
            "id": "NVIDIA H100 PCIe",
            "name": "H100 PCIe",
            "memory": 80,
            "price": {"secure": 2.39, "community": None},
            "availability": "HIGH",
        },
        {  # too little VRAM for the asks below
            "id": "NVIDIA GeForce RTX 3070",
            "name": "RTX 3070",
            "memory": 8,
            "price": {"secure": 0.20, "community": 0.11},
            "availability": "HIGH",
        },
    ]
}

#: Pod runtime block once RUNNING; ssh rides the ports array as the private-22/tcp entry.
RUNPOD_RUNTIME = {
    "uptime": 5400,
    "gpus": [{"id": "GPU-0", "utilization": 31}],
    "ports": [
        {"private": 8888, "public": None, "type": "http", "ip": None},
        {"private": 22, "public": 30022, "type": "tcp", "ip": "203.0.113.7"},
    ],
}


def _runpod_pod(
    status: str,
    *,
    runtime: dict | None = None,
    pod_id: str = "k2h8xpod1",
    name: str = "coryphaeus-r6",
) -> dict:
    return {
        "id": pod_id,
        "name": name,
        "status": status,
        "gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 1},
        "disk": 40,
        "cost": 0.34,
        "createdAt": "2026-07-30T12:00:00Z",
        "startedAt": "2026-07-30T12:01:12Z" if runtime else None,
        "runtime": runtime,
        "dataCenterId": None,
    }


# Vast: POST /api/v0/bundles → {"offers": [...]}; gpu_ram in MB, dph_* in $/hr, inet_* in Mb/s,
# inet_*_cost in $/GB (verified July 2026). The server should already have filtered — these rows
# include floor-violators on purpose, to prove the client-side re-check is a guarantee and not
# an echo of the query.
VAST_OFFERS = {
    "offers": [
        {  # good: cheap 4090 on a fast, reliable host (note gpu_ram just under 24*1024)
            "id": 18077244,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "gpu_ram": 24564,
            "dph_total": 0.31,
            "dph_base": 0.25,
            "min_bid": 0.155,
            "inet_down": 842.1,
            "inet_up": 615.3,
            "inet_down_cost": 0.0021,
            "inet_up_cost": 0.0044,
            "storage_cost": 0.12,
            "reliability2": 0.9971,
            "verification": "verified",
            "rentable": True,
            "geolocation": "Sweden, SE",
            "cuda_max_good": 12.8,
        },
        {  # good but pricier — must sort after the one above
            "id": 18077391,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "gpu_ram": 24564,
            "dph_total": 0.38,
            "dph_base": 0.31,
            "min_bid": 0.19,
            "inet_down": 1210.4,
            "inet_up": 980.2,
            "inet_down_cost": 0.0,
            "inet_up_cost": 0.0,
            "storage_cost": 0.10,
            "reliability2": 0.9903,
            "verification": "verified",
            "rentable": True,
            "geolocation": "Quebec, CA",
            "cuda_max_good": 12.8,
        },
        {  # cheapest of all, but residential DSL — the network floor exists for this host
            "id": 18070001,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "gpu_ram": 24564,
            "dph_total": 0.19,
            "dph_base": 0.16,
            "min_bid": 0.09,
            "inet_down": 87.4,
            "inet_up": 22.6,
            "inet_down_cost": 0.0,
            "inet_up_cost": 0.0,
            "storage_cost": 0.08,
            "reliability2": 0.9942,
            "verification": "verified",
            "rentable": True,
            "geolocation": "Ohio, US",
            "cuda_max_good": 12.4,
        },
        {  # flappy host — fails the reliability floor
            "id": 18070002,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "gpu_ram": 24564,
            "dph_total": 0.22,
            "dph_base": 0.18,
            "min_bid": 0.11,
            "inet_down": 512.0,
            "inet_up": 388.0,
            "inet_down_cost": 0.001,
            "inet_up_cost": 0.002,
            "storage_cost": 0.09,
            "reliability2": 0.912,
            "verification": "verified",
            "rentable": True,
            "geolocation": "Sofia, BG",
            "cuda_max_good": 12.2,
        },
        {  # too small for the ask below
            "id": 18070003,
            "gpu_name": "RTX 3070",
            "num_gpus": 1,
            "gpu_ram": 8192,
            "dph_total": 0.08,
            "dph_base": 0.06,
            "min_bid": 0.04,
            "inet_down": 640.0,
            "inet_up": 512.0,
            "inet_down_cost": 0.0,
            "inet_up_cost": 0.0,
            "storage_cost": 0.05,
            "reliability2": 0.995,
            "verification": "verified",
            "rentable": True,
            "geolocation": "Warsaw, PL",
            "cuda_max_good": 12.4,
        },
        {  # over the price ceiling used below
            "id": 18070004,
            "gpu_name": "RTX 5090",
            "num_gpus": 1,
            "gpu_ram": 32768,
            "dph_total": 0.55,
            "dph_base": 0.48,
            "min_bid": 0.27,
            "inet_down": 950.0,
            "inet_up": 940.0,
            "inet_down_cost": 0.0,
            "inet_up_cost": 0.0,
            "storage_cost": 0.15,
            "reliability2": 0.999,
            "verification": "verified",
            "rentable": True,
            "geolocation": "Oregon, US",
            "cuda_max_good": 12.8,
        },
    ]
}

VAST_START_DATE = 1753860000.0  # unix seconds, as the API reports start_date


def _vast_row(
    instance_id: int, *, label: str | None = "coryphaeus-r6", actual_status: str | None
) -> dict:
    """One instance object as the API reports it (fields verified July 2026). ``label`` is
    nullable on the wire — the backend must still guarantee the sweep's join key."""
    return {
        "id": instance_id,
        "actual_status": actual_status,
        "intended_status": "running",
        "cur_state": "running",
        "next_state": "running",
        "ssh_host": "ssh5.vast.ai",
        "ssh_port": 34567,
        "public_ipaddr": "203.0.113.44",
        "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "34567"}]},
        "dph_total": 0.31,
        "start_date": VAST_START_DATE,
        "duration": 259200.0,
        "label": label,
        "gpu_name": "RTX 4090",
    }


def _vast_instance(actual_status: str | None) -> dict:
    """GET /api/v0/instances/{id} body. The single instance arrives under the PLURAL
    ``instances`` key — the API's real shape (verified July 2026), and the reason this helper
    exists: a parser that guessed ``instance`` or a list would pass prettier fixtures. The
    LIST endpoint puts a real array under the same key (see the list tests)."""
    return {"instances": _vast_row(22913044, actual_status=actual_status)}


# --- protocol conformance -----------------------------------------------------------------------


def test_every_backend_satisfies_the_provider_protocol():
    backends = [
        FakeCloudProvider(),
        RunPodProvider(api_key="test-key"),
        VastProvider(api_key="test-key"),
    ]
    for backend in backends:
        assert isinstance(backend, CloudProvider)


# --- runpod -------------------------------------------------------------------------------------


async def test_runpod_offers_filter_sort_and_quote_community_pricing():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=RUNPOD_CATALOG)

    provider, client = _runpod(handler)
    async with client:
        offers = await provider.offers(min_vram_gb=24, max_price_per_hour=1.50)

    assert seen["path"] == "/v2/catalog/gpus"
    assert seen["params"]["include"] == "AVAILABILITY"
    assert seen["params"]["cloud"] == "COMMUNITY"
    # 4090 and A100 survive, cheapest first. A5000 is sold out (availability NONE), the H100 is
    # secure-only (community price null), the 3070 is under the VRAM floor.
    assert [(o.gpu_name, o.price_per_hour) for o in offers] == [
        ("RTX 4090", 0.34),
        ("A100 80GB PCIe", 1.19),
    ]
    assert all(o.provider == "runpod" for o in offers)


async def test_runpod_provision_sends_the_documented_body_with_env_injected():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.read())
        return httpx.Response(201, json=_runpod_pod("PROVISIONING"))

    provider, client = _runpod(handler)
    async with client:
        instance = await provider.provision(
            RUNPOD_4090, image=IMAGE, env=ENV, volume_gb=40, label="coryphaeus-r6"
        )

    body = seen["body"]
    assert seen["path"] == "/v2/pods"
    assert body["env"] == ENV  # the whole point: the box must boot knowing its run config
    assert body["image"] == IMAGE
    assert body["name"] == "coryphaeus-r6"
    assert body["gpu"] == {"id": "NVIDIA GeForce RTX 4090", "count": 1}
    assert body["cloud"] == "COMMUNITY"
    assert body["mounts"] == {"persistent": {"size": 40, "path": "/workspace"}}
    assert "22/tcp" in body["ports"]
    # `disk` is mandatory in practice though optional in the schema: a body without it 400s as
    # "no pod configuration parameters" (bisected live 2026-07-31). This pin keeps it mandatory
    # in our payload forever.
    assert body["disk"] >= 10
    assert instance.instance_id == "k2h8xpod1"
    assert instance.state is InstanceState.PENDING
    assert instance.price_per_hour == 0.34


async def test_runpod_describe_reads_ssh_out_of_runtime_ports():
    provider, client = _runpod(
        lambda r: httpx.Response(200, json=_runpod_pod("RUNNING", runtime=RUNPOD_RUNTIME))
    )
    async with client:
        instance = await provider.describe("k2h8xpod1")
    assert instance.state is InstanceState.RUNNING
    # The 8888/http mapping must not be mistaken for the ssh endpoint.
    assert instance.ssh_host == "203.0.113.7"
    assert instance.ssh_port == 30022
    assert instance.gpu_name == "NVIDIA GeForce RTX 4090"


async def test_runpod_describe_404_is_terminated_not_an_error():
    """A deleted pod has no record; "nothing to describe" must read as "stopped billing"."""
    body = {"title": "Not Found", "status": 404, "detail": "pod not found"}
    provider, client = _runpod(lambda r: httpx.Response(404, json=body))
    async with client:
        instance = await provider.describe("k2h8xpod1")
    assert instance.state is InstanceState.TERMINATED


async def test_runpod_describe_failure_is_unknown_not_an_exception():
    """The base contract: describe never raises — the caller decides what UNKNOWN means."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider, client = _runpod(handler)
    async with client:
        instance = await provider.describe("k2h8xpod1")
    assert instance.state is InstanceState.UNKNOWN


async def test_runpod_terminate_is_idempotent_including_the_404_second_call():
    """The launcher terminates from crash handlers too; the second call meets a 404 and MUST
    treat it as success — the pod being gone is exactly the goal state."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if len(calls) == 1:
            return httpx.Response(204)
        body = {"title": "Not Found", "status": 404, "detail": "pod not found"}
        return httpx.Response(404, json=body)

    provider, client = _runpod(handler)
    async with client:
        await provider.terminate("k2h8xpod1")
        await provider.terminate("k2h8xpod1")  # no raise — that is the assertion
    assert calls == ["DELETE /v2/pods/k2h8xpod1", "DELETE /v2/pods/k2h8xpod1"]


async def test_runpod_transient_status_is_retried_and_can_succeed():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            body = {"title": "Too Many Requests", "status": 429, "detail": "rate limited"}
            return httpx.Response(429, json=body)
        return httpx.Response(200, json=RUNPOD_CATALOG)

    provider, client = _runpod(handler)
    async with client:
        offers = await provider.offers(min_vram_gb=24, max_price_per_hour=1.50)
    assert calls["n"] == 2
    assert offers


async def test_runpod_exhausted_retries_become_a_provider_error_with_the_reason():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = {"title": "Service Unavailable", "status": 503, "detail": "upstream capacity"}
        return httpx.Response(503, json=body)

    provider, client = _runpod(handler, attempts=3)
    async with client:
        with pytest.raises(ProviderError, match="3 attempts.*upstream capacity"):
            await provider.offers(min_vram_gb=24, max_price_per_hour=1.50)
    assert calls["n"] == 3


async def test_runpod_permanent_4xx_is_not_retried_and_carries_the_api_reason():
    """Retrying a rejected request burns nothing but time — and would double-provision if the
    rejection were ever wrongly classified. One call, one clear error."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = {"title": "Unprocessable Entity", "status": 422, "detail": "gpu.id is unknown"}
        return httpx.Response(422, json=body)

    provider, client = _runpod(handler)
    async with client:
        with pytest.raises(ProviderError, match="gpu.id is unknown"):
            await provider.provision(
                RUNPOD_4090, image=IMAGE, env=ENV, volume_gb=0, label="coryphaeus-r6"
            )
    assert calls["n"] == 1


async def test_runpod_provision_timeout_is_not_blindly_retried():
    """A timed-out create may have created. Re-sending it could rent a second GPU, so the
    ambiguity must surface instead of being retried away."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("read timed out")

    provider, client = _runpod(handler)
    async with client:
        with pytest.raises(ProviderError, match="reconcile"):
            await provider.provision(
                RUNPOD_4090, image=IMAGE, env=ENV, volume_gb=40, label="coryphaeus-r6"
            )
    assert calls["n"] == 1


async def test_runpod_cost_so_far_uses_provider_uptime_not_wall_clock():
    provider, client = _runpod(
        lambda r: httpx.Response(200, json=_runpod_pod("RUNNING", runtime=RUNPOD_RUNTIME))
    )
    async with client:
        cost = await provider.cost_so_far("k2h8xpod1")
    assert cost == pytest.approx(5400 / 3600 * 0.34)  # 1.5 h at $0.34/hr


async def test_runpod_cost_so_far_is_none_before_the_pod_runs():
    """runtime is null until RUNNING; None hands the ledger its own fallback, a made-up zero
    would be mistaken for a report."""
    provider, client = _runpod(lambda r: httpx.Response(200, json=_runpod_pod("PROVISIONING")))
    async with client:
        assert await provider.cost_so_far("k2h8xpod1") is None


async def test_runpod_sends_bearer_auth():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=RUNPOD_CATALOG)

    provider, client = _runpod(handler)
    async with client:
        await provider.offers(min_vram_gb=24, max_price_per_hour=1.50)
    assert seen["auth"] == "Bearer test-key"


def test_runpod_missing_key_fails_at_construction_not_mid_run(monkeypatch):
    # setenv("") rather than delenv: load_dotenv never overrides an exported var, so the empty
    # export also shields this test from a developer's real .env.
    monkeypatch.setenv("RUNPOD_API_KEY", "")
    from coryphaeus import config

    config.settings.cache_clear()
    try:
        with pytest.raises(RunPodMissingKey):
            RunPodProvider()
    finally:
        config.settings.cache_clear()


async def test_runpod_list_instances_maps_every_pod_and_surfaces_labels():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["call"] = f"{request.method} {request.url.path}"
        pods = [
            _runpod_pod("RUNNING", runtime=RUNPOD_RUNTIME),
            _runpod_pod("EXITED", pod_id="z9forgot0", name="coryphaeus-r4-orphan"),
        ]
        return httpx.Response(200, json={"pods": pods})

    provider, client = _runpod(handler)
    async with client:
        instances = await provider.list_instances()

    assert seen["call"] == "GET /v2/pods"
    assert [(i.instance_id, i.state) for i in instances] == [
        ("k2h8xpod1", InstanceState.RUNNING),
        ("z9forgot0", InstanceState.STOPPED),
    ]
    # The orphan sweep joins on raw["label"]; RunPod's "name" field must land there.
    assert [i.raw["label"] for i in instances] == ["coryphaeus-r6", "coryphaeus-r4-orphan"]


async def test_runpod_list_instances_empty_account_is_an_empty_list():
    provider, client = _runpod(lambda r: httpx.Response(200, json={"pods": []}))
    async with client:
        assert list(await provider.list_instances()) == []


async def test_runpod_list_transient_429_is_retried_like_every_other_call():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            body = {"title": "Too Many Requests", "status": 429, "detail": "rate limited"}
            return httpx.Response(429, json=body)
        return httpx.Response(200, json={"pods": [_runpod_pod("RUNNING")]})

    provider, client = _runpod(handler)
    async with client:
        instances = await provider.list_instances()
    assert calls["n"] == 2
    assert len(instances) == 1


# --- vast ---------------------------------------------------------------------------------------


async def test_vast_offers_enforce_the_network_floors_client_side():
    """The floors are the point of this backend: our reward is network-bound, so the cheapest
    host on a DSL line is a slower step, not a cheaper one. The server rows here violate the
    floors on purpose — the client-side re-check is the guarantee, not an echo of the query."""
    provider, client = _vast(lambda r: httpx.Response(200, json=VAST_OFFERS))
    async with client:
        offers = await provider.offers(min_vram_gb=24, max_price_per_hour=0.40)

    # Survivors sorted cheapest-first. The 0.19 DSL host (87 Mb/s), the 0.22 flapper (0.912
    # reliability), the 8 GB card and the 0.55 over-ceiling card are all refused.
    assert [o.offer_id for o in offers] == ["18077244", "18077391"]
    best = offers[0]
    assert best.vram_gb == 24  # 24564 MB rounds to the card class, not down to 23
    assert best.price_per_hour == 0.31
    assert best.interruptible_price_per_hour == 0.155  # min_bid, $/hr
    assert best.egress_per_gb == 0.0044  # the host's upload price — what a chatty loop pays
    assert best.provider == "vast"


async def test_vast_offers_send_the_documented_server_side_filters():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = json.loads(request.read())
        return httpx.Response(200, json=VAST_OFFERS)

    provider, client = _vast(handler)
    async with client:
        await provider.offers(min_vram_gb=24, max_price_per_hour=0.40)

    query = seen["query"]
    assert seen["path"] == "/api/v0/bundles"
    assert query["type"] == "ondemand"
    assert query["rentable"] == {"eq": True}
    assert query["verified"] == {"eq": True}
    assert query["num_gpus"] == {"eq": 1}
    # MB, with half a GB of tolerance for hosts reporting usable (not nameplate) VRAM.
    assert query["gpu_ram"] == {"gte": 24 * 1024 - 512}
    assert query["dph_total"] == {"lte": 0.40}
    assert query["inet_down"] == {"gte": 200.0}
    assert query["reliability2"] == {"gte": 0.98}
    assert query["order"] == [["dph_total", "asc"]]


async def test_vast_provision_injects_env_and_returns_the_new_contract_id():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"success": True, "new_contract": 22913044})

    provider, client = _vast(handler)
    async with client:
        instance = await provider.provision(
            VAST_4090, image=IMAGE, env=ENV, volume_gb=40, label="coryphaeus-r6"
        )

    assert (seen["method"], seen["path"]) == ("PUT", "/api/v0/asks/18077244")
    body = seen["body"]
    assert body["env"] == ENV
    assert body["image"] == IMAGE
    assert body["disk"] == 40
    assert body["label"] == "coryphaeus-r6"
    assert body["runtype"] == "ssh"
    assert body["target_state"] == "running"
    # The id arrives under new_contract, not id — mapping it wrong strands the instance.
    assert instance.instance_id == "22913044"
    assert instance.state is InstanceState.PENDING


async def test_vast_a_snatched_offer_is_a_clear_provider_error():
    """Marketplace race: an offer can be rented out from under us between search and PUT."""
    body = {"success": False, "msg": "no such ask"}
    provider, client = _vast(lambda r: httpx.Response(200, json=body))
    async with client:
        with pytest.raises(ProviderError, match="no contract"):
            await provider.provision(
                VAST_4090, image=IMAGE, env=ENV, volume_gb=0, label="coryphaeus-r6"
            )


async def test_vast_provision_translates_public_key_into_an_onstart_script():
    """RunPod's stock templates consume PUBLIC_KEY at boot; Vast images do not, so the backend
    must build the same contract by hand — or the launcher can never reach the box it paid for."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"success": True, "new_contract": 22913044})

    provider, client = _vast(handler)
    env = {**ENV, "PUBLIC_KEY": SSH_PUBLIC_KEY}
    async with client:
        await provider.provision(
            VAST_4090, image=IMAGE, env=env, volume_gb=0, label="coryphaeus-r6"
        )

    onstart = seen["body"]["onstart"]
    assert SSH_PUBLIC_KEY in onstart
    assert "mkdir -p /root/.ssh" in onstart
    assert "chmod 700 /root/.ssh" in onstart
    assert ">> /root/.ssh/authorized_keys" in onstart  # append — never clobber existing keys
    assert "chmod 600 /root/.ssh/authorized_keys" in onstart
    assert "sshd" in onstart  # some images ship no running sshd; the script must start one
    # The key ALSO stays in env — harmless duplication beats a missing key on an image that
    # does understand the RunPod convention.
    assert seen["body"]["env"]["PUBLIC_KEY"] == SSH_PUBLIC_KEY
    assert seen["body"]["env"]["CORYPHAEUS_RUN"] == "r6"


async def test_vast_provision_without_public_key_sends_no_onstart():
    """Omission, not an empty script: an empty onstart still lands in the instance record and
    reads as intent, while absence leaves the image's own boot defaults untouched."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read())
        return httpx.Response(200, json={"success": True, "new_contract": 22913044})

    provider, client = _vast(handler)
    async with client:
        await provider.provision(
            VAST_4090, image=IMAGE, env=ENV, volume_gb=0, label="coryphaeus-r6"
        )
    assert "onstart" not in seen["body"]


async def test_vast_describe_unwraps_the_plural_key_quirk():
    provider, client = _vast(lambda r: httpx.Response(200, json=_vast_instance("running")))
    async with client:
        instance = await provider.describe("22913044")
    assert instance.state is InstanceState.RUNNING
    assert instance.ssh_host == "ssh5.vast.ai"
    assert instance.ssh_port == 34567
    assert instance.gpu_name == "RTX 4090"
    assert instance.price_per_hour == 0.31


@pytest.mark.parametrize(
    ("actual_status", "expected"),
    [
        (None, InstanceState.PENDING),  # container has not reported yet
        ("loading", InstanceState.PENDING),
        ("running", InstanceState.RUNNING),
        # Per the docs, these three "will never reach running" — the launcher must clean up.
        ("exited", InstanceState.STOPPED),
        ("offline", InstanceState.STOPPED),
        ("unknown", InstanceState.STOPPED),
        # A status this module has never heard of must surface as UNKNOWN, not crash or guess.
        ("hibernating", InstanceState.UNKNOWN),
    ],
)
async def test_vast_actual_status_mapping(actual_status, expected):
    provider, client = _vast(lambda r: httpx.Response(200, json=_vast_instance(actual_status)))
    async with client:
        instance = await provider.describe("22913044")
    assert instance.state is expected


async def test_vast_describe_404_is_terminated_not_an_error():
    body = {"success": False, "error": "not_found", "msg": "Instance not found"}
    provider, client = _vast(lambda r: httpx.Response(404, json=body))
    async with client:
        instance = await provider.describe("22913044")
    assert instance.state is InstanceState.TERMINATED


async def test_vast_terminate_treats_404_as_already_gone():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.method} {request.url.path}")
        if len(calls) == 1:
            body = {"success": True, "msg": "Instance destroyed successfully"}
            return httpx.Response(200, json=body)
        body = {"success": False, "error": "not_found", "msg": "Instance not found"}
        return httpx.Response(404, json=body)

    provider, client = _vast(handler)
    async with client:
        await provider.terminate("22913044")
        await provider.terminate("22913044")  # crash-handler double-call — must not raise
    assert calls == ["DELETE /api/v0/instances/22913044", "DELETE /api/v0/instances/22913044"]


async def test_vast_transient_status_is_retried_with_the_same_reasoning_as_workers():
    """429/5xx on the control plane mean "ask again", exactly as WorkerBusy does for workers —
    except no governor exists above this layer, so the provider retries internally."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"success": False, "msg": "try again later"})
        return httpx.Response(200, json=_vast_instance("running"))

    provider, client = _vast(handler)
    async with client:
        instance = await provider.describe("22913044")
    assert calls["n"] == 2
    assert instance.state is InstanceState.RUNNING


async def test_vast_provision_timeout_is_not_blindly_retried():
    """PUT /asks opens a rental contract per accepted call — the verb is PUT but the operation
    is not idempotent, so an in-flight death must surface, not re-send."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("read timed out")

    provider, client = _vast(handler)
    async with client:
        with pytest.raises(ProviderError, match="reconcile"):
            await provider.provision(
                VAST_4090, image=IMAGE, env=ENV, volume_gb=0, label="coryphaeus-r6"
            )
    assert calls["n"] == 1


async def test_vast_cost_so_far_is_wall_clock_free_given_an_injected_now():
    two_hours_in = VAST_START_DATE + 7200.0
    provider, client = _vast(
        lambda r: httpx.Response(200, json=_vast_instance("running")),
        now=lambda: two_hours_in,
    )
    async with client:
        cost = await provider.cost_so_far("22913044")
    assert cost == pytest.approx(2.0 * 0.31)  # 2 h at the instance's own dph_total


def test_vast_missing_key_fails_at_construction_not_mid_run(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "")
    from coryphaeus import config

    config.settings.cache_clear()
    try:
        with pytest.raises(VastMissingKey):
            VastProvider()
    finally:
        config.settings.cache_clear()


async def test_vast_list_instances_returns_the_real_array_with_labels():
    """The LIST endpoint's {"instances": [...]} is a real array — unlike show-instance, which
    wraps a single object under the same plural key. Both shapes are exercised in this file so
    neither parser can quietly assume the other's envelope."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["call"] = f"{request.method} {request.url.path}"
        rows = [
            _vast_row(22913044, actual_status="running"),
            _vast_row(22801077, label=None, actual_status="exited"),
        ]
        return httpx.Response(200, json={"instances": rows})

    provider, client = _vast(handler)
    async with client:
        instances = await provider.list_instances()

    # v1, with the trailing slash: v0's list answers 410 `deprecated_endpoint` (observed live
    # 2026-07-31), and bare paths 301 to their slash canonicals with HTML bodies.
    assert seen["call"] == "GET /api/v1/instances/"
    assert [(i.instance_id, i.state) for i in instances] == [
        ("22913044", InstanceState.RUNNING),
        ("22801077", InstanceState.STOPPED),
    ]
    # label is nullable on the wire; the sweep's join key must still exist, as "".
    assert [i.raw["label"] for i in instances] == ["coryphaeus-r6", ""]


async def test_vast_list_instances_empty_account_is_an_empty_list():
    provider, client = _vast(lambda r: httpx.Response(200, json={"instances": []}))
    async with client:
        assert list(await provider.list_instances()) == []


async def test_vast_list_transient_429_is_retried_like_every_other_call():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"success": False, "msg": "too many requests"})
        return httpx.Response(
            200, json={"instances": [_vast_row(22913044, actual_status="running")]}
        )

    provider, client = _vast(handler)
    async with client:
        instances = await provider.list_instances()
    assert calls["n"] == 2
    assert len(instances) == 1


# --- fake ---------------------------------------------------------------------------------------


async def test_fake_offers_are_deterministic_filtered_and_sorted():
    fake = FakeCloudProvider()
    offers = await fake.offers(min_vram_gb=16, max_price_per_hour=2.0)
    assert [o.offer_id for o in offers] == ["fake-4090", "fake-a100"]
    assert [o.offer_id for o in await fake.offers(min_vram_gb=48, max_price_per_hour=2.0)] == [
        "fake-a100"
    ]
    assert await fake.offers(min_vram_gb=48, max_price_per_hour=0.50) == []


async def test_fake_lifecycle_is_pending_until_polled_ready():
    fake = FakeCloudProvider(ready_after=2)
    offer = (await fake.offers(min_vram_gb=16, max_price_per_hour=2.0))[0]
    instance = await fake.provision(
        offer, image=IMAGE, env=ENV, volume_gb=10, label="coryphaeus-r6"
    )
    assert instance.state is InstanceState.PENDING
    assert (await fake.describe(instance.instance_id)).state is InstanceState.PENDING
    ready = await fake.describe(instance.instance_id)
    assert ready.state is InstanceState.RUNNING
    assert ready.ssh_host and ready.ssh_port


async def test_fake_records_exactly_what_provision_sent():
    """The layer above must be able to assert what it *sent* — env injection is how the rented
    box learns its run config, and a silent drop there only surfaces as a dead run."""
    fake = FakeCloudProvider()
    offer = (await fake.offers(min_vram_gb=16, max_price_per_hour=2.0))[0]
    await fake.provision(offer, image=IMAGE, env=ENV, volume_gb=25, label="coryphaeus-r6")
    assert fake.provisions == [
        {
            "instance_id": "fake-1",
            "offer_id": "fake-4090",
            "image": IMAGE,
            "env": ENV,
            "volume_gb": 25,
            "label": "coryphaeus-r6",
        }
    ]


async def test_fake_terminate_is_recorded_and_idempotent():
    fake = FakeCloudProvider(ready_after=1)
    offer = (await fake.offers(min_vram_gb=16, max_price_per_hour=2.0))[0]
    instance = await fake.provision(offer, image=IMAGE, env={}, volume_gb=0, label="t")
    await fake.terminate(instance.instance_id)
    await fake.terminate(instance.instance_id)  # crash-handler double-call
    await fake.terminate("never-existed")  # stale id from a crashed ledger — still a no-op
    assert fake.terminations == [instance.instance_id, instance.instance_id, "never-existed"]
    assert (await fake.describe(instance.instance_id)).state is InstanceState.TERMINATED
    assert (await fake.describe("never-existed")).state is InstanceState.TERMINATED


async def test_fake_scripted_failures_mirror_the_real_contract():
    failing = FakeCloudProvider(fail_provision=True)
    offer = (await failing.offers(min_vram_gb=16, max_price_per_hour=2.0))[0]
    with pytest.raises(ProviderError):
        await failing.provision(offer, image=IMAGE, env={}, volume_gb=0, label="t")

    fake = FakeCloudProvider(ready_after=1)
    instance = await fake.provision(offer, image=IMAGE, env={}, volume_gb=0, label="t")
    fake.fail_describe = True  # script an API outage mid-watch
    polled = await fake.describe(instance.instance_id)
    assert polled.state is InstanceState.UNKNOWN  # describe never raises — the caller decides


async def test_fake_cost_accrues_per_poll_and_freezes_at_termination():
    fake = FakeCloudProvider(ready_after=1, cost_per_poll=0.05)
    offer = (await fake.offers(min_vram_gb=16, max_price_per_hour=2.0))[0]
    instance = await fake.provision(offer, image=IMAGE, env={}, volume_gb=0, label="t")
    assert await fake.cost_so_far(instance.instance_id) == 0.0
    await fake.describe(instance.instance_id)
    await fake.describe(instance.instance_id)
    assert await fake.cost_so_far(instance.instance_id) == pytest.approx(0.10)
    await fake.terminate(instance.instance_id)
    await fake.describe(instance.instance_id)  # watching a corpse is free
    assert await fake.cost_so_far(instance.instance_id) == pytest.approx(0.10)
    assert await fake.cost_so_far("never-existed") is None


async def test_fake_list_instances_reflects_provision_and_terminate_history():
    fake = FakeCloudProvider(ready_after=1)
    assert list(await fake.list_instances()) == []

    offer = (await fake.offers(min_vram_gb=16, max_price_per_hour=2.0))[0]
    kept = await fake.provision(offer, image=IMAGE, env={}, volume_gb=0, label="r6-live")
    orphan = await fake.provision(offer, image=IMAGE, env={}, volume_gb=0, label="r4-orphan")
    await fake.describe(kept.instance_id)  # boots it (ready_after=1)
    await fake.terminate(orphan.instance_id)

    listed = await fake.list_instances()
    assert [(i.instance_id, i.state, i.raw["label"]) for i in listed] == [
        (kept.instance_id, InstanceState.RUNNING, "r6-live"),
        (orphan.instance_id, InstanceState.TERMINATED, "r4-orphan"),
    ]

    # Listing is a read: a second sweep sees the same world, and the meter has not moved.
    cost_before = await fake.cost_so_far(kept.instance_id)
    again = await fake.list_instances()
    assert [i.state for i in again] == [InstanceState.RUNNING, InstanceState.TERMINATED]
    assert await fake.cost_so_far(kept.instance_id) == cost_before


# --- non-JSON bodies: the first live Vast run's lesson ------------------------------------------
# A just-created contract answered 2xx with an EMPTY body and JSONDecodeError crashed the poll
# loop straight through describe()'s never-raises contract. Every verb now has a defined answer
# to "the provider spoke, but not in JSON."


async def test_vast_describe_treats_a_non_json_body_as_unknown():
    """The observed live failure (2026-07-31): empty 200 from a fresh contract must poll on."""
    provider, client = _vast(lambda r: httpx.Response(200, text=""))
    async with client:
        instance = await provider.describe("46343228")
    assert instance.state is InstanceState.UNKNOWN


async def test_runpod_describe_treats_a_non_json_body_as_unknown():
    provider, client = _runpod(lambda r: httpx.Response(200, text="<html>gateway</html>"))
    async with client:
        instance = await provider.describe("pod-1")
    assert instance.state is InstanceState.UNKNOWN


async def test_vast_cost_returns_none_on_a_non_json_body():
    provider, client = _vast(lambda r: httpx.Response(200, text=""))
    async with client:
        assert await provider.cost_so_far("46343228") is None


async def test_runpod_cost_returns_none_on_a_non_json_body():
    provider, client = _runpod(lambda r: httpx.Response(200, text=""))
    async with client:
        assert await provider.cost_so_far("pod-1") is None


async def test_vast_offers_raise_a_named_error_on_a_non_json_body():
    provider, client = _vast(lambda r: httpx.Response(200, text="<html>cdn hiccup</html>"))
    async with client:
        with pytest.raises(ProviderError, match="non-JSON"):
            await provider.offers(min_vram_gb=24, max_price_per_hour=0.60)


async def test_runpod_list_raises_on_a_non_json_body():
    """could-not-look must never read as nothing-there while a GPU bills."""
    provider, client = _runpod(lambda r: httpx.Response(200, text=""))
    async with client:
        with pytest.raises(ProviderError, match="could-not-look"):
            await provider.list_instances()


async def test_vast_provision_2xx_with_garbage_says_reconcile_not_retry():
    """A 2xx whose body we cannot read means a contract MAY exist — same ambiguity as a
    timed-out create, same rule: reconcile, never blind-retry into a second rental."""
    provider, client = _vast(lambda r: httpx.Response(200, text=""))
    offer = GpuOffer(
        provider="vast", offer_id="123", gpu_name="RTX 3090", vram_gb=24, price_per_hour=0.12
    )
    async with client:
        with pytest.raises(ProviderError, match="reconcile"):
            await provider.provision(offer, image="img", env={}, volume_gb=10, label="coryphaeus-x")


async def test_vast_describe_uses_the_trailing_slash_canonical_path():
    """The bare path 301s to its slash canonical with an HTML body; a JSON client that does not
    land on the canonical reads ten minutes of UNKNOWN and times out (lived it, 2026-07-31)."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(200, json={"instances": _vast_row(46343811, actual_status="running")})

    provider, client = _vast(handler)
    async with client:
        instance = await provider.describe("46343811")
    assert seen["path"] == "/api/v0/instances/46343811/"
    assert instance.state is InstanceState.RUNNING
