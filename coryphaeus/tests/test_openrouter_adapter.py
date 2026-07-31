"""The OpenRouter adapter, against a stubbed transport.

Two questions dominate. First, the shared one — **transient vs permanent** — because under GRPO a
failed rollout scores zero and zero teaches the policy "that worker was a bad choice"; OpenRouter
adds a twist by riding upstream failures *inside HTTP 200s*. Second, the one this adapter exists
for — **provenance**: the same model id can be served by several upstreams, so an unpinned or
mispinned call is a different served system, and treating its answer as this worker's poisons
calibration. Pinning is asserted on the request; provenance is verified on the response.
"""

from __future__ import annotations

import json

import httpx
import pytest

from coryphaeus.workers import Governor, WorkerBusy
from coryphaeus.workers.openrouter import (
    MissingApiKey,
    MissingPin,
    OpenRouterWorker,
    openrouter_spec,
)

SPEC = openrouter_spec(
    "qwen25-32b-or",
    "qwen/qwen2.5-32b-instruct",
    usd_per_mtok_in=0.5,
    usd_per_mtok_out=2.0,
    params_b=32.0,
)
PIN = "deepinfra"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _worker(handler, *, pin=PIN, **kwargs) -> tuple[OpenRouterWorker, httpx.AsyncClient]:
    client = _client(handler)
    worker = OpenRouterWorker(SPEC, pin=pin, api_key="test-key", client=client, **kwargs)
    return worker, client


async def _no_sleep(_seconds: float) -> None:
    return None


def _governor() -> Governor:
    return Governor(budgets={"openrouter": 4}, sleeper=_no_sleep, base_delay=0.0, jitter=0.0)


def _completion(text: str, *, provider: str | None = "DeepInfra", **extra) -> dict:
    """A normal response body. ``provider`` is the top-level provenance field; ``None`` omits it."""
    body: dict = {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 52, "completion_tokens": 6},
    }
    if provider is not None:
        body["provider"] = provider
    body.update(extra)
    return body


async def test_normal_completion_records_provenance():
    worker, client = _worker(lambda r: httpx.Response(200, json=_completion("\\boxed{42}")))
    async with client:
        result = await worker.invoke("q")
    assert result.ok
    assert result.text == "\\boxed{42}"
    assert result.tokens_in == 52
    assert result.tokens_out == 6
    assert result.meta["served_by"] == "DeepInfra"
    assert result.meta["think_leak_stripped"] is False


# --- pinning: the request-side determinism contract -----------------------------------------


async def test_pinning_payload_shape_is_exactly_the_contract():
    """order + allow_fallbacks + require_parameters, together. Any one missing reopens a hole:
    no order = route anywhere; fallbacks = route elsewhere under load; parameters not required =
    a provider may silently drop the sampler settings. Each is a different served system."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.read()))
        return httpx.Response(200, json=_completion("42"))

    worker, client = _worker(handler)
    async with client:
        await worker.invoke("q")
    assert seen["provider"] == {
        "order": ["deepinfra"],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


async def test_a_tuple_pin_preserves_preference_order():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.read()))
        return httpx.Response(200, json=_completion("42", provider="Together"))

    worker, client = _worker(handler, pin=("deepinfra", "together"))
    async with client:
        result = await worker.invoke("q")
    assert seen["provider"]["order"] == ["deepinfra", "together"]
    assert result.ok  # any listed upstream is correctly-pinned provenance
    assert result.meta["served_by"] == "Together"


def test_pin_is_required_to_construct():
    """An unpinned OpenRouter worker must be impossible by accident, not merely discouraged."""
    with pytest.raises(TypeError):
        OpenRouterWorker(SPEC, api_key="test-key")  # no pin kwarg at all
    for bad in ("", "   ", (), ("deepinfra", " ")):
        with pytest.raises(MissingPin):
            OpenRouterWorker(SPEC, pin=bad, api_key="test-key")


def test_missing_key_fails_at_construction_not_mid_run(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from coryphaeus import config

    config.settings.cache_clear()
    try:
        with pytest.raises(MissingApiKey):
            OpenRouterWorker(SPEC, pin=PIN, api_key="")
    finally:
        config.settings.cache_clear()


async def test_bearer_and_provenance_headers_are_sent():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        seen["metadata"] = request.headers.get("X-OpenRouter-Metadata")
        return httpx.Response(200, json=_completion("42"))

    worker, client = _worker(handler)
    async with client:
        await worker.invoke("q")
    assert seen["auth"] == "Bearer test-key"
    assert seen["metadata"] == "enabled"  # opt-in provenance channel, requested on every call


# --- provenance: the response-side verification ----------------------------------------------


async def test_served_by_mismatch_is_a_failure_naming_both():
    """The critical case: OpenRouter answered, the answer may even be right — but a different
    provider served it, so it is data from a different experiment. Never a silent success."""
    body = _completion("\\boxed{42}", provider="Together")
    worker, client = _worker(lambda r: httpx.Response(200, json=body))
    async with client:
        result = await worker.invoke("q")
    assert not result.ok
    assert result.text == ""  # zeroed so nothing downstream can score it by accident
    assert "Together" in (result.error or "")
    assert "deepinfra" in (result.error or "")
    assert result.meta["served_by"] == "Together"
    # The money was still spent; the ledger reports what was spent, not what we wish had happened.
    assert result.cost_usd == pytest.approx((52 * 0.5 + 6 * 2.0) / 1_000_000.0)


async def test_display_name_casing_is_not_a_mismatch():
    """Provenance is display-cased ("DeepInfra"); pins are slugs ("deepinfra"). A naive equality
    check would fail every healthy response — worse than no check at all."""
    worker, client = _worker(lambda r: httpx.Response(200, json=_completion("42")))
    async with client:
        result = await worker.invoke("q")
    assert result.ok


async def test_variant_pin_verifies_at_provider_granularity():
    """The response names the provider, not the endpoint variant — 'Google Vertex' must satisfy a
    'google-vertex/us-east5' pin. The variant half is enforced request-side by the order list."""
    body = _completion("42", provider="Google Vertex")
    worker, client = _worker(lambda r: httpx.Response(200, json=body), pin="google-vertex/us-east5")
    async with client:
        result = await worker.invoke("q")
    assert result.ok


async def test_absent_provenance_is_recorded_not_failed():
    """No provider field anywhere: the pin is still *enforced* request-side (allow_fallbacks
    false); verification just has nothing to verify. Failing here would brick the whole pool on a
    cosmetic schema change, so the blind spot is recorded honestly instead."""
    worker, client = _worker(lambda r: httpx.Response(200, json=_completion("42", provider=None)))
    async with client:
        result = await worker.invoke("q")
    assert result.ok
    assert result.meta["served_by"] is None


async def test_provenance_falls_back_to_router_metadata():
    """The opt-in metadata channel names the selected endpoint; if the top-level field ever goes
    away, the mispin check degrades to this rather than going blind."""
    metadata = {
        "endpoints": {
            "available": [
                {"provider": "Together", "model": "qwen/qwen2.5-32b-instruct", "selected": False},
                {"provider": "DeepInfra", "model": "qwen/qwen2.5-32b-instruct", "selected": True},
            ]
        }
    }
    body = _completion("42", provider=None, openrouter_metadata=metadata)
    worker, client = _worker(lambda r: httpx.Response(200, json=body))
    async with client:
        result = await worker.invoke("q")
    assert result.ok
    assert result.meta["served_by"] == "DeepInfra"


# --- reasoning control and the think-leak belt-and-suspenders --------------------------------


async def test_reasoning_disabled_by_default():
    """A reasoning model left thinking burns the budget and returns a truncated, plausible,
    WRONG answer — which scores as incompetence rather than misconfiguration."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.read()))
        return httpx.Response(200, json=_completion("42"))

    worker, client = _worker(handler)
    async with client:
        await worker.invoke("q")
    assert seen["reasoning"] == {"enabled": False}


async def test_think_none_omits_the_reasoning_field():
    """For a pinned endpoint that does not support reasoning control: require_parameters would
    otherwise filter it out of routing entirely."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.read()))
        return httpx.Response(200, json=_completion("42"))

    worker, client = _worker(handler, think=None)
    async with client:
        await worker.invoke("q")
    assert "reasoning" not in seen


async def test_think_leak_is_stripped_and_flagged():
    """Belt and suspenders: reasoning was disabled, a <think> block came back anyway. Left in
    place it would be scored as the answer."""
    leaked = "<think>Let me work this out... 6*7.</think>\n\\boxed{42}"
    worker, client = _worker(lambda r: httpx.Response(200, json=_completion(leaked)))
    async with client:
        result = await worker.invoke("q")
    assert result.ok
    assert result.text == "\\boxed{42}"
    assert result.meta["think_leak_stripped"] is True


async def test_unclosed_think_leak_fails_with_a_named_reason():
    """Opened and never closed: the whole turn was reasoning. 'empty response' would send a human
    hunting the wrong bug, so the leak is named."""
    worker, client = _worker(lambda r: httpx.Response(200, json=_completion("<think>forever and")))
    async with client:
        result = await worker.invoke("q")
    assert not result.ok
    assert result.meta["think_leak_stripped"] is True
    assert "think leak" in (result.error or "")


async def test_a_think_tag_quoted_mid_answer_is_not_stripped():
    """The stripper is anchored to the start: a model *talking about* think tags is content."""
    text = "Wrap reasoning in a <think> tag, e.g. <think>steps</think>, then answer."
    worker, client = _worker(lambda r: httpx.Response(200, json=_completion(text)))
    async with client:
        result = await worker.invoke("q")
    assert result.ok
    assert result.text == text
    assert result.meta["think_leak_stripped"] is False


async def test_empty_content_beside_a_reasoning_field_is_a_named_failure():
    """Some endpoints accept the disable knob, ignore it, and stream reasoning to a separate
    message field — observed live (qwen3-14b@deepinfra, 2026-07-31; the 32B on the same host
    honors the knob). When the budget dies before content, 'empty response' would hide the actual
    fix (raise max_tokens), so the burn is named and metered. The tokens are billed either way."""
    body = _completion("")
    body["choices"][0]["message"]["reasoning"] = "Okay, let's see. The user is asking for 17+25…"
    worker, client = _worker(lambda r: httpx.Response(200, json=body))
    async with client:
        result = await worker.invoke("q")
    assert not result.ok
    assert "reasoning field" in (result.error or "")
    assert result.meta["reasoning_chars"] > 0
    assert result.cost_usd > 0


async def test_max_tokens_floor_raises_a_lower_caller_cap():
    """Baseline 2026-07-31: qwen3 seats burn ~1.1k tokens of reasoning on MATH-hard prompts
    regardless of the disable knob, so a generic 1024 cap truncated them into 4.8-12.2% solo —
    fake incompetence. The floor buys the burn plus the answer."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.read()))
        return httpx.Response(200, json=_completion("42"))

    worker, client = _worker(handler, max_tokens_floor=2048)
    async with client:
        await worker.invoke("q", max_tokens=1024)
    assert seen["max_tokens"] == 2048


async def test_a_caller_cap_above_the_floor_stands():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.read()))
        return httpx.Response(200, json=_completion("42"))

    worker, client = _worker(handler, max_tokens_floor=2048)
    async with client:
        await worker.invoke("q", max_tokens=4096)
    assert seen["max_tokens"] == 4096


async def test_no_floor_leaves_the_caller_cap_alone():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.read()))
        return httpx.Response(200, json=_completion("42"))

    worker, client = _worker(handler)
    async with client:
        await worker.invoke("q", max_tokens=256)
    assert seen["max_tokens"] == 256


async def test_reasoning_beside_real_content_is_metered_not_failed():
    """The same quirk at an adequate budget: content arrives after the burn. The call succeeds —
    the burn is a cost property, recorded per call for the ledger and the world model."""
    body = _completion("\\boxed{42}")
    body["choices"][0]["message"]["reasoning"] = "step by step…"
    worker, client = _worker(lambda r: httpx.Response(200, json=body))
    async with client:
        result = await worker.invoke("q")
    assert result.ok
    assert result.text == "\\boxed{42}"
    assert result.meta["reasoning_chars"] > 0


async def test_a_reasoning_rejecting_endpoint_is_retried_without_the_knob():
    """Same shape as the Featherless chat_template_kwargs retry: never lose a working worker to
    an optimisation."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        calls.append(payload)
        if "reasoning" in payload:
            return httpx.Response(
                400, json={"error": {"code": 400, "message": "unknown parameter: reasoning"}}
            )
        return httpx.Response(200, json=_completion("42"))

    worker, client = _worker(handler)
    async with client:
        result = await worker.invoke("q")
    assert result.ok
    assert len(calls) == 2
    assert "reasoning" not in calls[1]


async def test_a_transient_400_is_not_mistaken_for_a_rejected_reasoning_knob():
    """A 400 wrapping an upstream 502 must reach the governor as WorkerBusy on the first response
    — burning the strip-retry on it would both hide the cause and waste a call."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({})
        return httpx.Response(400, json={"error": {"code": 502, "message": "provider hiccup"}})

    worker, client = _worker(handler)
    async with client:
        with pytest.raises(WorkerBusy):
            await worker.invoke("q")
    assert len(calls) == 1


# --- cost: the token-spend ledger ------------------------------------------------------------


async def test_cost_is_computed_from_usage_times_spec_prices():
    body = _completion("42")
    body["usage"] = {"prompt_tokens": 1000, "completion_tokens": 500}
    worker, client = _worker(lambda r: httpx.Response(200, json=body))
    async with client:
        result = await worker.invoke("q")
    # 1000 in @ $0.50/Mtok + 500 out @ $2.00/Mtok
    assert result.cost_usd == pytest.approx(0.0005 + 0.001)


def test_spec_requires_prices_and_carries_the_provider():
    """Prices have no default on purpose: a manifest omission must be a loud TypeError, not a
    worker that silently reports every call as free."""
    assert SPEC.provider == "openrouter"
    assert SPEC.units == 1
    with pytest.raises(TypeError):
        openrouter_spec("x", "y")  # type: ignore[call-arg]


# --- transient vs permanent, including errors that ride inside a 200 -------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
async def test_transient_statuses_reach_the_governor_as_busy(status):
    """429 is rate limiting, 408 OpenRouter's documented request timeout, 5xx includes 'no
    provider meeting routing requirements' — the pinned upstream briefly out of capacity."""
    worker, client = _worker(
        lambda r: httpx.Response(status, json={"error": {"code": status, "message": "later"}})
    )
    async with client:
        with pytest.raises(WorkerBusy):
            await worker.invoke("q")


async def test_upstream_error_inside_a_200_transient_is_busy():
    """OpenRouter's signature failure shape: HTTP 200, error object in the body — the router was
    fine, the upstream failed mid-generation. A 200 is not proof of an answer."""
    body = {
        "error": {
            "code": 502,
            "message": "Provider returned error",
            "metadata": {"provider_name": "DeepInfra"},
        }
    }
    worker, client = _worker(lambda r: httpx.Response(200, json=body))
    async with client:
        with pytest.raises(WorkerBusy) as exc:
            await worker.invoke("q")
    assert "DeepInfra" in str(exc.value)


async def test_upstream_error_inside_a_200_permanent_is_an_outcome():
    body = {
        "error": {
            "code": 403,
            "message": "flagged by moderation",
            "metadata": {"provider_name": "DeepInfra"},
        }
    }
    worker, client = _worker(lambda r: httpx.Response(200, json=body))
    async with client:
        result = await worker.invoke("q")
    assert not result.ok
    assert "403" in (result.error or "")
    assert "moderation" in (result.error or "")


async def test_an_embedded_transient_is_retried_by_the_governor_and_can_succeed():
    """The whole point of the classification: an upstream hiccup inside a 200 must not become a
    zero-scored rollout."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"error": {"code": 503, "message": "overloaded"}})
        return httpx.Response(200, json=_completion("42"))

    worker, client = _worker(handler)
    async with client:
        result = await _governor().invoke(worker, "q")
    assert result.ok
    assert result.attempts == 2


async def test_out_of_credits_402_is_permanent_and_not_retried():
    """Retrying 402 cannot mint credits; it only hides the reason and burns time."""
    body = {"error": {"code": 402, "message": "Insufficient credits"}}
    worker, client = _worker(lambda r: httpx.Response(402, json=body))
    async with client:
        result = await _governor().invoke(worker, "q")
    assert not result.ok
    assert result.attempts == 1
    assert "402" in (result.error or "")


async def test_a_timeout_is_transient_not_an_outcome():
    """A stopwatch must not hand out zeros — same contract as the Featherless adapter."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out")

    worker, client = _worker(handler)
    async with client:
        with pytest.raises(WorkerBusy, match="timed out"):
            await worker.invoke("q")


async def test_other_transport_failures_are_still_outcomes():
    """A refused connection is not a timeout — do not launder every transport error into a retry."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    worker, client = _worker(handler)
    async with client:
        result = await worker.invoke("q")
    assert not result.ok
    assert "transport" in (result.error or "")
