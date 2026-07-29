"""The remote adapter, against a stubbed transport.

The central question is **transient vs permanent**, because it decides whether a failure becomes a
routing lesson. Under GRPO a failed rollout scores zero, and zero is how the policy learns "that
worker was a bad choice". A provider briefly at capacity must therefore be retried, not scored —
otherwise infrastructure noise gets laundered into a routing preference.

Both cases here are real, observed against the live API on 2026-07-29.
"""

from __future__ import annotations

import httpx
import pytest

from coryphaeus.workers import Governor, WorkerBusy
from coryphaeus.workers.featherless import FeatherlessWorker, MissingApiKey, featherless_spec

SPEC = featherless_spec("qwen25-32b", "Qwen/Qwen2.5-32B-Instruct", params_b=32.0, units=2)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _worker(handler) -> tuple[FeatherlessWorker, httpx.AsyncClient]:
    client = _client(handler)
    return FeatherlessWorker(SPEC, api_key="test-key", client=client), client


async def _no_sleep(_seconds: float) -> None:
    return None


def _completion(text: str) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 52, "completion_tokens": 6},
    }


async def test_normal_completion():
    worker, client = _worker(lambda r: httpx.Response(200, json=_completion("\\boxed{42}")))
    async with client:
        result = await worker.invoke("q")
    assert result.ok
    assert result.text == "\\boxed{42}"
    assert result.tokens_in == 52
    assert result.tokens_out == 6


async def test_capacity_503_is_transient_not_an_outcome():
    """Observed live: Qwen2.5-Math-7B-Instruct returns 503 capacity_exhausted when spinning up."""
    body = {
        "error": {
            "message": "Qwen/Qwen2.5-Math-7B-Instruct is temporarily at capacity. "
            "Please try again shortly.",
            "type": "server_error",
            "code": "capacity_exhausted",
        }
    }
    worker, client = _worker(lambda r: httpx.Response(503, json=body))
    async with client:
        with pytest.raises(WorkerBusy) as exc:
            await worker.invoke("q")
    assert "capacity_exhausted" in str(exc.value)


@pytest.mark.parametrize("status", [429, 502, 503, 504])
async def test_all_transient_statuses_reach_the_governor_as_busy(status):
    worker, client = _worker(lambda r: httpx.Response(status, json={"error": {"message": "later"}}))
    async with client:
        with pytest.raises(WorkerBusy):
            await worker.invoke("q")


async def test_a_transient_failure_is_retried_and_can_succeed():
    """The whole point: a hiccup must not become a zero-scored rollout."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": {"code": "capacity_exhausted"}})
        return httpx.Response(200, json=_completion("42"))

    worker, client = _worker(handler)
    governor = Governor(budgets={"featherless": 4}, sleeper=_no_sleep, base_delay=0.0, jitter=0.0)
    async with client:
        result = await governor.invoke(worker, "q")
    assert result.ok
    assert result.attempts == 2
    assert governor.busy_events == 1


async def test_gated_model_403_is_permanent_and_not_retried():
    """Observed live: meta-llama/Llama-3.3-70B-Instruct is gated behind an OAuth connection.

    Retrying a permanent failure just burns the concurrency budget, so it is returned as an outcome
    with the provider's own reason attached.
    """
    body = {
        "error": {
            "message": "This model is gated. Connect HuggingFace for this organization ... "
            "to verify access.",
            "type": "invalid_request_error",
            "code": "model_gated_needs_oauth",
        }
    }
    worker, client = _worker(lambda r: httpx.Response(403, json=body))
    governor = Governor(budgets={"featherless": 4}, sleeper=_no_sleep, base_delay=0.0, jitter=0.0)
    async with client:
        result = await governor.invoke(worker, "q")
    assert not result.ok
    assert result.attempts == 1  # not retried
    assert "model_gated_needs_oauth" in (result.error or "")


async def test_400_completion_error_is_transient_despite_being_a_4xx():
    """Observed live: Qwen2.5-14B-Instruct returned 400 completion_error, then answered correctly.

    Status alone cannot classify this — a 4xx normally means "your request is wrong, do not retry".
    Scoring it as an outcome would mark a working worker as a bad routing choice.
    """
    body = {
        "error": {"message": "This model is temporarily unavailable", "code": "completion_error"}
    }
    worker, client = _worker(lambda r: httpx.Response(400, json=body))
    async with client:
        with pytest.raises(WorkerBusy):
            await worker.invoke("q")


async def test_enable_thinking_false_is_sent_by_default():
    """A reasoning model left thinking returns a truncated, plausible, WRONG answer.

    Observed live: Qwen3-32B spent 256 tokens reasoning and answered 25 instead of 42. With thinking
    disabled it answered correctly in 6 tokens.
    """
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen.update(_json.loads(request.read()))
        return httpx.Response(200, json=_completion("42"))

    worker, client = _worker(handler)
    async with client:
        await worker.invoke("q")
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


async def test_a_template_that_rejects_the_kwarg_is_retried_without_it():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        payload = _json.loads(request.read())
        calls.append(payload)
        if "chat_template_kwargs" in payload:
            return httpx.Response(400, json={"error": {"message": "unknown kwarg", "code": "bad"}})
        return httpx.Response(200, json=_completion("42"))

    worker, client = _worker(handler)
    async with client:
        result = await worker.invoke("q")
    assert result.ok
    assert len(calls) == 2


async def test_a_transient_400_is_not_mistaken_for_a_rejected_kwarg():
    """The kwarg-retry must not swallow a transient 400 — that burns a retry and hides the cause."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({})
        return httpx.Response(400, json={"error": {"code": "completion_error", "message": "later"}})

    worker, client = _worker(handler)
    async with client:
        with pytest.raises(WorkerBusy):
            await worker.invoke("q")
    assert len(calls) == 1  # classified as transient on the first response, no kwarg retry


async def test_error_reason_survives_a_non_json_body():
    worker, client = _worker(lambda r: httpx.Response(500, text="upstream exploded"))
    async with client:
        result = await worker.invoke("q")
    assert not result.ok
    assert "upstream exploded" in (result.error or "")


async def test_bearer_token_is_sent():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=_completion("42"))

    worker, client = _worker(handler)
    async with client:
        await worker.invoke("q")
    assert seen["auth"] == "Bearer test-key"


def test_missing_key_fails_at_construction_not_mid_run(monkeypatch):
    monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
    from coryphaeus import config

    config.settings.cache_clear()
    try:
        with pytest.raises(MissingApiKey):
            FeatherlessWorker(SPEC, api_key="")
    finally:
        config.settings.cache_clear()


def test_units_are_taken_from_the_spec_not_guessed():
    assert SPEC.units == 2
    assert featherless_spec("x", "y", params_b=32.0).units == 2  # fallback now has four tiers
