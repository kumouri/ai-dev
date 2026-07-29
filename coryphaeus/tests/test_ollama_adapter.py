"""The Ollama adapter, against a stubbed transport — no server needed.

The thinking-model case is the reason this file exists. A reasoning model fills ``message.thinking``
and leaves ``message.content`` empty until it finishes reasoning, so a modest token budget returns a
*blank answer* while looking like a healthy 200. Left undetected that scores zero on every rollout
and looks like a bad conductor rather than a misconfigured worker.
"""

from __future__ import annotations

import json

import httpx

from coryphaeus.workers import WorkerBusy
from coryphaeus.workers.ollama import OllamaWorker, ollama_spec

SPEC = ollama_spec("q4", "qwen3.5:4b", params_b=4.0, size_gb=3.2)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _chat_body(**message) -> dict:
    return {
        "message": {"role": "assistant", **message},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 33,
        "eval_count": 12,
    }


async def test_normal_answer():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_body(content="\\boxed{42}"))

    async with _client(handler) as client:
        result = await OllamaWorker(SPEC, client=client).invoke("q")
    assert result.ok
    assert result.text == "\\boxed{42}"
    assert result.tokens_in == 33
    assert result.tokens_out == 12


async def test_thinking_only_response_is_a_named_failure_not_a_mystery():
    body = _chat_body(content="", thinking="Let me think about this at length. " * 20)
    body["done_reason"] = "length"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    async with _client(handler) as client:
        result = await OllamaWorker(SPEC, client=client).invoke("q", max_tokens=64)

    assert not result.ok
    assert "reasoning consumed" in (result.error or "")
    assert "64-token budget" in (result.error or "")
    assert result.meta["thinking_chars"] > 0


async def test_truncation_without_thinking_says_truncated():
    body = _chat_body(content="")
    body["done_reason"] = "length"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    async with _client(handler) as client:
        result = await OllamaWorker(SPEC, client=client).invoke("q", max_tokens=8)
    assert not result.ok
    assert "truncated" in (result.error or "")


def _capturing_handler(seen: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.read()))
        return httpx.Response(200, json=_chat_body(content="42"))

    return handler


async def test_think_false_is_sent_by_default():
    seen: dict = {}
    async with _client(_capturing_handler(seen)) as client:
        await OllamaWorker(SPEC, client=client).invoke("q")
    assert seen["think"] is False


async def test_think_can_be_omitted_entirely():
    seen: dict = {}
    async with _client(_capturing_handler(seen)) as client:
        await OllamaWorker(SPEC, client=client, think=None).invoke("q")
    assert "think" not in seen


async def test_a_400_on_think_is_retried_without_it():
    """An older server that rejects `think` must not turn an optimisation into a hard failure."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.read())
        calls.append(payload)
        if "think" in payload:
            return httpx.Response(400, json={"error": "unknown field think"})
        return httpx.Response(200, json=_chat_body(content="42"))

    async with _client(handler) as client:
        result = await OllamaWorker(SPEC, client=client).invoke("q")
    assert result.ok
    assert len(calls) == 2
    assert "think" in calls[0] and "think" not in calls[1]


async def test_system_prompt_is_sent_first():
    seen: dict = {}
    async with _client(_capturing_handler(seen)) as client:
        await OllamaWorker(SPEC, client=client).invoke("q", system="be terse")
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]


async def test_429_raises_worker_busy_for_the_governor():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "busy"})

    async with _client(handler) as client:
        try:
            await OllamaWorker(SPEC, client=client).invoke("q")
        except WorkerBusy:
            return
    raise AssertionError("a 429 must surface as WorkerBusy so the governor can back off")


async def test_server_error_is_returned_not_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async with _client(handler) as client:
        result = await OllamaWorker(SPEC, client=client).invoke("q")
    assert not result.ok
    assert "http 500" in (result.error or "")


async def test_transport_failure_is_returned_not_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with _client(handler) as client:
        result = await OllamaWorker(SPEC, client=client).invoke("q")
    assert not result.ok
    assert "transport" in (result.error or "")


def test_units_default_from_footprint_versus_vram():
    """A model that spills past VRAM must be near-serial, or it drags the whole pool down."""
    assert ollama_spec("small", "m", size_gb=6.0).units == 1
    assert ollama_spec("huge", "m", size_gb=74.7).units == 4
    assert ollama_spec("explicit", "m", size_gb=74.7, units=1).units == 1
