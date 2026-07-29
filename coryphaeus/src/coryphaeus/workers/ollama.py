"""Ollama worker — the local half of the pool.

Free, private, and the reason the whole harness can be exercised end to end without an account. The
constraint here is VRAM rather than a quota: a model larger than available VRAM spills to system RAM
and becomes extremely slow, so such workers are given a high unit cost to keep them near-serial.

**Thinking models return their reasoning in a separate field.** On a reasoning-capable model,
``/api/chat`` fills ``message.thinking`` and leaves ``message.content`` empty until the reasoning
ends — so a modest ``num_predict`` produces a *blank answer* with ``done_reason: "length"``, having
spent the entire budget thinking. That is silent, looks like a broken model, and would have quietly
zeroed every reward in a run. Hence ``think=False`` by default (and an explicit ``truncated`` error
when the budget runs out anyway) rather than a mystery.
"""

from __future__ import annotations

import httpx

from ..config import settings
from .base import WorkerBusy, WorkerResult, WorkerSpec, timed

#: Rough VRAM available for inference, in GB. Models above this still work but spill to system RAM.
#: Only used to *default* a unit cost; override per worker when you know better.
ASSUMED_VRAM_GB = 24.0


class OllamaWorker:
    """One Ollama model, invoked over ``/api/chat``."""

    def __init__(
        self,
        spec: WorkerSpec,
        *,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 600.0,
        think: bool | None = False,
    ) -> None:
        """Args:
        think: ``False`` (default) disables reasoning, so the token budget buys *answer*.
            ``True`` enables it — worth an arm of its own once the baseline exists, with a much
            larger ``max_tokens``. ``None`` omits the field entirely and takes the model's default.
        """
        self.spec = spec
        self.base_url = (base_url or settings().ollama_base_url).rstrip("/")
        self._client = client
        self._timeout = timeout
        self.think = think

    async def _post(self, payload: dict) -> httpx.Response:
        if self._client is not None:
            return await self._client.post(f"{self.base_url}/api/chat", json=payload)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.post(f"{self.base_url}/api/chat", json=payload)

    async def invoke(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> WorkerResult:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": self.spec.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if self.think is not None:
            payload["think"] = self.think

        with timed() as t:
            try:
                response = await self._post(payload)
                # Older Ollama builds reject an unknown `think` field. Retry once without it rather
                # than reporting a hard failure for a knob that is only an optimisation.
                if response.status_code == 400 and "think" in payload:
                    payload.pop("think")
                    response = await self._post(payload)
            except httpx.HTTPError as exc:
                return WorkerResult(
                    worker=self.spec.name,
                    text="",
                    ok=False,
                    latency_s=t.elapsed,
                    error=f"transport: {type(exc).__name__}: {exc}",
                )

        if response.status_code == 429:
            raise WorkerBusy(f"{self.spec.name}: ollama returned 429")
        if response.status_code >= 400:
            return WorkerResult(
                worker=self.spec.name,
                text="",
                ok=False,
                latency_s=t.elapsed,
                error=f"http {response.status_code}: {response.text[:200]}",
            )

        try:
            body = response.json()
        except ValueError as exc:
            return WorkerResult(
                worker=self.spec.name,
                text="",
                ok=False,
                latency_s=t.elapsed,
                error=f"decode: {exc}",
            )

        message = body.get("message") or {}
        text = message.get("content", "") or ""
        thinking = message.get("thinking", "") or ""
        tokens_in = int(body.get("prompt_eval_count") or 0)
        tokens_out = int(body.get("eval_count") or 0)
        done_reason = body.get("done_reason")

        # Name the failure precisely. "empty response" sends you hunting the wrong thing; "the
        # budget went on reasoning" tells you to raise max_tokens or turn thinking off.
        error: str | None = None
        if not text.strip():
            if thinking.strip():
                error = (
                    f"empty content: {len(thinking)} chars of reasoning consumed the "
                    f"{max_tokens}-token budget (done_reason={done_reason}); "
                    "raise max_tokens or set think=False"
                )
            elif done_reason == "length":
                error = f"truncated: hit the {max_tokens}-token budget with no content"
            else:
                error = f"empty response (done_reason={done_reason})"

        return WorkerResult(
            worker=self.spec.name,
            text=text,
            ok=not error,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=t.elapsed,
            cost_usd=self.spec.cost(tokens_in, tokens_out),
            error=error,
            meta={"done_reason": done_reason, "thinking_chars": len(thinking)},
        )


def ollama_spec(
    name: str,
    model: str,
    *,
    params_b: float | None = None,
    size_gb: float | None = None,
    tags: tuple[str, ...] = (),
    units: int | None = None,
) -> WorkerSpec:
    """Build a spec for a local model.

    ``units`` defaults to 1, or 4 when the model's on-disk size exceeds :data:`ASSUMED_VRAM_GB` —
    a model that spills to system RAM should not be run alongside anything else.
    """
    if units is None:
        units = 4 if (size_gb is not None and size_gb > ASSUMED_VRAM_GB) else 1
    return WorkerSpec(
        name=name,
        model=model,
        provider="ollama",
        params_b=params_b,
        units=units,
        tags=tags,
    )
