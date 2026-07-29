"""Featherless worker — the remote half of the pool, on an OpenAI-compatible API.

Flat-rate billing is the point: GRPO's appetite for rollouts makes per-token worker billing the
dominant cost of this whole project, and a flat rate removes it. What replaces it is a **concurrency
unit** budget (sub-16B model = 1 unit, 70B+ = 4, and a Premium account holds 4 total before HTTP
429) — which is why :mod:`coryphaeus.workers.governor` exists, and why nothing here retries alone.
"""

from __future__ import annotations

import httpx

from ..config import settings
from .base import WorkerBusy, WorkerResult, WorkerSpec, timed, units_for_params


class MissingApiKey(RuntimeError):
    """No ``FEATHERLESS_API_KEY`` available. Raised at construction, not mid-run."""


class FeatherlessWorker:
    """One Featherless-hosted model, invoked over ``/chat/completions``."""

    def __init__(
        self,
        spec: WorkerSpec,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 300.0,
    ) -> None:
        cfg = settings()
        key = api_key or cfg.featherless_api_key
        if not key:
            raise MissingApiKey(
                "FEATHERLESS_API_KEY is not set. Put it in .env (see .env.example); "
                "the local Ollama pool and the offline test suite do not need it."
            )
        self.spec = spec
        self._api_key = key
        self.base_url = (base_url or cfg.featherless_base_url).rstrip("/")
        self._client = client
        self._timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def _post(self, payload: dict) -> httpx.Response:
        url = f"{self.base_url}/chat/completions"
        if self._client is not None:
            return await self._client.post(url, json=payload, headers=self._headers)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            return await client.post(url, json=payload, headers=self._headers)

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

        payload = {
            "model": self.spec.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        with timed() as t:
            try:
                response = await self._post(payload)
            except httpx.HTTPError as exc:
                return WorkerResult(
                    worker=self.spec.name,
                    text="",
                    ok=False,
                    latency_s=t.elapsed,
                    error=f"transport: {type(exc).__name__}: {exc}",
                )

        # 429 is the unit budget talking. Let the governor decide how long to wait.
        if response.status_code == 429:
            raise WorkerBusy(f"{self.spec.name}: 429 (concurrency units exhausted)")
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

        choices = body.get("choices") or []
        text = ""
        finish_reason = None
        if choices:
            text = ((choices[0].get("message") or {}).get("content")) or ""
            finish_reason = choices[0].get("finish_reason")
        usage = body.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or 0)

        return WorkerResult(
            worker=self.spec.name,
            text=text,
            ok=bool(text.strip()),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=t.elapsed,
            cost_usd=self.spec.cost(tokens_in, tokens_out),
            error=None if text.strip() else "empty response",
            meta={"finish_reason": finish_reason},
        )


def featherless_spec(
    name: str,
    model: str,
    *,
    params_b: float | None = None,
    tags: tuple[str, ...] = (),
    units: int | None = None,
) -> WorkerSpec:
    """Build a spec, defaulting ``units`` from the published size-to-unit rule."""
    return WorkerSpec(
        name=name,
        model=model,
        provider="featherless",
        params_b=params_b,
        units=units if units is not None else units_for_params(params_b),
        tags=tags,
    )
