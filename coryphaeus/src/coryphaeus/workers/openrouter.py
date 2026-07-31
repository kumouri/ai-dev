"""OpenRouter worker — many upstreams behind one OpenAI-compatible API, pinned per worker.

OpenRouter is a *router*: the same model id can be served by several upstream providers with
different quantizations, context handling, and sampler behavior. For an experiment calibrating a
policy over specific served systems, silent rerouting is poison — worker N would be a moving
target, and a reward difference between runs could be a provider swap rather than anything the
conductor did. So this adapter refuses to exist unpinned, disables fallbacks on every request, and
verifies on every response that the pinned provider actually served it. A mispinned response is
returned as a **failure**, never a silent success: it is data from a different experiment.

Field shapes verified against openrouter.ai docs, July 2026:

- request ``provider`` preferences: ``{"order": [<slug>, ...], "allow_fallbacks": false,
  "require_parameters": true}`` — slugs are lowercase (``"deepinfra"``, variant form
  ``"deepinfra/turbo"``).
- request ``reasoning``: ``{"enabled": false}`` fully disables reasoning (distinct from
  ``exclude``, which reasons silently and still burns the token budget).
- error envelope: ``{"error": {"code": <number>, "message": str, "metadata": {...}}}`` — and it
  can arrive **inside an HTTP 200** when the upstream fails mid-generation, so a 200 is not proof
  of an answer.
- provenance: the serving provider rides in the response body (top-level ``provider``; the opt-in
  ``X-OpenRouter-Metadata: enabled`` header additionally yields ``openrouter_metadata`` with the
  selected endpoint). Provenance names are display-cased ("DeepInfra") while pins are slugs
  ("deepinfra"), so the comparison normalizes.
"""

from __future__ import annotations

import re

import httpx

from ..config import settings
from .base import TRANSIENT_STATUSES, WorkerBusy, WorkerResult, WorkerSpec, timed


class MissingApiKey(RuntimeError):
    """No ``OPENROUTER_API_KEY`` available. Raised at construction, not mid-run."""


class MissingPin(ValueError):
    """An OpenRouter worker was constructed without a usable upstream pin.

    Deliberately fatal at construction: an unpinned OpenRouter call may be served by any provider
    the router likes, which is a *different served system* on every request. One accidental
    unpinned worker would quietly poison a whole calibration run, so the failure mode is made
    impossible rather than merely discouraged.
    """


def _error_block(response: httpx.Response) -> dict:
    try:
        error = (response.json() or {}).get("error")
    except ValueError:
        return {}
    return error if isinstance(error, dict) else {}


def _as_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _transient_code(code: int) -> bool:
    """OpenRouter's transient band: the shared statuses plus 408 and the whole 5xx range.

    408 is their documented request timeout — a moment, not a verdict on the request. 5xx beyond
    502/503/504 covers upstream provider errors, which OpenRouter relays with upstream-ish codes.
    Permanent 4xx (400 bad request, 401 bad key, 402 out of credits, 403 moderation) stay
    outcomes: retrying them cannot succeed and only hides the reason.
    """
    return code in TRANSIENT_STATUSES or code == 408 or code >= 500


def _embedded_reason(error: dict) -> str:
    """Render OpenRouter's error object (numeric ``code``, ``message``, optional metadata) legibly.

    ``metadata.provider_name`` is included when present because "which upstream broke" is the bit
    a human needs — the router's own message often just says a provider failed.
    """
    code = error.get("code")
    message = str(error.get("message") or "").strip()
    metadata = error.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    upstream = str(metadata.get("provider_name") or "").strip()
    reason = f"{code}: {message}" if code is not None else message
    if upstream:
        reason = f"{reason} (upstream: {upstream})"
    return reason[:200] or "unknown error"


def _reason(response: httpx.Response) -> str:
    error = _error_block(response)
    return _embedded_reason(error) if error else response.text[:120]


def _is_transient(response: httpx.Response) -> bool:
    """Whether to retry rather than report an outcome.

    Two signals, mirroring the Featherless adapter: the HTTP status, and the error envelope's own
    numeric code — which can disagree with the status when OpenRouter wraps an upstream failure.
    Getting this wrong scores a working worker as a bad routing choice (see ``WorkerBusy``).
    """
    if _transient_code(response.status_code):
        return True
    return _transient_code(_as_int(_error_block(response).get("code")))


#: A leading ``<think>...</think>`` block. Anchored at the start on purpose: a think tag *quoted
#: mid-answer* is content, not a leak.
_THINK_BLOCK = re.compile(r"\s*<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN = re.compile(r"\s*<think>")


def _strip_think(text: str) -> tuple[str, bool]:
    """Strip a leaked reasoning block; report whether one was found.

    Belt and suspenders: ``reasoning.enabled=false`` is sent on every request, but some templates
    leak an inline ``<think>`` block anyway. Left in place it would be scored as the worker's
    answer — a thinking leak reads as incompetence rather than misconfiguration, exactly the
    laundering this codebase refuses elsewhere. The flag is recorded so telemetry can distinguish
    "answered" from "answered after we cut out a leak".
    """
    match = _THINK_BLOCK.match(text)
    if match:
        return text[match.end() :], True
    if _THINK_OPEN.match(text):
        # Opened and never closed: the entire "answer" is reasoning. Nothing usable remains, and
        # pretending otherwise would score reasoning spill as an answer.
        return "", True
    return text, False


def _normalize_provider(name: str) -> str:
    """Fold a provider name to slug-ish form: response provenance is display-cased ("DeepInfra"),
    pins are lowercase slugs ("deepinfra"). A naive equality check would flag every healthy
    response as mispinned."""
    return name.strip().lower().replace(" ", "-")


def _pin_matches(served_by: str, pin: tuple[str, ...]) -> bool:
    """Compare provenance to the pin at *provider* granularity.

    The response names the provider, not the endpoint variant, so a variant pin like
    ``"deepinfra/turbo"`` is verified here as ``deepinfra``; the variant half is enforced
    request-side by ``order`` + ``allow_fallbacks: false``.
    """
    served_base = _normalize_provider(served_by).split("/")[0]
    return any(served_base == _normalize_provider(entry).split("/")[0] for entry in pin)


def _served_by(body: dict) -> str | None:
    """Extract the serving provider from a response body, from either provenance channel.

    The top-level ``provider`` field is the long-standing one; ``openrouter_metadata`` (opt-in via
    the ``X-OpenRouter-Metadata: enabled`` request header, which this adapter always sends) carries
    the selected endpoint. Reading both means a docs-era schema shuffle degrades to the other
    channel instead of blinding the mispin check.
    """
    served = body.get("provider")
    if isinstance(served, str) and served.strip():
        return served.strip()
    metadata = body.get("openrouter_metadata")
    endpoints = metadata.get("endpoints") if isinstance(metadata, dict) else None
    available = endpoints.get("available") if isinstance(endpoints, dict) else None
    for endpoint in available or []:
        if isinstance(endpoint, dict) and endpoint.get("selected"):
            name = endpoint.get("provider")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return None


class OpenRouterWorker:
    """One OpenRouter-routed model, pinned to a specific upstream, over ``/chat/completions``."""

    def __init__(
        self,
        spec: WorkerSpec,
        *,
        pin: str | tuple[str, ...],
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
        think: bool | None = False,
        max_tokens_floor: int = 0,
    ) -> None:
        """Args:
        pin: upstream provider slug(s) this worker is locked to — **required**, no default. The
            manifest carries one per worker. A tuple expresses an explicit preference order; all
            entries count as correctly-pinned provenance. There is no way to construct an
            unpinned worker, because a silently-fallback-routed call is a different served system
            and one such worker poisons a whole calibration run.
        timeout: same 60s ceiling and WorkerBusy-on-timeout semantics as the Featherless adapter —
            a stopwatch must not hand out zeros, and an over-tight cap punishes
            slow-but-correct workers.
        think: ``False`` (default) sends ``reasoning={"enabled": False}``, so the token budget
            buys *answer* rather than reasoning — left on, a reasoning model burns the budget
            thinking and returns a truncated, plausible, **wrong** answer that scores as
            incompetence. ``None`` omits the field: use it for a pinned endpoint that does not
            support reasoning control at all, because ``require_parameters`` would otherwise
            filter that endpoint out of routing entirely.
        max_tokens_floor: raise any lower caller ``max_tokens`` to this. For seats whose
            endpoint reasons regardless of the knob, a generic cap truncates before the answer;
            the floor buys the burn *plus* the answer. 0 (default) = the caller's cap stands.
        """
        cfg = settings()
        key = api_key or cfg.openrouter_api_key
        if not key:
            raise MissingApiKey(
                "OPENROUTER_API_KEY is not set. Put it in .env (see .env.example); "
                "the local Ollama pool and the offline test suite do not need it."
            )
        entries = (pin,) if isinstance(pin, str) else tuple(pin)
        if not entries or any(not (isinstance(e, str) and e.strip()) for e in entries):
            raise MissingPin(
                f"{spec.name}: an OpenRouter worker requires a non-empty upstream pin "
                "(provider slug, e.g. 'deepinfra'). Unpinned routing is a different served "
                "system per request and would poison calibration."
            )
        self.spec = spec
        self.pin: tuple[str, ...] = tuple(e.strip() for e in entries)
        self._api_key = key
        self.base_url = (base_url or cfg.openrouter_base_url).rstrip("/")
        self._client = client
        self._timeout = timeout
        self.think = think
        self.max_tokens_floor = max(0, int(max_tokens_floor))

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            # Opt-in routing provenance (selected endpoint, fallback attempts). Harmless where
            # unsupported; where supported it is the second channel for the mispin check.
            "X-OpenRouter-Metadata": "enabled",
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

        # A token cap must not hand out zeros: a seat whose serving burns budget on reasoning the
        # disable knob fails to stop (baseline 2026-07-31: both qwen3 seats ~1.1k tokens of burn
        # on MATH-hard prompts) truncates before the answer under a generic cap and scores as
        # incompetence. The floor is seat data — the burn is a property of the served system.
        effective_max_tokens = max(max_tokens, self.max_tokens_floor)

        payload: dict = {
            "model": self.spec.model,
            "messages": messages,
            "max_tokens": effective_max_tokens,
            "temperature": temperature,
            "stream": False,
            # The determinism contract, per request: only the pinned upstream(s) may serve, no
            # fallback to whatever else hosts the model, and no provider that would silently drop
            # a parameter (a provider that ignores `temperature` is a different experiment too).
            "provider": {
                "order": list(self.pin),
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        }
        if self.think is not None:
            payload["reasoning"] = {"enabled": self.think}

        with timed() as t:
            try:
                response = await self._post(payload)
                # A pinned endpoint may reject the reasoning knob outright. Retry once without it
                # rather than lose a working worker to an optimisation — but never burn the retry
                # on a transient 400, which must reach the governor as WorkerBusy instead.
                if (
                    response.status_code == 400
                    and "reasoning" in payload
                    and not _is_transient(response)
                ):
                    payload.pop("reasoning")
                    response = await self._post(payload)
            except httpx.TimeoutException as exc:
                # A slow call is a moment, not a verdict. Scoring it zero would teach the policy
                # that routing to a congested-but-correct worker is a routing mistake.
                raise WorkerBusy(
                    f"{self.spec.name}: timed out after {t.elapsed:.0f}s ({type(exc).__name__})"
                ) from exc
            except httpx.RemoteProtocolError as exc:
                # The provider hung up mid-response — a property of the moment, not the request.
                raise WorkerBusy(f"{self.spec.name}: connection dropped ({exc})") from exc
            except httpx.HTTPError as exc:
                return WorkerResult(
                    worker=self.spec.name,
                    text="",
                    ok=False,
                    latency_s=t.elapsed,
                    error=f"transport: {type(exc).__name__}: {exc}",
                )

        # Transient means "ask again", not "this worker is a bad choice" — under GRPO an ordinary
        # failure is scored zero, and zero teaches the policy to avoid a worker that was merely
        # briefly unavailable. 429 is rate limiting; 408 their request timeout; 5xx includes "no
        # provider meeting routing requirements", i.e. the pinned upstream briefly out of capacity.
        if _is_transient(response):
            raise WorkerBusy(f"{self.spec.name}: http {response.status_code} — {_reason(response)}")
        if response.status_code >= 400:
            # Permanent: bad request, bad key, out of credits (402), moderation (403). Retrying
            # wastes budget, so this is an outcome with OpenRouter's own reason attached.
            return WorkerResult(
                worker=self.spec.name,
                text="",
                ok=False,
                latency_s=t.elapsed,
                error=f"http {response.status_code}: {_reason(response)}",
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

        # OpenRouter rides upstream failures inside HTTP 200s: the request reached the router fine,
        # then the provider failed mid-generation. A 200 is therefore not proof of an answer, and
        # the embedded error must be classified exactly like a status would be — an upstream 429 or
        # 5xx retried as WorkerBusy, anything else returned as an outcome.
        embedded = body.get("error") if isinstance(body, dict) else None
        if isinstance(embedded, dict) and embedded:
            reason = _embedded_reason(embedded)
            if _transient_code(_as_int(embedded.get("code"))):
                raise WorkerBusy(f"{self.spec.name}: upstream error inside 200 — {reason}")
            return WorkerResult(
                worker=self.spec.name,
                text="",
                ok=False,
                latency_s=t.elapsed,
                error=f"upstream error inside 200 — {reason}",
            )

        choices = body.get("choices") or []
        text = ""
        finish_reason = None
        reasoning_chars = 0
        if choices:
            message = choices[0].get("message") or {}
            text = message.get("content") or ""
            finish_reason = choices[0].get("finish_reason")
            # Some endpoints ignore reasoning={"enabled": false} and stream their reasoning to a
            # separate message field instead (observed live: qwen3-14b@deepinfra; the 32B on the
            # same host honors the knob). The tokens are billed either way, so the burn is
            # recorded per call — and an empty content alongside a fat reasoning field is a
            # nameable failure, not an "empty response".
            reasoning_chars = len(message.get("reasoning") or "")
        usage = body.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or 0)
        cost_usd = self.spec.cost(tokens_in, tokens_out)

        text, leaked = _strip_think(text)
        served_by = _served_by(body)
        meta = {
            "finish_reason": finish_reason,
            "served_by": served_by,
            "think_leak_stripped": leaked,
            "reasoning_chars": reasoning_chars,
        }

        # Provenance check. A response served by anyone other than the pin is a different served
        # system — different weights-as-quantized, different sampler — so its answer is data from
        # a different experiment and must NOT be scored as this worker's. It fails loudly, naming
        # both parties, with the text zeroed so nothing downstream can score it by accident. Cost
        # and tokens stay: the money was spent, and the ledger reports what was spent, not what we
        # wish had happened. served_by=None (provenance absent) is recorded but not failed — the
        # pin is *enforced* request-side by allow_fallbacks=false; this check is verification.
        if served_by is not None and not _pin_matches(served_by, self.pin):
            return WorkerResult(
                worker=self.spec.name,
                text="",
                ok=False,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_s=t.elapsed,
                cost_usd=cost_usd,
                error=(
                    f"mispinned: served by '{served_by}', pinned to {list(self.pin)} — "
                    "response discarded (a different provider is a different experiment)"
                ),
                meta=meta,
            )

        # Name the failure precisely: "empty response" sends a human hunting the wrong thing when
        # the truth is the model spent its whole turn inside a leaked reasoning block.
        error: str | None = None
        if not text.strip():
            if leaked:
                error = (
                    "think leak consumed the response: reasoning was disabled but the model "
                    "returned only a <think> block"
                )
            elif reasoning_chars:
                error = (
                    f"reasoning consumed the budget: content is empty but {reasoning_chars} chars "
                    "arrived in the reasoning field — this endpoint ignores the disable knob; "
                    "raise max_tokens so the answer fits after the burn"
                )
            else:
                error = "empty response"

        return WorkerResult(
            worker=self.spec.name,
            text=text,
            ok=not error,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=t.elapsed,
            cost_usd=cost_usd,
            error=error,
            meta=meta,
        )


def openrouter_spec(
    name: str,
    model: str,
    *,
    usd_per_mtok_in: float,
    usd_per_mtok_out: float,
    params_b: float | None = None,
    tags: tuple[str, ...] = (),
    units: int = 1,
) -> WorkerSpec:
    """Build a spec for an OpenRouter-served model.

    Prices are **required**, not defaulted: OpenRouter bills per token, and the result's cost
    fields are how the token-spend ledger gets fed. A price that silently defaulted to zero would
    make a paid worker look free in every report — the manifest carries the real per-model prices
    (and the pin) per worker. ``units`` defaults to 1 because OpenRouter meters spend per token
    rather than per concurrent request; raise it only to deliberately serialize a worker.
    """
    return WorkerSpec(
        name=name,
        model=model,
        provider="openrouter",
        params_b=params_b,
        units=units,
        tags=tags,
        usd_per_mtok_in=usd_per_mtok_in,
        usd_per_mtok_out=usd_per_mtok_out,
    )
