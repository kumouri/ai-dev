"""Run notifications — how a launcher tells a human the box stopped billing.

A cloud run ends hours after anyone stopped watching, and every ending is one of two messages:
"your checkpoint is ready" or "you are still paying". Both deserve a push. Telegram is the
transport because it is free, needs exactly one HTTP POST, and — the property that matters in a
public repo — is *entirely optional*: without ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` in
the environment, :func:`make_notifier` hands back a :class:`NullNotifier` and nothing else
changes. Nobody's private daemon plumbing is assumed (see the package docstring).

The one structural rule: **a notification failure must never kill a training run.** ``send()`` is
called on the launcher's exit paths — the same paths that call ``terminate()`` — so an exception
escaping it would convert "Telegram was down while the run finished" into "the GPU is still
billing". Every failure is logged and swallowed; there is deliberately no way to make this raise.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)

#: Generous for one small POST, but finite: ``send()`` sits on the very exit path it reports on,
#: and a notification hanging forever in front of ``terminate()`` would be the billing leak this
#: module exists to prevent.
DEFAULT_TIMEOUT_S = 30.0


@runtime_checkable
class Notifier(Protocol):
    """What the launcher needs: fire-and-forget text. No result, no raise."""

    async def send(self, text: str) -> None:
        """Deliver ``text`` if possible. Failures are the implementation's to log, never raise."""
        ...


class NullNotifier:
    """The unconfigured default: accepts anything, sends nothing, changes nothing.

    Exists so callers never branch on "is notification configured?" — a public clone without
    Telegram runs the exact same launcher code paths as a configured one.
    """

    async def send(self, text: str) -> None:
        return None


def _description(response: httpx.Response) -> str:
    """Telegram's own reason out of an error body — safe to log, unlike the URL.

    Mirrors the Featherless adapter's ``_reason``: a bare "http 400" leaves a human guessing,
    where "chat not found" (typo'd chat id) and "Unauthorized" (revoked token) each say exactly
    what to fix.
    """
    try:
        body = response.json()
    except ValueError:
        return ""
    return str(body.get("description") or "")[:200]


class TelegramNotifier:
    """Sends via the Bot API's ``sendMessage`` — one endpoint, nothing beyond httpx.

    ``token`` and ``chat_id`` default from ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID``, but
    prefer :func:`make_notifier`, which degrades to :class:`NullNotifier` instead of raising
    when they are absent. ``client`` injection exists for tests (a ``MockTransport`` needs no
    network); without it each send opens a short-lived client.
    """

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self._token = (token or os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
        self._chat_id = (chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")).strip()
        if not self._token or not self._chat_id:
            raise ValueError(
                "TelegramNotifier needs both TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID; "
                "use make_notifier() to fall back to NullNotifier instead."
            )
        self._client = client
        self._timeout = timeout

    async def send(self, text: str) -> None:
        """POST ``text`` to the configured chat. Never raises — see the module docstring."""
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {"chat_id": self._chat_id, "text": text}
        try:
            if self._client is not None:
                response = await self._client.post(url, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload)
        except Exception as exc:
            # Broad on purpose: the contract is "never raise", not "never raise httpx errors".
            # Only the exception TYPE is logged — httpx error strings can embed the request URL,
            # and this URL embeds the bot token, which must never reach a log line.
            logger.warning("notification not sent (%s)", type(exc).__name__)
            return
        if response.status_code >= 400:
            reason = _description(response)
            logger.warning(
                "notification not sent (http %s%s)",
                response.status_code,
                f" — {reason}" if reason else "",
            )


def make_notifier() -> Notifier:
    """Telegram when the env is fully configured, Null otherwise.

    Reads the process environment; ``.env`` reaches it through ``coryphaeus.config.load_dotenv``
    the moment any entry point touches settings. Half a configuration counts as none, but says
    so: a typo'd variable name should degrade to silence with a pointer at the typo, not to an
    exception at 3am.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if token and chat_id:
        return TelegramNotifier(token, chat_id)
    if token or chat_id:
        missing = "TELEGRAM_CHAT_ID" if token else "TELEGRAM_BOT_TOKEN"
        present = "TELEGRAM_BOT_TOKEN" if token else "TELEGRAM_CHAT_ID"
        logger.warning("%s is set but %s is not — notifications are off", present, missing)
    return NullNotifier()
