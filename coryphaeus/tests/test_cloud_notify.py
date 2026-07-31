"""Notifications, offline.

The property under test is mostly negative: ``send()`` must never raise. The launcher calls it
on the same exit paths that call ``terminate()``, so an exception escaping a notification would
turn "Telegram was down while the run finished" into "the GPU is still billing". The rest is the
factory's env contract — a public clone with no Telegram configured must lose nothing.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from coryphaeus.cloud.notify import Notifier, NullNotifier, TelegramNotifier, make_notifier

TOKEN = "123456:TEST-not-a-real-token"
CHAT = "424242"
LOGGER = "coryphaeus.cloud.notify"


def _telegram(handler) -> tuple[TelegramNotifier, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TelegramNotifier(TOKEN, CHAT, client=client), client


def _clear_env(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)


# --- the factory's env contract ---------------------------------------------------------------


def test_factory_without_env_is_null(monkeypatch):
    _clear_env(monkeypatch)
    assert isinstance(make_notifier(), NullNotifier)


def test_factory_with_env_is_telegram(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    assert isinstance(make_notifier(), TelegramNotifier)


@pytest.mark.parametrize("present", ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"])
def test_factory_half_configured_is_null_and_says_so(monkeypatch, caplog, present):
    """A typo'd variable name degrades to silence WITH a pointer, not to a 3am exception."""
    _clear_env(monkeypatch)
    monkeypatch.setenv(present, "something")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        notifier = make_notifier()
    assert isinstance(notifier, NullNotifier)
    assert "notifications are off" in caplog.text


def test_direct_construction_without_config_refuses(monkeypatch):
    """The factory degrades; the class itself refuses — half a notifier is not a notifier."""
    _clear_env(monkeypatch)
    with pytest.raises(ValueError, match="make_notifier"):
        TelegramNotifier(token=TOKEN, chat_id=None)


def test_both_implementations_satisfy_the_protocol():
    assert isinstance(NullNotifier(), Notifier)
    assert isinstance(TelegramNotifier(TOKEN, CHAT), Notifier)


# --- the wire shape ---------------------------------------------------------------------------


async def test_telegram_request_shape():
    """Token in the URL path, chat id and text in the JSON body — the Bot API's contract."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.read())
        return httpx.Response(200, json={"ok": True})

    notifier, client = _telegram(handler)
    async with client:
        await notifier.send("grpo15b-r5: finished, checkpoint saved, instance terminated")
    assert seen["method"] == "POST"
    assert seen["url"] == f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    assert seen["payload"]["chat_id"] == CHAT
    assert seen["payload"]["text"].startswith("grpo15b-r5")


# --- the never-raise contract -----------------------------------------------------------------


async def test_a_500_is_logged_not_raised(caplog):
    body = {"ok": False, "error_code": 500, "description": "Internal Server Error"}
    notifier, client = _telegram(lambda r: httpx.Response(500, json=body))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        async with client:
            await notifier.send("done")  # an exception here would unwind the terminate path
    assert "http 500" in caplog.text
    assert "Internal Server Error" in caplog.text


async def test_a_timeout_is_logged_not_raised(caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    notifier, client = _telegram(handler)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        async with client:
            await notifier.send("done")
    assert "ReadTimeout" in caplog.text


async def test_the_token_never_reaches_a_log_line(caplog):
    """The URL embeds the bot token and httpx error strings can embed the URL. Public repo:
    a failed notification must not launder the credential into a log file."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed to reach {request.url}")

    notifier, client = _telegram(handler)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        async with client:
            await notifier.send("done")
    assert "ConnectError" in caplog.text  # the failure is visible…
    assert TOKEN not in caplog.text  # …the credential is not


# --- the null default -------------------------------------------------------------------------


async def test_null_notifier_accepts_anything():
    notifier = NullNotifier()
    for text in ("", "done", "x" * 10_000, "unicode — ✓ π", "line\nbreaks\ttabs"):
        await notifier.send(text)
