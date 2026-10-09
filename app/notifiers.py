"""Delivery channels. In-app (server-sent events) always; Slack, Discord and Telegram if configured.

Mobile push (FCM/APNs) and email would slot in as further Notifier classes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Protocol

from . import http
from .config import Settings

log = logging.getLogger(__name__)


def format_alert(alert: dict) -> str:
    a = alert["article"]
    e = a.get("enrichment") or {}
    sent = e.get("sentiment", 0)
    arrow = "▲" if sent > 0.15 else "▼" if sent < -0.15 else "•"
    return (
        f"{arrow} {alert['ticker']} | {a['title']}\n"
        f"{e.get('summary', '')}\n"
        f"Why it matters: {e.get('why_it_matters', '')}\n"
        f"{a['source_name']} | {a['url']}"
    )


class Notifier(Protocol):
    name: str

    def send(self, alert: dict) -> None: ...


class SlackNotifier:
    name = "slack"

    def __init__(self, url: str):
        self.url = url

    def send(self, alert: dict) -> None:
        http.post_json(self.url, {"text": format_alert(alert)}, timeout=10, retries=1)


class DiscordNotifier:
    name = "discord"

    def __init__(self, url: str):
        self.url = url

    def send(self, alert: dict) -> None:
        http.request(self.url, method="POST", payload={"content": format_alert(alert)[:1900]}, timeout=10, retries=1)


class TelegramNotifier:
    name = "telegram"

    def __init__(self, token: str, chat_id: str):
        self.url = f"https://api.telegram.org/bot{token}/sendMessage"
        self.chat_id = chat_id

    def send(self, alert: dict) -> None:
        http.post_json(
            self.url,
            {"chat_id": self.chat_id, "text": format_alert(alert), "disable_web_page_preview": True},
            timeout=10,
            retries=1,
        )


def build_notifiers(s: Settings) -> list[Notifier]:
    out: list[Notifier] = []
    if s.slack_webhook_url:
        out.append(SlackNotifier(s.slack_webhook_url))
    if s.discord_webhook_url:
        out.append(DiscordNotifier(s.discord_webhook_url))
    if s.telegram_bot_token and s.telegram_chat_id:
        out.append(TelegramNotifier(s.telegram_bot_token, s.telegram_chat_id))
    return out


class Broadcaster:
    """Fan-out of events to every open browser tab via server-sent events."""

    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    def publish(self, event: str, data: dict) -> None:
        msg = f"event: {event}\ndata: {json.dumps(data)}\n\n"
        for q in list(self._subs):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:  # slow client; drop rather than block the pipeline
                pass

    @property
    def subscribers(self) -> int:
        return len(self._subs)
