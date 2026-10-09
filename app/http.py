"""Minimal HTTP helpers on the standard library, with retry and backoff."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


def request(
    url: str,
    *,
    method: str = "GET",
    payload: Any | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    retries: int = 2,
) -> bytes:
    data = json.dumps(payload).encode() if payload is not None else None
    hdrs = {"User-Agent": "WatchlistAssistant/0.1", **(headers or {})}
    if data is not None:
        hdrs.setdefault("Content-Type", "application/json")

    attempt = 0
    while True:
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code in RETRY_STATUS and attempt < retries:
                wait = float(e.headers.get("Retry-After") or 2**attempt)
                log.warning("HTTP %s from %s, retrying in %.1fs", e.code, url.split("?")[0], wait)
                time.sleep(min(wait, 10))
                attempt += 1
                continue
            raise HttpError(e.code, body) from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < retries:
                time.sleep(2**attempt)
                attempt += 1
                continue
            raise HttpError(0, str(e)) from None


def get_json(url: str, **kw: Any) -> Any:
    return json.loads(request(url, **kw))


def post_json(url: str, payload: Any, **kw: Any) -> Any:
    raw = request(url, method="POST", payload=payload, **kw)
    return json.loads(raw) if raw else None
