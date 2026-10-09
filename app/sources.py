"""News source connectors. Each returns RawArticle objects for a set of tickers.

Live sources:
  - FinnhubSource: company news API (free tier: 60 calls/min), source-tagged tickers
  - YahooRSSSource: per-ticker headline RSS, no key needed
  - SECEdgarSource: 8-K current reports straight from EDGAR (primary source)
Offline:
  - DemoSource: replays bundled sample items, dripping "breaking" ones over time
"""

from __future__ import annotations

import html
import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Protocol

from . import http
from .companies import cik, company_name
from .config import ROOT, Settings
from .models import RawArticle

log = logging.getLogger(__name__)


class Source(Protocol):
    name: str

    def fetch(self, tickers: list[str], since: datetime) -> list[RawArticle]: ...


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


class FinnhubSource:
    name = "finnhub"
    BASE = "https://finnhub.io/api/v1/company-news"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch(self, tickers: list[str], since: datetime) -> list[RawArticle]:
        out: list[RawArticle] = []
        today = datetime.now(timezone.utc).date()
        for t in tickers:
            url = f"{self.BASE}?symbol={t}&from={since.date()}&to={today}&token={self.api_key}"
            try:
                items = http.get_json(url, timeout=15)
            except http.HttpError as e:
                log.warning("finnhub %s failed: %s", t, e)
                continue
            for it in items or []:
                ts = datetime.fromtimestamp(it.get("datetime", 0), tz=timezone.utc)
                if ts < since or not it.get("url") or not it.get("headline"):
                    continue
                related = [x.strip().upper() for x in (it.get("related") or "").split(",") if x.strip()]
                out.append(
                    RawArticle(
                        source_name=it.get("source", ""),
                        url=it["url"],
                        title=it["headline"],
                        summary=it.get("summary", ""),
                        published_at=ts,
                        source_tickers=related or [t],
                        provider=self.name,
                    )
                )
        return out


class YahooRSSSource:
    name = "yahoo_rss"
    BASE = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"

    def fetch(self, tickers: list[str], since: datetime) -> list[RawArticle]:
        out: list[RawArticle] = []
        for t in tickers:
            try:
                xml = http.request(self.BASE.format(t=t), timeout=15)
                root = ET.fromstring(xml)
            except (http.HttpError, ET.ParseError) as e:
                log.warning("yahoo rss %s failed: %s", t, e)
                continue
            for item in root.iter("item"):
                link = (item.findtext("link") or "").strip()
                title = (item.findtext("title") or "").strip()
                try:
                    ts = parsedate_to_datetime(item.findtext("pubDate") or "")
                except (TypeError, ValueError):
                    continue
                if not link or not title or ts < since:
                    continue
                out.append(
                    RawArticle(
                        source_name="Yahoo Finance",
                        url=link,
                        title=title,
                        summary=_strip_html(item.findtext("description") or ""),
                        published_at=ts,
                        source_tickers=[],  # RSS query is per ticker but items can be loosely related; let the matcher decide
                        provider=self.name,
                    )
                )
        return out


# 8-K item codes -> plain English (https://www.sec.gov/fast-answers/answersform8khtm.html)
EIGHT_K_ITEMS = {
    "1.01": "entered a material agreement",
    "1.02": "terminated a material agreement",
    "2.01": "completed an acquisition or disposition of assets",
    "2.02": "reported results of operations (earnings)",
    "2.05": "announced restructuring or exit costs",
    "2.06": "recorded a material impairment",
    "5.02": "reported a departure or appointment of directors or officers",
    "5.07": "reported shareholder vote results",
    "7.01": "made a Regulation FD disclosure",
    "8.01": "reported other material events",
}


class SECEdgarSource:
    name = "sec_edgar"
    BASE = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=8-K&dateb=&owner=include&count=10&output=atom"
    NS = {"a": "http://www.w3.org/2005/Atom"}

    def __init__(self, user_agent: str):
        # SEC requires a descriptive User-Agent with contact details.
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}

    def fetch(self, tickers: list[str], since: datetime) -> list[RawArticle]:
        out: list[RawArticle] = []
        for t in tickers:
            c = cik(t)
            if not c:
                continue
            try:
                root = ET.fromstring(http.request(self.BASE.format(cik=c), headers=self.headers, timeout=15))
            except (http.HttpError, ET.ParseError) as e:
                log.warning("edgar %s failed: %s", t, e)
                continue
            for e in root.findall("a:entry", self.NS):
                link_el = e.find("a:link", self.NS)
                updated = e.findtext("a:updated", default="", namespaces=self.NS)
                try:
                    ts = datetime.fromisoformat(updated)
                except ValueError:
                    continue
                if link_el is None or ts < since:
                    continue
                summary = _strip_html(e.findtext("a:summary", default="", namespaces=self.NS))
                items = re.findall(r"\b(\d\.\d\d)\b", summary)
                described = [EIGHT_K_ITEMS[i] for i in items if i in EIGHT_K_ITEMS]
                what = "; ".join(dict.fromkeys(described)) or "filed a current report"
                out.append(
                    RawArticle(
                        source_name="SEC EDGAR",
                        url=link_el.get("href", ""),
                        title=f"{company_name(t)} files 8-K: {what}",
                        summary=f"Form 8-K filed with the SEC. Items: {', '.join(items) or 'n/a'}. {summary[:300]}",
                        published_at=ts,
                        source_tickers=[t],
                        provider=self.name,
                    )
                )
        return out


class DemoSource:
    """Replays data/demo_news.json. Items marked "breaking" arrive one at a time."""

    name = "demo"

    def __init__(self, path: Path | None = None):
        self.path = path or ROOT / "data" / "demo_news.json"
        data = json.loads(self.path.read_text())
        self.history = [d for d in data["items"] if not d.get("breaking")]
        self.breaking = [d for d in data["items"] if d.get("breaking")]
        self._history_sent = False
        self._next = 0

    @property
    def remaining(self) -> int:
        return len(self.breaking) - self._next

    def rewind(self) -> None:
        self._history_sent = False
        self._next = 0

    def _to_raw(self, d: dict, ts: datetime) -> RawArticle:
        return RawArticle(
            source_name=d["source"],
            url=d["url"],
            title=d["title"],
            summary=d["summary"],
            published_at=ts,
            source_tickers=d.get("tickers", []),
            provider=self.name,
        )

    def backfill(self, tickers: list[str], since: datetime) -> list[RawArticle]:
        """History plus breaking items already released, for a newly added ticker."""
        now = datetime.now(timezone.utc)
        items = [self._to_raw(d, now - timedelta(minutes=d.get("minutes_ago", 60))) for d in self.history]
        items += [self._to_raw(d, now - timedelta(minutes=1)) for d in self.breaking[: self._next]]
        return items

    def fetch(self, tickers: list[str], since: datetime) -> list[RawArticle]:
        now = datetime.now(timezone.utc)
        if not self._history_sent:
            self._history_sent = True
            return [self._to_raw(d, now - timedelta(minutes=d.get("minutes_ago", 60))) for d in self.history]
        if self._next < len(self.breaking):
            d = self.breaking[self._next]
            self._next += 1
            return [self._to_raw(d, now)]
        return []


def build_sources(s: Settings) -> list[Source]:
    if s.is_demo:
        return [DemoSource()]
    sources: list[Source] = []
    if s.finnhub_api_key:
        sources.append(FinnhubSource(s.finnhub_api_key))
    if s.enable_yahoo_rss:
        sources.append(YahooRSSSource())
    if s.enable_sec_edgar:
        sources.append(SECEdgarSource(s.sec_user_agent))
    if not sources:
        log.warning("Live mode with no sources enabled; nothing will be fetched")
    return sources
