"""Core data types passed between pipeline stages."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum


class EventType(str, Enum):
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    M_AND_A = "m&a"
    REGULATORY = "regulatory"
    LEGAL = "legal"
    PRODUCT = "product"
    LEADERSHIP = "leadership"
    ANALYST = "analyst"
    FILING = "filing"
    PARTNERSHIP = "partnership"
    MACRO = "macro"
    OTHER = "other"


@dataclass
class RawArticle:
    """An item as it comes out of a source, before any processing."""

    source_name: str  # e.g. "Reuters"
    url: str
    title: str
    summary: str
    published_at: datetime
    source_tickers: list[str] = field(default_factory=list)  # tickers the source tagged
    provider: str = ""  # which connector fetched it: finnhub | yahoo_rss | sec_edgar | demo

    @property
    def id(self) -> str:
        return hashlib.sha1(self.url.encode()).hexdigest()[:16]


@dataclass
class Enrichment:
    summary: str
    sentiment: float  # -1 (very negative) .. +1 (very positive)
    event_type: EventType
    impact: int  # 1 (noise) .. 5 (material, likely to move the stock)
    why_it_matters: str
    model: str = "heuristic"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d


@dataclass
class Article:
    """A cleaned, deduplicated, ticker-matched and enriched news item."""

    id: str
    source_name: str
    source_domain: str
    source_trust: float
    url: str
    title: str
    summary: str
    published_at: datetime
    tickers: list[str]
    provider: str
    enrichment: Enrichment | None = None
    cluster_id: str | None = None  # same story from several outlets shares a cluster
    also_reported_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_name": self.source_name,
            "source_domain": self.source_domain,
            "source_trust": self.source_trust,
            "url": self.url,
            "title": self.title,
            "summary": self.summary,
            "published_at": self.published_at.isoformat(),
            "tickers": self.tickers,
            "provider": self.provider,
            "enrichment": self.enrichment.to_dict() if self.enrichment else None,
            "cluster_id": self.cluster_id,
            "also_reported_by": self.also_reported_by,
        }


@dataclass
class Alert:
    id: str
    article_id: str
    ticker: str
    priority: str  # "push" (sent now) | "digest" (batched)
    score: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    channels: list[str] = field(default_factory=list)
