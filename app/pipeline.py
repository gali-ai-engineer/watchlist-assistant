"""The ingestion pipeline: fetch -> exact dedupe -> trust filter -> ticker match
-> near-duplicate clustering -> LLM enrichment -> alert policy -> store.

run_once() is synchronous and side-effect free apart from the store, which
keeps it easy to test and to move into a queue worker (Celery, SQS, Cloud
Tasks) later. Delivery happens in the async runner in main.py.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from . import dedupe, matcher, trust
from .alerts import AlertPolicy
from .config import Settings
from .enrich import enrich
from .llm import LLM
from .models import Article, RawArticle
from .sources import Source
from .storage import Store

log = logging.getLogger(__name__)


@dataclass
class RunReport:
    started_at: str
    duration_ms: int
    counts: dict
    new_alerts: list[dict] = field(default_factory=list)


class Pipeline:
    def __init__(self, settings: Settings, store: Store, sources: list[Source], llm: LLM | None):
        self.s = settings
        self.store = store
        self.sources = sources
        self.llm = llm
        self.policy = AlertPolicy(settings.alert_min_impact, settings.min_source_trust)
        self.totals: Counter = Counter()
        self.last_report: RunReport | None = None

    def process(self, raw: RawArticle, watchlist: list[str], c: Counter) -> list[dict]:
        c["fetched"] += 1
        key = dedupe.url_key(raw.url)
        if self.store.seen(key):
            c["dup_url"] += 1
            return []

        domain, source_name, trust_score = trust.score(raw.url, raw.source_name)
        if trust_score < self.s.min_source_trust:
            self.store.mark_seen(key)  # an untrusted source never becomes trusted; skip for good
            c["untrusted"] += 1
            log.debug("drop untrusted %s: %s", domain, raw.title)
            return []

        tickers = matcher.match(raw.title, raw.summary, raw.source_tickers, watchlist)
        if not tickers:
            # Not marked seen: if the user adds this ticker later, a backfill can still pick it up.
            c["unmatched"] += 1
            return []
        self.store.mark_seen(key)

        dup = dedupe.find_duplicate(
            raw.title, raw.summary, tickers, raw.published_at,
            self.store.dedupe_candidates(), self.s.dedupe_threshold,
        )
        if dup:
            original_id, _cluster, sim = dup
            self.store.add_also_reported_by(original_id, source_name)
            c["dup_story"] += 1
            log.debug("near-duplicate (%.2f) of %s: %s", sim, original_id, raw.title)
            return []

        enrichment = enrich(self.llm, raw.title, raw.summary, tickers, source_name, trust_score)
        c["enriched"] += 1
        if enrichment.model != "heuristic":
            c["llm_calls"] += 1

        article = Article(
            id=raw.id,
            source_name=source_name,
            source_domain=domain,
            source_trust=trust_score,
            url=raw.url,
            title=raw.title,
            summary=raw.summary,
            published_at=raw.published_at,
            tickers=tickers,
            provider=raw.provider,
            enrichment=enrichment,
            cluster_id=raw.id,
        )
        self.store.save_article(article)

        out = []
        for al in self.policy.decide(article):
            self.store.save_alert(al)
            c[f"alert_{al.priority}"] += 1
            out.append(
                {
                    "id": al.id,
                    "ticker": al.ticker,
                    "priority": al.priority,
                    "score": al.score,
                    "created_at": al.created_at.isoformat(),
                    "channels": [],
                    "read": False,
                    "article": article.to_dict(),
                }
            )
        return out

    def run_once(self) -> RunReport:
        return self._run(self.store.get_watchlist(), backfill=False)

    def backfill(self, tickers: list[str]) -> RunReport:
        """Pull recent news for newly added tickers without waiting for the next poll."""
        return self._run([t.upper() for t in tickers], backfill=True)

    def _run(self, watchlist: list[str], backfill: bool) -> RunReport:
        t0 = time.perf_counter()
        started = datetime.now(timezone.utc)
        since = started - timedelta(hours=self.s.lookback_hours)
        c: Counter = Counter()
        new_alerts: list[dict] = []
        if watchlist:
            for src in self.sources:
                try:
                    fetch = getattr(src, "backfill", None) if backfill else None
                    raws = fetch(watchlist, since) if fetch else src.fetch(watchlist, since)
                except Exception:  # one broken source must not stop the others
                    log.exception("source %s failed", src.name)
                    c["source_errors"] += 1
                    continue
                for raw in sorted(raws, key=lambda r: r.published_at):
                    try:
                        new_alerts.extend(self.process(raw, watchlist, c))
                    except Exception:
                        log.exception("failed to process %s", raw.url)
                        c["item_errors"] += 1
        if backfill:
            # A backfill re-reads stories the funnel already counted; only record what changed:
            # items that were "unmatched" before and now matched the new ticker.
            moved = c["enriched"] + c["dup_story"]
            self.totals["unmatched"] = max(0, self.totals["unmatched"] - moved)
            self.totals.update({k: v for k, v in c.items() if k not in {"fetched", "dup_url", "unmatched"}})
        else:
            self.totals.update(c)
        report = RunReport(
            started_at=started.isoformat(),
            duration_ms=int((time.perf_counter() - t0) * 1000),
            counts=dict(c),
            new_alerts=new_alerts,
        )
        self.last_report = report
        if c:
            log.info("pipeline run: %s in %dms", dict(c), report.duration_ms)
        return report
