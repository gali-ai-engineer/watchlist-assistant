"""Alert policy: decide which enriched articles notify the user now, and which wait for a digest.

The hard part of this product isn't finding news, it's not spamming people.
Rules here:
  - only trusted sources ever alert
  - impact >= ALERT_MIN_IMPACT is pushed immediately; the rest goes to a digest
  - per-ticker rate limit on pushes so one busy news day doesn't flood the phone
  - a duplicate of a story already alerted never alerts again
"""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from .models import Alert, Article

MAX_PUSH_PER_TICKER_PER_HOUR = 3


class AlertPolicy:
    def __init__(self, min_impact: int, min_trust: float, max_per_hour: int = MAX_PUSH_PER_TICKER_PER_HOUR):
        self.min_impact = min_impact
        self.min_trust = min_trust
        self.max_per_hour = max_per_hour
        self._recent_push: dict[str, deque[datetime]] = defaultdict(deque)

    @staticmethod
    def score(a: Article) -> float:
        """0..10 ranking score: impact weighted by source trust and sentiment strength."""
        e = a.enrichment
        if not e:
            return 0.0
        return round(e.impact * 1.6 * (0.6 + 0.4 * a.source_trust) + abs(e.sentiment) * 1.5, 2)

    def _rate_ok(self, ticker: str, now: datetime) -> bool:
        q = self._recent_push[ticker]
        while q and now - q[0] > timedelta(hours=1):
            q.popleft()
        return len(q) < self.max_per_hour

    def decide(self, a: Article, now: datetime | None = None) -> list[Alert]:
        now = now or datetime.now(timezone.utc)
        if a.enrichment is None or a.source_trust < self.min_trust:
            return []
        s = self.score(a)
        alerts = []
        for t in a.tickers:
            priority = "push" if a.enrichment.impact >= self.min_impact else "digest"
            if priority == "push" and not self._rate_ok(t, now):
                priority = "digest"
            if priority == "push":
                self._recent_push[t].append(now)
            alert_id = hashlib.sha1(f"{a.id}:{t}".encode()).hexdigest()[:16]
            alerts.append(Alert(id=alert_id, article_id=a.id, ticker=t, priority=priority, score=s, created_at=now))
        return alerts
