"""SQLite persistence. Small and dependency-free; swap for Postgres in production."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .dedupe import Candidate
from .models import Alert, Article, Enrichment, EventType

SCHEMA = """
CREATE TABLE IF NOT EXISTS watchlist (
    ticker TEXT PRIMARY KEY,
    added_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    cluster_id TEXT,
    published_at TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL            -- full Article as JSON
);
CREATE INDEX IF NOT EXISTS idx_articles_published ON articles(published_at DESC);
CREATE TABLE IF NOT EXISTS article_tickers (
    article_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    PRIMARY KEY (article_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_at_ticker ON article_tickers(ticker);
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    article_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    priority TEXT NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL,
    channels TEXT NOT NULL DEFAULT '[]',
    read INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);
CREATE TABLE IF NOT EXISTS seen_urls (
    url_key TEXT PRIMARY KEY,
    seen_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def article_from_json(raw: str) -> Article:
    d = json.loads(raw)
    enr = d.get("enrichment")
    enrichment = None
    if enr:
        enrichment = Enrichment(
            summary=enr["summary"],
            sentiment=enr["sentiment"],
            event_type=EventType(enr["event_type"]),
            impact=enr["impact"],
            why_it_matters=enr["why_it_matters"],
            model=enr.get("model", "heuristic"),
        )
    return Article(
        id=d["id"],
        source_name=d["source_name"],
        source_domain=d["source_domain"],
        source_trust=d["source_trust"],
        url=d["url"],
        title=d["title"],
        summary=d["summary"],
        published_at=datetime.fromisoformat(d["published_at"]),
        tickers=d["tickers"],
        provider=d["provider"],
        enrichment=enrichment,
        cluster_id=d.get("cluster_id"),
        also_reported_by=d.get("also_reported_by", []),
    )


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # ---- watchlist -------------------------------------------------------
    def get_watchlist(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT ticker FROM watchlist ORDER BY added_at").fetchall()
        return [r["ticker"] for r in rows]

    def add_ticker(self, ticker: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO watchlist(ticker, added_at) VALUES (?, ?)",
                (ticker.upper(), _now()),
            )
            self._conn.commit()

    def remove_ticker(self, ticker: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker.upper(),))
            self._conn.commit()

    # ---- dedupe memory -----------------------------------------------------
    def seen(self, url_key: str) -> bool:
        with self._lock:
            row = self._conn.execute("SELECT 1 FROM seen_urls WHERE url_key = ?", (url_key,)).fetchone()
        return row is not None

    def mark_seen(self, url_key: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO seen_urls(url_key, seen_at) VALUES (?, ?)", (url_key, _now())
            )
            self._conn.commit()

    # ---- articles ----------------------------------------------------------
    def save_article(self, a: Article) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO articles(id, cluster_id, published_at, title, body) VALUES (?,?,?,?,?)",
                (a.id, a.cluster_id, a.published_at.isoformat(), a.title, json.dumps(a.to_dict())),
            )
            self._conn.executemany(
                "INSERT OR IGNORE INTO article_tickers(article_id, ticker) VALUES (?, ?)",
                [(a.id, t) for t in a.tickers],
            )
            self._conn.commit()

    def get_article(self, article_id: str) -> Article | None:
        with self._lock:
            row = self._conn.execute("SELECT body FROM articles WHERE id = ?", (article_id,)).fetchone()
        return article_from_json(row["body"]) if row else None

    def recent_articles(self, tickers: list[str] | None = None, limit: int = 200) -> list[Article]:
        with self._lock:
            if tickers:
                marks = ",".join("?" * len(tickers))
                rows = self._conn.execute(
                    f"""SELECT DISTINCT a.body, a.published_at FROM articles a
                        JOIN article_tickers t ON t.article_id = a.id
                        WHERE t.ticker IN ({marks})
                        ORDER BY a.published_at DESC LIMIT ?""",
                    (*[t.upper() for t in tickers], limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT body FROM articles ORDER BY published_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [article_from_json(r["body"]) for r in rows]

    def dedupe_candidates(self, limit: int = 500) -> list[Candidate]:
        """Recent stories for near-duplicate detection."""
        return [
            Candidate(a.id, a.title, a.summary, a.tickers, a.published_at, a.cluster_id)
            for a in self.recent_articles(limit=limit)
        ]

    def add_also_reported_by(self, article_id: str, source_name: str) -> None:
        art = self.get_article(article_id)
        if art and source_name not in art.also_reported_by and source_name != art.source_name:
            art.also_reported_by.append(source_name)
            self.save_article(art)

    # ---- alerts ------------------------------------------------------------
    def save_alert(self, al: Alert) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO alerts(id, article_id, ticker, priority, score, created_at, channels) VALUES (?,?,?,?,?,?,?)",
                (al.id, al.article_id, al.ticker, al.priority, al.score, al.created_at.isoformat(), json.dumps(al.channels)),
            )
            self._conn.commit()

    def list_alerts(self, limit: int = 100, ticker: str | None = None) -> list[dict]:
        with self._lock:
            q = """SELECT al.*, a.body FROM alerts al JOIN articles a ON a.id = al.article_id"""
            args: tuple = ()
            if ticker:
                q += " WHERE al.ticker = ?"
                args = (ticker.upper(),)
            q += " ORDER BY al.created_at DESC LIMIT ?"
            rows = self._conn.execute(q, (*args, limit)).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "ticker": r["ticker"],
                    "priority": r["priority"],
                    "score": r["score"],
                    "created_at": r["created_at"],
                    "channels": json.loads(r["channels"]),
                    "read": bool(r["read"]),
                    "article": json.loads(r["body"]),
                }
            )
        return out

    def mark_alert_read(self, alert_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE alerts SET read = 1 WHERE id = ?", (alert_id,))
            self._conn.commit()

    def counts(self) -> dict:
        with self._lock:
            c = self._conn.execute
            return {
                "articles": c("SELECT COUNT(*) FROM articles").fetchone()[0],
                "alerts_push": c("SELECT COUNT(*) FROM alerts WHERE priority='push'").fetchone()[0],
                "alerts_digest": c("SELECT COUNT(*) FROM alerts WHERE priority='digest'").fetchone()[0],
                "unread": c("SELECT COUNT(*) FROM alerts WHERE read=0").fetchone()[0],
            }

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def reset(self) -> None:
        with self._lock:
            for t in ("articles", "article_tickers", "alerts", "seen_urls"):
                self._conn.execute(f"DELETE FROM {t}")
            self._conn.commit()
