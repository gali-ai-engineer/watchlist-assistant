"""Deduplication: exact (canonical URL) and near-duplicate (same story, different outlet).

A single earnings release shows up on 5-10 sites within minutes. Users want one
alert, so near-duplicates are grouped into a cluster and the extra outlets are
shown as "also reported by" instead of firing new notifications.

Near-duplicate test: only stories about the same ticker within 48 hours are
compared. Text is normalised (synonyms, light stemming) and the score is the
best of headline similarity and lead-paragraph similarity. With an LLM or an
embedding model available this could be upgraded to semantic similarity; the
lexical version is fast, free and explainable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

TRACKING_PARAMS = re.compile(r"^(utm_|fbclid|gclid|mc_|cmpid|ref$|src$|guccounter|tsrc$)")
STOPWORDS = set(
    "a an the and or of to in on for with by at from as is are was were be its it this that will would "
    "after over amid says said say report reports new inc corp co ltd shares stock stocks company "
    "people familiar matter according s than more into about".split()
)
# Map different words outlets use for the same thing onto one token.
SYNONYMS = {
    "investigation": "probe", "investigating": "probe", "inquiry": "probe", "investigates": "probe",
    "european": "eu", "commission": "eu", "regulators": "regulator",
    "tops": "beat", "topping": "beat", "beats": "beat", "exceeds": "beat", "surpasses": "beat",
    "misses": "miss", "falls": "miss",
    "forecasts": "estimate", "expectations": "estimate", "estimates": "estimate",
    "acquire": "buy", "acquisition": "buy", "purchase": "buy", "acquires": "buy",
    "chief": "", "operating": "coo", "officer": "",
    "resigns": "depart", "quits": "depart", "exit": "depart", "step": "depart", "down": "",
    "departure": "depart", "leave": "depart", "leaving": "depart",
}


def canonical_url(url: str) -> str:
    p = urlparse(url.strip())
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    query = [(k, v) for k, v in parse_qsl(p.query) if not TRACKING_PARAMS.match(k.lower())]
    path = re.sub(r"/+$", "", p.path) or "/"
    return urlunparse(("https", host, path, "", urlencode(sorted(query)), ""))


def url_key(url: str) -> str:
    return hashlib.sha1(canonical_url(url).encode()).hexdigest()


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokens(text: str, limit: int | None = None) -> set[str]:
    words = re.findall(r"[a-z0-9$%]+", text.lower())
    if limit:
        words = words[:limit]
    out = set()
    for w in words:
        if w in STOPWORDS:
            continue
        w = SYNONYMS.get(w, w)
        if w:
            out.add(_stem(w))
    return out


def _sim(sa: set[str], sb: set[str]) -> float:
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    # Blend of overlap coefficient (kind to short vs long headlines) and Jaccard (penalises padding).
    return inter / min(len(sa), len(sb)) * 0.6 + inter / len(sa | sb) * 0.4


def similarity(title_a: str, title_b: str, body_a: str = "", body_b: str = "") -> float:
    title = _sim(tokens(title_a), tokens(title_b))
    if body_a and body_b:
        body = _sim(tokens(body_a, 45), tokens(body_b, 45))
        return max(title, body * 0.95)
    return title


@dataclass
class Candidate:
    article_id: str
    title: str
    summary: str
    tickers: list[str]
    published_at: datetime
    cluster_id: str | None = None


def find_duplicate(
    title: str,
    summary: str,
    tickers: list[str],
    published_at: datetime,
    recent: list[Candidate],
    threshold: float,
    window: timedelta = timedelta(hours=48),
) -> tuple[str, str, float] | None:
    """Return (article_id, cluster_id, score) of the closest earlier story above threshold."""
    best: tuple[float, str, str] | None = None
    tick = set(tickers)
    for c in recent:
        if not tick & set(c.tickers):
            continue
        if abs(published_at - c.published_at) > window:
            continue
        s = similarity(title, c.title, summary, c.summary)
        if s >= threshold and (best is None or s > best[0]):
            best = (s, c.article_id, c.cluster_id or c.article_id)
    return (best[1], best[2], round(best[0], 3)) if best else None
