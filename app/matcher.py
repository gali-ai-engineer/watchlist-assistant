"""Map an article to the watchlist tickers it is actually about.

Uses three signals, strongest first:
  1. tickers the source itself tagged (Finnhub, EDGAR)
  2. explicit symbol mentions: "$NVDA", "(NASDAQ: NVDA)", "NVDA shares"
  3. company names / well-known aliases with word boundaries

Mentions in the headline count more than mentions in the body, which cuts
down on "Apple" turning up in a story that is really about a supplier.
"""

from __future__ import annotations

import re

from .companies import aliases


def _alias_pattern(alias: str) -> re.Pattern:
    # Case-sensitive for short or ambiguous names ("Apple" the company, not "apple" the fruit)
    flags = 0 if len(alias) <= 6 or alias[0].isupper() else re.I
    return re.compile(rf"(?<![\w$]){re.escape(alias)}(?![\w-])", flags)


def _symbol_pattern(ticker: str) -> re.Pattern:
    t = re.escape(ticker)
    return re.compile(rf"(\${t}\b|\b{t}\s+(?:shares|stock)\b)", re.I)


def _listing_pattern(ticker: str) -> re.Pattern:
    # "(NASDAQ: MSFT)" is how press releases formally identify the issuer: decisive anywhere.
    return re.compile(rf"\((?:NASDAQ|NYSE|NYSEARCA|NYSE American|AMEX)\s*:\s*{re.escape(ticker)}\)", re.I)


def relevance(ticker: str, title: str, body: str, source_tickers: list[str]) -> float:
    """0..1 score of how much the article is about this ticker."""
    t = ticker.upper()
    score = 0.0
    if t in {x.upper() for x in source_tickers}:
        score += 0.5
    if _listing_pattern(t).search(f"{title} {body}"):
        score += 0.6
    sym = _symbol_pattern(t)
    if sym.search(title):
        score += 0.6
    elif sym.search(body):
        score += 0.3
    for alias in aliases(t):
        p = _alias_pattern(alias)
        if p.search(title):
            score += 0.6
            break
        if p.search(body):
            score += 0.25
            break
    return min(score, 1.0)


def match(title: str, body: str, source_tickers: list[str], watchlist: list[str], min_score: float = 0.5) -> list[str]:
    scored = [(t, relevance(t, title, body, source_tickers)) for t in watchlist]
    return [t for t, s in sorted(scored, key=lambda x: -x[1]) if s >= min_score]
