"""Source credibility. Only news from outlets above a trust threshold reaches users.

Scores are a product decision, kept in one place so compliance/editorial can
own them. Primary sources (regulatory filings, company IR) rank highest.
"""

from __future__ import annotations

from urllib.parse import urlparse

# domain -> (display name, trust 0..1)
SOURCES: dict[str, tuple[str, float]] = {
    # Primary sources
    "sec.gov": ("SEC EDGAR", 1.0),
    "investor.apple.com": ("Apple IR", 0.95),
    "businesswire.com": ("Business Wire", 0.85),
    "prnewswire.com": ("PR Newswire", 0.8),
    "globenewswire.com": ("GlobeNewswire", 0.8),
    # Tier-1 newswires and financial press
    "reuters.com": ("Reuters", 0.95),
    "bloomberg.com": ("Bloomberg", 0.95),
    "apnews.com": ("AP", 0.9),
    "wsj.com": ("The Wall Street Journal", 0.9),
    "ft.com": ("Financial Times", 0.9),
    "cnbc.com": ("CNBC", 0.85),
    "barrons.com": ("Barron's", 0.85),
    "marketwatch.com": ("MarketWatch", 0.8),
    "nytimes.com": ("The New York Times", 0.85),
    # Aggregators and analysis (useful, lower weight)
    "finance.yahoo.com": ("Yahoo Finance", 0.7),
    "investors.com": ("Investor's Business Daily", 0.7),
    "fool.com": ("The Motley Fool", 0.5),
    "seekingalpha.com": ("Seeking Alpha", 0.5),
    "benzinga.com": ("Benzinga", 0.6),
    "zacks.com": ("Zacks", 0.5),
}

# Finnhub reports a "source" name rather than always a canonical URL.
NAME_TO_DOMAIN = {name.lower(): dom for dom, (name, _) in SOURCES.items()}
NAME_TO_DOMAIN.update({"yahoo": "finance.yahoo.com", "sec": "sec.gov", "dow jones": "wsj.com"})

UNKNOWN_TRUST = 0.3


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def score(url: str, source_name: str = "") -> tuple[str, str, float]:
    """Return (domain, display_name, trust) for an article."""
    dom = domain_of(url)
    # Match exact domain or a parent domain (e.g. "uk.reuters.com" -> "reuters.com")
    parts = dom.split(".")
    for i in range(len(parts) - 1):
        cand = ".".join(parts[i:])
        if cand in SOURCES:
            name, trust = SOURCES[cand]
            return cand, name, trust
    # Fall back to the name the provider reported
    mapped = NAME_TO_DOMAIN.get(source_name.strip().lower())
    if mapped:
        name, trust = SOURCES[mapped]
        return mapped, name, trust
    return dom, source_name or dom, UNKNOWN_TRUST
