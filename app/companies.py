"""Reference data: ticker -> company name and the names news uses for it.

In production this comes from a security master (e.g. Finnhub /stock/profile2
or OpenFIGI). A static map keeps the PoC self-contained; unknown tickers still
work through the source's own ticker tags and "$TICKER"/"(NASDAQ: X)" mentions.
"""

from __future__ import annotations

COMPANIES: dict[str, dict] = {
    "AAPL": {"name": "Apple Inc.", "aliases": ["Apple", "iPhone", "Tim Cook"], "cik": "0000320193"},
    "MSFT": {"name": "Microsoft Corp.", "aliases": ["Microsoft", "Azure", "Satya Nadella"], "cik": "0000789019"},
    "NVDA": {"name": "NVIDIA Corp.", "aliases": ["Nvidia", "NVIDIA", "Jensen Huang"], "cik": "0001045810"},
    "AMZN": {"name": "Amazon.com Inc.", "aliases": ["Amazon", "AWS", "Andy Jassy"], "cik": "0001018724"},
    "GOOGL": {"name": "Alphabet Inc.", "aliases": ["Alphabet", "Google", "YouTube", "Sundar Pichai", "GOOG"], "cik": "0001652044"},
    "META": {"name": "Meta Platforms Inc.", "aliases": ["Meta Platforms", "Facebook", "Instagram", "WhatsApp", "Mark Zuckerberg"], "cik": "0001326801"},
    "TSLA": {"name": "Tesla Inc.", "aliases": ["Tesla", "Elon Musk"], "cik": "0001318605"},
    "AMD": {"name": "Advanced Micro Devices", "aliases": ["Advanced Micro Devices", "Lisa Su"], "cik": "0000002488"},
    "NFLX": {"name": "Netflix Inc.", "aliases": ["Netflix"], "cik": "0001065280"},
    "JPM": {"name": "JPMorgan Chase & Co.", "aliases": ["JPMorgan", "JP Morgan", "Jamie Dimon"], "cik": "0000019617"},
    "AVGO": {"name": "Broadcom Inc.", "aliases": ["Broadcom"], "cik": "0001730168"},
    "ORCL": {"name": "Oracle Corp.", "aliases": ["Oracle"], "cik": "0001341439"},
    "INTC": {"name": "Intel Corp.", "aliases": ["Intel"], "cik": "0000050863"},
    "TSM": {"name": "Taiwan Semiconductor", "aliases": ["TSMC", "Taiwan Semiconductor"], "cik": "0001046179"},
    "V": {"name": "Visa Inc.", "aliases": ["Visa Inc"], "cik": "0001403161"},
}


def company_name(ticker: str) -> str:
    return COMPANIES.get(ticker.upper(), {}).get("name", ticker.upper())


def aliases(ticker: str) -> list[str]:
    info = COMPANIES.get(ticker.upper())
    return [info["name"].split(" Inc")[0].split(" Corp")[0], *info["aliases"]] if info else []


def cik(ticker: str) -> str | None:
    return COMPANIES.get(ticker.upper(), {}).get("cik")
