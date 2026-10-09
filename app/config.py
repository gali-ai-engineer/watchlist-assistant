"""Settings, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader so we don't need python-dotenv. Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


@dataclass
class Settings:
    # Mode: "demo" replays bundled sample news offline; "live" calls real sources.
    mode: str = "demo"
    db_path: str = str(ROOT / "data" / "assistant.db")

    # LLM: "auto" picks the first provider with a key, else the offline heuristic.
    llm_provider: str = "auto"  # auto | openai | anthropic | gemini | heuristic
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    llm_timeout_s: float = 20.0

    # News sources (live mode)
    finnhub_api_key: str = ""
    enable_yahoo_rss: bool = True
    enable_sec_edgar: bool = True
    sec_user_agent: str = "WatchlistAssistant demo contact@example.com"
    lookback_hours: int = 24

    # Pipeline
    poll_interval_s: int = 300  # live mode
    demo_interval_s: float = 6.0  # demo mode: one "breaking" item every N seconds
    min_source_trust: float = 0.6
    alert_min_impact: int = 3  # 1-5; below this goes to the digest, not a push
    dedupe_threshold: float = 0.45

    # Notifications (optional)
    slack_webhook_url: str = ""
    discord_webhook_url: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    default_watchlist: list[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
    )

    @property
    def is_demo(self) -> bool:
        return self.mode.lower() != "live"


def load_settings(env_file: Path | None = None) -> Settings:
    _load_dotenv(env_file or ROOT / ".env")
    return Settings(
        mode=os.getenv("APP_MODE", "demo"),
        db_path=os.getenv("DB_PATH", str(ROOT / "data" / "assistant.db")),
        llm_provider=os.getenv("LLM_PROVIDER", "auto").lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5"),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        llm_timeout_s=float(os.getenv("LLM_TIMEOUT_S", "20")),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY", ""),
        enable_yahoo_rss=_bool("ENABLE_YAHOO_RSS", True),
        enable_sec_edgar=_bool("ENABLE_SEC_EDGAR", True),
        sec_user_agent=os.getenv("SEC_USER_AGENT", "WatchlistAssistant demo contact@example.com"),
        lookback_hours=int(os.getenv("LOOKBACK_HOURS", "24")),
        poll_interval_s=int(os.getenv("POLL_INTERVAL_S", "300")),
        demo_interval_s=float(os.getenv("DEMO_INTERVAL_S", "6")),
        min_source_trust=float(os.getenv("MIN_SOURCE_TRUST", "0.6")),
        alert_min_impact=int(os.getenv("ALERT_MIN_IMPACT", "3")),
        dedupe_threshold=float(os.getenv("DEDUPE_THRESHOLD", "0.45")),
        slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL", ""),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        default_watchlist=_list("DEFAULT_WATCHLIST", ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]),
    )
