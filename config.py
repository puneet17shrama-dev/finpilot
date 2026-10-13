"""
FinPilot central configuration.

All settings can be overridden from the .env file (see .env.example).
Secrets (Telegram bot token, optional LLM key) are ONLY read from .env and
are never hard-coded here.

FinPilot is a monitoring / analysis tool. It never places orders and never
connects to a broker. The single market-data source is Yahoo Finance (yfinance).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is in requirements.txt; degrade gracefully
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on", "y"}


def _env_list(name: str) -> list[str]:
    return [x.strip() for x in _env(name).split(",") if x.strip()]


# ------------------------------------------------------------------ project
PROJECT_NAME = _env("PROJECT_NAME", "FinPilot")
PROJECT_TAGLINE = "Personal Portfolio Intelligence"
TIMEZONE = _env("TIMEZONE", "Asia/Kolkata")
CURRENCY_SYMBOL = _env("CURRENCY_SYMBOL", "₹")

# ------------------------------------------------------------------ paths
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
# ONE authoritative portfolio file. Every component (dashboard, Telegram, PDF, alerts,
# scanner, risk, agent, sell analysis) reads this file through portfolio.build_snapshot().
# No other file (e.g. a holdings.csv lying around) is ever discovered or merged implicitly.
HOLDINGS_FILE = Path(_env("HOLDINGS_FILE", "data/holdings.xlsx"))
if not HOLDINGS_FILE.is_absolute():
    HOLDINGS_FILE = BASE_DIR / HOLDINGS_FILE
# A CSV can be used only when explicitly configured AND allowed – it never overrides the XLSX silently.
ALLOW_CSV_HOLDINGS = _env_bool("ALLOW_CSV_HOLDINGS", False)
DB_PATH = Path(_env("DB_PATH", str(DATA_DIR / "portfolio.db")))
if not DB_PATH.is_absolute():
    DB_PATH = BASE_DIR / DB_PATH

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ refresh
MARKET_REFRESH_SECONDS = max(5, _env_int("MARKET_REFRESH_SECONDS", 10))
DASHBOARD_URL = _env("DASHBOARD_URL", "http://localhost:8501")

# ------------------------------------------------------------------ Yahoo Finance (the ONLY market-data source)
MARKET_DATA_PROVIDER = "Yahoo Finance (yfinance)"
# Exchange suffixes tried, in order, when resolving an Indian symbol.
# Each candidate is VERIFIED against Yahoo before being accepted.
YAHOO_SUFFIXES = _env_list("YAHOO_SUFFIXES") or [".NS", ".BO"]
# Manual overrides:  PORTFOLIO_SYMBOL=YAHOO_SYMBOL, comma separated
# e.g. SYMBOL_OVERRIDES=GMRP&UI=GMRP&UI.NS,TTML=TTML.NS
SYMBOL_OVERRIDES: dict[str, str] = {}
for _pair in _env_list("SYMBOL_OVERRIDES"):
    if "=" in _pair:
        _k, _v = _pair.split("=", 1)
        SYMBOL_OVERRIDES[_k.strip().upper()] = _v.strip()
BENCHMARK_SYMBOL = _env("BENCHMARK_SYMBOL", "^NSEI")  # Nifty 50 on Yahoo
HISTORY_PERIOD = _env("HISTORY_PERIOD", "1y")
HISTORY_CACHE_MINUTES = _env_int("HISTORY_CACHE_MINUTES", 360)        # daily bars
FUNDAMENTALS_CACHE_HOURS = _env_int("FUNDAMENTALS_CACHE_HOURS", 24)
NEWS_CACHE_MINUTES = _env_int("NEWS_CACHE_MINUTES", 30)
SYMBOL_RECHECK_HOURS = _env_int("SYMBOL_RECHECK_HOURS", 24)           # retry unresolved symbols
YAHOO_TIMEOUT_SECONDS = _env_int("YAHOO_TIMEOUT_SECONDS", 20)
# A quote older than this (while the market is open) is shown as "Stale".
STALE_AFTER_SECONDS = _env_int("STALE_AFTER_SECONDS", 180)

# ------------------------------------------------------------------ market session (NSE, IST)
MARKET_OPEN_TIME = _env("MARKET_OPEN_TIME", "09:15")
MARKET_CLOSE_TIME = _env("MARKET_CLOSE_TIME", "15:30")
PRE_OPEN_TIME = _env("PRE_OPEN_TIME", "09:00")
# Exchange holidays (YYYY-MM-DD, comma separated). Keep this updated from the
# official NSE holiday calendar – FinPilot does not guess holidays.
# Official NSE Capital-Market (equities) trading holidays for 2026, from NSE's holiday master
# (nseindia.com → Resources → Exchange Communication → Holidays), checked 30 Sep 2026.
# Weekend entries are listed by NSE too; they are harmless here. Add later years via
# MARKET_HOLIDAYS in .env (comma separated YYYY-MM-DD) – those dates are ADDED to this list.
NSE_HOLIDAYS = {
    "2026-01-15": "Municipal Corporation Election - Maharashtra",
    "2026-01-26": "Republic Day",
    "2026-02-15": "Mahashivratri",
    "2026-03-03": "Holi",
    "2026-03-21": "Id-Ul-Fitr (Ramadan Eid)",
    "2026-03-26": "Shri Ram Navami",
    "2026-03-31": "Shri Mahavir Jayanti",
    "2026-04-03": "Good Friday",
    "2026-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
    "2026-05-01": "Maharashtra Day",
    "2026-05-28": "Bakri Id",
    "2026-06-26": "Muharram",
    "2026-08-15": "Independence Day",
    "2026-09-14": "Ganesh Chaturthi",
    "2026-10-02": "Mahatma Gandhi Jayanti",
    "2026-10-20": "Dussehra",
    "2026-11-08": "Diwali Laxmi Pujan (Muhurat trading session announced separately)",
    "2026-11-10": "Diwali-Balipratipada",
    "2026-11-24": "Prakash Gurpurb Sri Guru Nanak Dev",
    "2026-12-25": "Christmas",
}
MARKET_HOLIDAYS = set(NSE_HOLIDAYS) | set(_env_list("MARKET_HOLIDAYS"))
HOLIDAY_YEARS = sorted({d[:4] for d in MARKET_HOLIDAYS})

# ------------------------------------------------------------------ Telegram (notifications only – never trading)
# SECURITY: the bot token is read ONLY from .env (never hard-code it, never log it).
# If the token has ever been pasted into chat, source code, screenshots or logs, revoke it in
# Telegram @BotFather (/revoke) and put the new token in .env.
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = _env("TELEGRAM_CHAT_ID")
TELEGRAM_ENABLED = _env_bool("TELEGRAM_ENABLED", True) and bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
TELEGRAM_SEND_STARTUP = _env_bool("TELEGRAM_SEND_STARTUP", True)
TELEGRAM_OPEN_REPORT_TIME = _env("TELEGRAM_OPEN_REPORT_TIME", "09:20")   # sent once per trading day
TELEGRAM_OPEN_REPORT_UNTIL = _env("TELEGRAM_OPEN_REPORT_UNTIL", "11:00")
TELEGRAM_CLOSE_REPORT_TIME = _env("TELEGRAM_CLOSE_REPORT_TIME", "15:40")
TELEGRAM_CLOSE_REPORT_UNTIL = _env("TELEGRAM_CLOSE_REPORT_UNTIL", "20:00")
TELEGRAM_MAX_ALERTS_PER_CYCLE = _env_int("TELEGRAM_MAX_ALERTS_PER_CYCLE", 5)
# Up to four portfolio symbols for the short performance report (user-configured, never assumed)
REPORT_STOCKS = [s.upper() for s in _env_list("REPORT_STOCKS")][:4]
SCHEDULED_PDF_AT_CLOSE = _env_bool("SCHEDULED_PDF_AT_CLOSE", False)

# ------------------------------------------------------------------ alerts (deduplicated)
ALERT_DAY_MOVE_PCT = _env_float("ALERT_DAY_MOVE_PCT", 5.0)        # stock moves ±X% vs previous close
ALERT_PNL_MOVE_PCT = _env_float("ALERT_PNL_MOVE_PCT", 10.0)       # position return crosses ±X% steps
ALERT_VOLUME_RATIO = _env_float("ALERT_VOLUME_RATIO", 2.5)        # today's volume ÷ 20-day average
ALERT_PORTFOLIO_MOVE_PCT = _env_float("ALERT_PORTFOLIO_MOVE_PCT", 2.0)  # portfolio day move
ALERT_IMPORTANT_NEWS = _env_bool("ALERT_IMPORTANT_NEWS", True)

# ------------------------------------------------------------------ optional LLM for the agent
# The agent answers from calculated data with built-in rules. An LLM is only
# used (optionally) to phrase answers for questions the rules don't cover,
# and it is given ONLY FinPilot's calculated data.
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")
LLM_MODEL = _env("LLM_MODEL")
LLM_ENABLED = bool(ANTHROPIC_API_KEY and LLM_MODEL)

# ------------------------------------------------------------------ logging
LOG_FILE = DATA_DIR / "finpilot.log"
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=getattr(logging, _env("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()])
# urllib3/requests debug logs would include the Telegram API URL (which contains the token)
logging.getLogger("urllib3").setLevel(logging.WARNING)

DISCLAIMER = (
    "FinPilot is an analysis and monitoring tool, not investment advice. "
    "It never places, modifies or executes trades. Market data: Yahoo Finance "
    "(may be delayed). Figures are estimates."
)
