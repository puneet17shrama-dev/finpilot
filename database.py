"""
SQLite persistence for FinPilot (data/portfolio.db).

Design rules
------------
* CREATE TABLE IF NOT EXISTS + additive column migration: an existing
  database is never dropped or rewritten.
* Every table that is written during the 10-second refresh has a natural
  UNIQUE key so repeated refreshes UPSERT instead of inserting duplicates
  (e.g. price snapshots are bucketed to one row per symbol per minute).
* WAL mode + a short busy timeout so the dashboard and the background
  monitor can share the file safely.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterable

import config

_LOCK = threading.RLock()
SCHEMA_VERSION = 1

TABLES: dict[str, str] = {
    "meta": """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )""",
    "holdings_lots": """
        CREATE TABLE IF NOT EXISTS holdings_lots (
            lot_id TEXT PRIMARY KEY,
            account TEXT,
            portfolio_symbol TEXT,
            base_symbol TEXT,
            quantity REAL,
            avg_price REAL,
            invested_value REAL,
            source_row INTEGER,
            source_file TEXT,
            file_signature TEXT,
            imported_at TEXT
        )""",
    "symbol_map": """
        CREATE TABLE IF NOT EXISTS symbol_map (
            base_symbol TEXT PRIMARY KEY,
            portfolio_symbol TEXT,
            yahoo_symbol TEXT,
            status TEXT,
            method TEXT,
            checked_at TEXT
        )""",
    "price_snapshots": """
        CREATE TABLE IF NOT EXISTS price_snapshots (
            yahoo_symbol TEXT,
            bucket TEXT,
            price REAL,
            previous_close REAL,
            open REAL,
            day_high REAL,
            day_low REAL,
            volume REAL,
            data_timestamp TEXT,
            fetched_at TEXT,
            PRIMARY KEY (yahoo_symbol, bucket)
        )""",
    "portfolio_snapshots": """
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            bucket TEXT PRIMARY KEY,
            invested_value REAL,
            current_value REAL,
            total_pnl REAL,
            todays_pnl REAL,
            priced_holdings INTEGER,
            total_holdings INTEGER,
            recorded_at TEXT
        )""",
    "technical_snapshots": """
        CREATE TABLE IF NOT EXISTS technical_snapshots (
            yahoo_symbol TEXT,
            as_of_date TEXT,
            data_json TEXT,
            computed_at TEXT,
            PRIMARY KEY (yahoo_symbol, as_of_date)
        )""",
    "fundamentals": """
        CREATE TABLE IF NOT EXISTS fundamentals (
            yahoo_symbol TEXT PRIMARY KEY,
            data_json TEXT,
            status TEXT,
            fetched_at TEXT
        )""",
    "news": """
        CREATE TABLE IF NOT EXISTS news (
            news_id TEXT PRIMARY KEY,
            yahoo_symbol TEXT,
            headline TEXT,
            publisher TEXT,
            published_at TEXT,
            url TEXT,
            summary TEXT,
            sentiment TEXT,
            category TEXT,
            matched_keywords TEXT,
            first_seen TEXT
        )""",
    "news_fetch_log": """
        CREATE TABLE IF NOT EXISTS news_fetch_log (
            yahoo_symbol TEXT PRIMARY KEY,
            fetched_at TEXT,
            status TEXT
        )""",
    "alert_rules": """
        CREATE TABLE IF NOT EXISTS alert_rules (
            rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
            base_symbol TEXT,
            rule_type TEXT,
            direction TEXT,
            threshold REAL,
            active INTEGER DEFAULT 1,
            note TEXT,
            created_at TEXT
        )""",
    "alerts": """
        CREATE TABLE IF NOT EXISTS alerts (
            alert_key TEXT PRIMARY KEY,
            alert_type TEXT,
            base_symbol TEXT,
            severity TEXT,
            message TEXT,
            created_at TEXT,
            telegram_sent INTEGER DEFAULT 0,
            telegram_sent_at TEXT
        )""",
    "telegram_log": """
        CREATE TABLE IF NOT EXISTS telegram_log (
            log_key TEXT PRIMARY KEY,
            message_type TEXT,
            status TEXT,
            detail TEXT,
            sent_at TEXT
        )""",
    "reports": """
        CREATE TABLE IF NOT EXISTS reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_type TEXT,
            file_path TEXT UNIQUE,
            trigger TEXT,
            created_at TEXT
        )""",
}

# Additive migrations: table -> {column: type}. Missing columns are added,
# nothing is ever dropped, so an older portfolio.db keeps its data.
EXPECTED_COLUMNS: dict[str, dict[str, str]] = {
    "holdings_lots": {"base_symbol": "TEXT", "file_signature": "TEXT", "source_row": "INTEGER"},
    "news": {"sentiment": "TEXT", "category": "TEXT", "matched_keywords": "TEXT"},
    "alerts": {"severity": "TEXT", "telegram_sent_at": "TEXT"},
}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_news_symbol ON news (yahoo_symbol, published_at)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts (created_at)",
    "CREATE INDEX IF NOT EXISTS idx_price_symbol ON price_snapshots (yahoo_symbol, fetched_at)",
]


def now_iso() -> str:
    from modules.market_data import now_ist  # local import avoids a cycle
    return now_ist().isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(config.DB_PATH), timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=15000")
    return conn


@contextmanager
def get_conn():
    with _LOCK:
        conn = _connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


_initialised = False


def init_db() -> None:
    """Create / migrate the schema. Safe to call many times."""
    global _initialised
    if _initialised:
        return
    with get_conn() as conn:
        for ddl in TABLES.values():
            conn.execute(ddl)
        for table, cols in EXPECTED_COLUMNS.items():
            existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            for col, ctype in cols.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ctype}")
        for idx in INDEXES:
            conn.execute(idx)
        conn.execute(
            "INSERT INTO meta(key, value, updated_at) VALUES('schema_version', ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (str(SCHEMA_VERSION), datetime.now().isoformat(timespec="seconds")),
        )
    _initialised = True


def execute(sql: str, params: Iterable[Any] = ()) -> int:
    init_db()
    with get_conn() as conn:
        cur = conn.execute(sql, tuple(params))
        return cur.rowcount


def executemany(sql: str, rows: list[tuple]) -> None:
    if not rows:
        return
    init_db()
    with get_conn() as conn:
        conn.executemany(sql, rows)


def query(sql: str, params: Iterable[Any] = ()) -> list[dict]:
    init_db()
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]


def query_one(sql: str, params: Iterable[Any] = ()) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def get_meta(key: str, default: str | None = None) -> str | None:
    row = query_one("SELECT value FROM meta WHERE key=?", (key,))
    return row["value"] if row else default


def set_meta(key: str, value: Any) -> None:
    execute(
        "INSERT INTO meta(key, value, updated_at) VALUES(?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value if isinstance(value, str) else json.dumps(value), now_iso()),
    )


def claim_once(log_key: str, message_type: str, detail: str = "") -> bool:
    """Atomically claim a one-time event (e.g. today's closing report).

    Returns True only for the first caller, so two browser tabs or a
    dashboard + headless monitor can never send the same scheduled report twice.
    """
    init_db()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO telegram_log(log_key, message_type, status, detail, sent_at) "
            "VALUES(?, ?, 'claimed', ?, ?)",
            (log_key, message_type, detail, now_iso()),
        )
        return cur.rowcount == 1


def update_log(log_key: str, status: str, detail: str = "") -> None:
    execute(
        "UPDATE telegram_log SET status=?, detail=?, sent_at=? WHERE log_key=?",
        (status, detail[:1000], now_iso(), log_key),
    )
