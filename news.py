"""
Portfolio news from Yahoo Finance (yfinance Ticker.news).

* Fetched per portfolio symbol at most every NEWS_CACHE_MINUTES.
* Stored in SQLite keyed by the Yahoo article id / URL → no duplicates,
  no matter how often the dashboard refreshes.
* Classified with transparent keyword rules (the matched words are stored
  and shown). Keyword tone is a description of the headline wording only; it
  does not predict how a stock will move.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

import config
from modules import database as db
from modules.market_data import TZ, now_ist, yf

log = logging.getLogger("finpilot.news")

IMPORTANT_KEYWORDS = [
    "earnings", "results", "quarterly", "q1", "q2", "q3", "q4", "dividend", "bonus", "split",
    "buyback", "merger", "acquisition", "acquire", "stake", "resignation", "resigns", "appointment",
    "appoints", "ceo", "cfo", "regulatory", "sebi", "rbi", "court", "tribunal", "nclt", "order",
    "contract", "guidance", "bankruptcy", "insolvency", "investigation", "probe", "raid", "default",
    "rating", "block deal", "bulk deal", "fund raise", "qip", "rights issue", "ipo", "delisting",
]
POSITIVE_KEYWORDS = [
    "surge", "surges", "jump", "jumps", "soar", "soars", "rally", "rallies", "gain", "gains", "rise",
    "rises", "up ", "record high", "beats", "beat estimates", "profit rises", "profit jumps", "growth",
    "upgrade", "upgraded", "wins", "bags", "secures", "order win", "expansion", "approval", "approved",
    "strong", "outperform", "higher", "positive", "boost",
]
NEGATIVE_KEYWORDS = [
    "fall", "falls", "drop", "drops", "plunge", "plunges", "slump", "slumps", "crash", "tumble",
    "decline", "declines", "down ", "loss", "losses", "miss", "misses", "downgrade", "downgraded",
    "weak", "lower", "penalty", "fine", "fraud", "probe", "raid", "default", "resigns", "cut",
    "warning", "negative", "concern", "slowdown", "lawsuit", "ban",
]


def _match(text: str, words: list[str]) -> list[str]:
    t = f" {text.lower()} "
    hits = []
    for w in words:
        pattern = r"(?<![a-z])" + re.escape(w.strip()) + r"(?![a-z])"
        if re.search(pattern, t):
            hits.append(w.strip())
    return sorted(set(hits))


def classify(headline: str, summary: str = "") -> dict:
    """Transparent rule-based classification.

    category  : 'Important Event' if any important-event keyword matched, else 'General'
    sentiment : 'Positive' / 'Negative' by keyword count, 'Neutral' when tied or none
    """
    text = f"{headline or ''} {summary or ''}"
    imp, pos, neg = _match(text, IMPORTANT_KEYWORDS), _match(text, POSITIVE_KEYWORDS), _match(text, NEGATIVE_KEYWORDS)
    sentiment = "Positive" if len(pos) > len(neg) else "Negative" if len(neg) > len(pos) else "Neutral"
    return {
        "category": "Important Event" if imp else "General",
        "sentiment": sentiment,
        "matched": {"important": imp, "positive": pos, "negative": neg},
    }


def _parse_item(item: dict, symbol: str) -> dict | None:
    """Support both the current yfinance news format ({'id','content':{...}}) and the legacy one."""
    c = item.get("content") if isinstance(item.get("content"), dict) else None
    if c:
        title = c.get("title")
        url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
        publisher = (c.get("provider") or {}).get("displayName")
        summary = c.get("summary") or c.get("description")
        pub = c.get("pubDate") or c.get("displayTime")
        published = None
        if pub:
            try:
                published = datetime.fromisoformat(pub.replace("Z", "+00:00")).astimezone(TZ)
            except ValueError:
                published = None
        nid = item.get("id") or c.get("id")
    else:
        title = item.get("title")
        url = item.get("link")
        publisher = item.get("publisher")
        summary = item.get("summary")
        ts = item.get("providerPublishTime")
        published = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(TZ) if ts else None
        nid = item.get("uuid")
    if not title:
        return None
    nid = nid or hashlib.sha1(f"{url or title}".encode()).hexdigest()
    cls = classify(title, summary or "")
    return {
        "news_id": str(nid), "yahoo_symbol": symbol, "headline": title.strip(),
        "publisher": publisher or "N/A", "published_at": published.isoformat() if published else None,
        "url": url, "summary": summary, **cls,
    }


def refresh_news(yahoo_symbol: str, force: bool = False) -> dict:
    """Fetch Yahoo news for one symbol if the cache is older than NEWS_CACHE_MINUTES.
    Returns {'status', 'new_items': int, 'error'}."""
    row = db.query_one("SELECT * FROM news_fetch_log WHERE yahoo_symbol=?", (yahoo_symbol,))
    if row and not force:
        age = now_ist() - datetime.fromisoformat(row["fetched_at"])
        if age < timedelta(minutes=config.NEWS_CACHE_MINUTES):
            return {"status": "cached", "new_items": 0, "error": None}
    if yf is None:
        return {"status": "error", "new_items": 0, "error": "yfinance is not installed"}
    try:
        items = yf.Ticker(yahoo_symbol).news or []
        if not isinstance(items, list):
            raise ValueError(f"Unexpected Yahoo news response type: {type(items).__name__}")
        parsed = []
        for item in items:
            try:
                p = _parse_item(item, yahoo_symbol) if isinstance(item, dict) else None
            except Exception as exc:  # one malformed article must not drop the rest
                log.warning("Skipping malformed Yahoo news item for %s: %s", yahoo_symbol, exc)
                p = None
            if p:
                parsed.append(p)
        new = 0
        with db.get_conn() as conn:
            for p in parsed:
                # dedupe by id and by URL/headline (same story syndicated with a different id)
                dup = conn.execute("SELECT 1 FROM news WHERE news_id=? OR (url IS NOT NULL AND url=?) "
                                   "OR (headline=? AND yahoo_symbol=?)",
                                   (p["news_id"], p["url"], p["headline"], yahoo_symbol)).fetchone()
                if dup:
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO news(news_id, yahoo_symbol, headline, publisher, published_at, url, "
                    "summary, sentiment, category, matched_keywords, first_seen) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (p["news_id"], yahoo_symbol, p["headline"], p["publisher"], p["published_at"], p["url"],
                     p["summary"], p["sentiment"], p["category"], _fmt_matched(p["matched"]), db.now_iso()))
                new += 1
            conn.execute("INSERT INTO news_fetch_log(yahoo_symbol, fetched_at, status) VALUES(?,?,?) "
                         "ON CONFLICT(yahoo_symbol) DO UPDATE SET fetched_at=excluded.fetched_at, status=excluded.status",
                         (yahoo_symbol, db.now_iso(), f"ok:{len(parsed)}"))
        return {"status": "ok", "new_items": new, "error": None}
    except Exception as exc:
        log.warning("Yahoo news fetch failed for %s: %s", yahoo_symbol, exc)
        db.execute("INSERT INTO news_fetch_log(yahoo_symbol, fetched_at, status) VALUES(?,?,?) "
                   "ON CONFLICT(yahoo_symbol) DO UPDATE SET fetched_at=excluded.fetched_at, status=excluded.status",
                   (yahoo_symbol, (now_ist() - timedelta(minutes=max(0, config.NEWS_CACHE_MINUTES - 5))).isoformat(
                       timespec="seconds"), f"error:{str(exc)[:200]}"))
        return {"status": "error", "new_items": 0, "error": str(exc)}


def _fmt_matched(m: dict) -> str:
    parts = [f"{k}: {', '.join(v)}" for k, v in m.items() if v]
    return "; ".join(parts)


def stale_symbols(yahoo_symbols: list[str]) -> list[str]:
    if not yahoo_symbols:
        return []
    rows = {r["yahoo_symbol"]: r for r in db.query("SELECT * FROM news_fetch_log")}
    cutoff = now_ist() - timedelta(minutes=config.NEWS_CACHE_MINUTES)
    return [s for s in yahoo_symbols
            if s not in rows or datetime.fromisoformat(rows[s]["fetched_at"]) < cutoff]


NEWS_COLUMNS = ["news_id", "yahoo_symbol", "headline", "publisher", "published_at", "url", "summary",
                "sentiment", "category", "matched_keywords", "first_seen"]


def get_news(yahoo_symbols: list[str] | None = None, days: int = 14, category: str | None = None,
             limit: int = 200) -> pd.DataFrame:
    sql = "SELECT * FROM news WHERE 1=1"
    params: list = []
    if yahoo_symbols:
        sql += f" AND yahoo_symbol IN ({','.join('?' * len(yahoo_symbols))})"
        params += list(yahoo_symbols)
    if days:
        cutoff = (now_ist() - timedelta(days=days)).isoformat()
        sql += " AND (published_at >= ? OR (published_at IS NULL AND first_seen >= ?))"
        params += [cutoff, cutoff]
    if category:
        sql += " AND category=?"
        params.append(category)
    sql += " ORDER BY COALESCE(published_at, first_seen) DESC LIMIT ?"
    params.append(limit)
    try:
        return pd.DataFrame(db.query(sql, params), columns=NEWS_COLUMNS)
    except Exception:
        log.exception("Reading stored news failed")
        return pd.DataFrame(columns=NEWS_COLUMNS)
