"""
Market data – Yahoo Finance (yfinance) ONLY.

No other provider is used, and no value is ever invented. When Yahoo fails:
  * the last good value is shown as "Stale" (with its real timestamp), or
  * the value is shown as "Unavailable",
and the request is retried on the next refresh.

Caching (module-level, thread-safe, shared by the Streamlit dashboard and the
background monitor in the same process):
  * live quotes ............ MARKET_REFRESH_SECONDS  (one batch request for all symbols)
  * daily history (1y) ..... HISTORY_CACHE_MINUTES   (one batch request, used by technicals/risk)
  * symbol resolution ...... persisted in SQLite, re-checked after SYMBOL_RECHECK_HOURS

Timestamps: `data_timestamp` is the time of the last Yahoo bar. It is NOT the
local clock. The dashboard shows both, clearly labelled.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo

import pandas as pd

import config
from modules import database as db

try:
    import yfinance as yf
except ImportError:  # shown as a data error in the UI instead of crashing
    yf = None

logging.getLogger("yfinance").setLevel(logging.CRITICAL)
log = logging.getLogger("finpilot.market_data")

TZ = ZoneInfo(config.TIMEZONE)


# ================================================================ clock & session
def now_ist() -> datetime:
    return datetime.now(TZ)


def _parse_hhmm(text: str) -> dtime:
    h, m = text.split(":")
    return dtime(int(h), int(m))


def is_trading_day(d: datetime) -> bool:
    return d.weekday() < 5 and d.strftime("%Y-%m-%d") not in config.MARKET_HOLIDAYS


def market_status(now: datetime | None = None) -> dict:
    """Schedule-based NSE session status from the local IST clock.

    This is derived from exchange hours (and MARKET_HOLIDAYS), not from Yahoo.
    """
    now = now or now_ist()
    t = now.time()
    open_t, close_t, pre_t = (_parse_hhmm(config.MARKET_OPEN_TIME),
                              _parse_hhmm(config.MARKET_CLOSE_TIME),
                              _parse_hhmm(config.PRE_OPEN_TIME))
    if now.weekday() >= 5:
        status, detail = "CLOSED", "Weekend"
    elif now.strftime("%Y-%m-%d") in config.MARKET_HOLIDAYS:
        name = config.NSE_HOLIDAYS.get(now.strftime("%Y-%m-%d"))
        status, detail = "CLOSED", f"Exchange holiday{': ' + name if name else ''}"
    elif pre_t <= t < open_t:
        status, detail = "PRE-OPEN", f"Opens {config.MARKET_OPEN_TIME} IST"
    elif open_t <= t < close_t:
        status, detail = "OPEN", f"Closes {config.MARKET_CLOSE_TIME} IST"
    elif t < pre_t:
        status, detail = "CLOSED", f"Opens {config.MARKET_OPEN_TIME} IST"
    else:
        status, detail = "CLOSED", "Session ended"
    return {"status": status, "detail": detail, "is_open": status == "OPEN",
            "trading_day": is_trading_day(now), "as_of": now}


_HOLIDAY_WARNED: set[str] = set()


def holiday_calendar_status(now: datetime | None = None) -> dict:
    """Is the exchange-holiday calendar loaded for the current year? (FinPilot never guesses holidays.)"""
    now = now or now_ist()
    year = now.strftime("%Y")
    covered = year in config.HOLIDAY_YEARS
    if not covered and year not in _HOLIDAY_WARNED:
        _HOLIDAY_WARNED.add(year)
        log.warning("No exchange holidays configured for %s – add them to MARKET_HOLIDAYS in .env", year)
    return {"year": year, "covered": covered, "years": config.HOLIDAY_YEARS,
            "count_this_year": sum(d.startswith(year) for d in config.MARKET_HOLIDAYS)}


def last_completed_session(now: datetime | None = None):
    """Date of the most recent trading session that has fully closed."""
    now = now or now_ist()
    d = now
    if not (is_trading_day(d) and d.time() >= _parse_hhmm(config.MARKET_CLOSE_TIME)):
        d = d - timedelta(days=1)
    for _ in range(15):
        if is_trading_day(d):
            return d.date()
        d = d - timedelta(days=1)
    return None


def stored_quote_status(data_ts: datetime | None, now: datetime | None = None) -> str:
    """Status for a price restored from the local database (or kept after a failed refresh).

    'Stored close' – market is closed and the stored bar is from the last completed session,
                     i.e. that session's last traded price (valid, ranked in top movers).
    'Stale'        – anything else (market open, or an older session) – flagged to the user."""
    now = now or now_ist()
    if data_ts is None or market_status(now)["is_open"]:
        return "Stale"
    ts = data_ts.astimezone(TZ) if data_ts.tzinfo else data_ts.replace(tzinfo=TZ)
    # Any bar from the last completed session is that session's last traded price (illiquid
    # stocks often have no trade in the final minutes), i.e. what Yahoo would return right now.
    if ts.date() == last_completed_session(now) and ts.time() >= _parse_hhmm(config.MARKET_OPEN_TIME):
        return "Stored close"
    return "Stale"


def fmt_time(dt: datetime | None, with_seconds: bool = True) -> str:
    if dt is None:
        return "—"
    dt = dt.astimezone(TZ) if dt.tzinfo else dt.replace(tzinfo=TZ)
    return dt.strftime("%I:%M:%S %p IST" if with_seconds else "%I:%M %p IST")


def fmt_datetime(dt: datetime | None) -> str:
    if dt is None:
        return "—"
    dt = dt.astimezone(TZ) if dt.tzinfo else dt.replace(tzinfo=TZ)
    return dt.strftime("%d %b %Y, %I:%M:%S %p IST")


# ================================================================ connection status
class _Status:
    def __init__(self):
        self.lock = threading.RLock()
        self.last_attempt: datetime | None = None
        self.last_success: datetime | None = None
        self.last_error: str | None = None
        self.consecutive_failures = 0
        self.last_ok_count = 0            # symbols priced in the last quote refresh
        self.last_failed_count = 0        # symbols with no fresh Yahoo price in the last refresh
        self.failed_symbols: list[str] = []

    def ok(self):
        with self.lock:
            self.last_attempt = self.last_success = now_ist()
            self.last_error, self.consecutive_failures = None, 0

    def fail(self, err: str):
        with self.lock:
            self.last_attempt = now_ist()
            self.last_error = err
            self.consecutive_failures += 1


STATUS = _Status()


def data_status() -> dict:
    with STATUS.lock:
        if yf is None:
            state = "ERROR"
        elif STATUS.last_attempt is None:
            state = "CONNECTING"
        elif STATUS.consecutive_failures == 0:
            state = "CONNECTED"
        elif STATUS.last_success is not None:
            state = "STALE"
        else:
            state = "ERROR"
        nxt = None
        if _QUOTE_CACHE["fetched_at"]:
            nxt = max(0, config.MARKET_REFRESH_SECONDS - (time.time() - _QUOTE_CACHE["fetched_at"]))
        return {
            "state": state,
            "provider": config.MARKET_DATA_PROVIDER,
            "last_attempt": STATUS.last_attempt,
            "last_success": STATUS.last_success,
            "last_error": "yfinance is not installed" if yf is None else STATUS.last_error,
            "failures": STATUS.consecutive_failures,
            "last_ok_count": STATUS.last_ok_count,
            "last_failed_count": STATUS.last_failed_count,
            "failed_symbols": list(STATUS.failed_symbols),
            "next_refresh_in": nxt,
        }


# ================================================================ helpers
def _split_download(df: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Split a yf.download() result into {ticker: OHLCV frame} for any column layout."""
    out: dict[str, pd.DataFrame] = {}
    if df is None or df.empty:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        lvl1 = set(df.columns.get_level_values(1))
        for t in tickers:
            if t in lvl0:
                sub = df[t]
            elif t in lvl1:
                sub = df.xs(t, axis=1, level=1)
            else:
                continue
            sub = sub.dropna(how="all")
            if not sub.empty and "Close" in sub and sub["Close"].notna().any():
                out[t] = sub
    elif len(tickers) == 1:
        sub = df.dropna(how="all")
        if not sub.empty and "Close" in sub and sub["Close"].notna().any():
            out[tickers[0]] = sub
    return out


def _to_ist_index(frame: pd.DataFrame) -> pd.DataFrame:
    idx = pd.to_datetime(frame.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    frame = frame.copy()
    frame.index = idx.tz_convert(TZ)
    return frame


def _yf_download(tickers: list[str], **kwargs) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    params = dict(group_by="ticker", progress=False, threads=True, auto_adjust=False,
                  timeout=config.YAHOO_TIMEOUT_SECONDS)
    params.update(kwargs)
    try:
        return yf.download(tickers, multi_level_index=True, **params)
    except TypeError:  # older yfinance without multi_level_index
        return yf.download(tickers, **params)


def _f(x) -> float | None:
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


# ================================================================ symbol resolution
_RESOLVE_LOCK = threading.Lock()
_last_resolve_attempt = 0.0


def _candidates(base: str) -> list[str]:
    if base.upper() in config.SYMBOL_OVERRIDES:
        return [config.SYMBOL_OVERRIDES[base.upper()]]
    if "." in base and base.rsplit(".", 1)[1] in {"NS", "BO"}:
        return [base]                      # already a Yahoo symbol
    if base.isdigit():                     # BSE scrip code
        return [f"{base}.BO"]
    return [f"{base}{sfx}" for sfx in config.YAHOO_SUFFIXES]


def resolve_symbols(base_to_portfolio: dict[str, str], force: bool = False,
                    allow_network: bool = True) -> dict[str, dict]:
    """Map base symbols to verified Yahoo symbols.

    Each candidate (e.g. NMDC.NS then NMDC.BO) is checked against Yahoo and the
    first one that returns price history is accepted. Results are stored in
    SQLite so this happens once, not every 10 seconds.

    Returns {base: {"yahoo_symbol", "status", "method", "portfolio_symbol"}}.
    status: Verified | Override | Not found | Pending (Yahoo unreachable – retried later)
    """
    global _last_resolve_attempt
    cached = {r["base_symbol"]: r for r in db.query("SELECT * FROM symbol_map")}
    now = now_ist()
    todo = []
    for base, ps in base_to_portfolio.items():
        row = cached.get(base)
        if base.upper() in config.SYMBOL_OVERRIDES:
            ysym = config.SYMBOL_OVERRIDES[base.upper()]
            if not row or row["yahoo_symbol"] != ysym or row["status"] != "Override":
                _save_map(base, ps, ysym, "Override", "SYMBOL_OVERRIDES in .env")
            continue
        if row and row["status"] == "Verified" and not force:
            continue
        if row and row["status"] == "Not found" and not force:
            checked = datetime.fromisoformat(row["checked_at"])
            if now - checked < timedelta(hours=config.SYMBOL_RECHECK_HOURS):
                continue
        todo.append(base)

    if todo and allow_network and (force or time.time() - _last_resolve_attempt > 60) \
            and _RESOLVE_LOCK.acquire(blocking=False):
        try:
            _last_resolve_attempt = time.time()
            _verify_candidates(todo, base_to_portfolio)
        finally:
            _RESOLVE_LOCK.release()

    result = {}
    rows = {r["base_symbol"]: r for r in db.query("SELECT * FROM symbol_map")}
    for base, ps in base_to_portfolio.items():
        r = rows.get(base)
        if r and r["status"] in {"Verified", "Override"}:
            result[base] = {"yahoo_symbol": r["yahoo_symbol"], "status": r["status"],
                            "method": r["method"], "portfolio_symbol": ps}
        else:
            result[base] = {"yahoo_symbol": None,
                            "status": r["status"] if r else "Pending",
                            "method": r["method"] if r else "Waiting for Yahoo Finance",
                            "portfolio_symbol": ps}
    return result


def _save_map(base, ps, ysym, status, method):
    db.execute(
        "INSERT INTO symbol_map(base_symbol, portfolio_symbol, yahoo_symbol, status, method, checked_at) "
        "VALUES(?,?,?,?,?,?) ON CONFLICT(base_symbol) DO UPDATE SET portfolio_symbol=excluded.portfolio_symbol, "
        "yahoo_symbol=excluded.yahoo_symbol, status=excluded.status, method=excluded.method, "
        "checked_at=excluded.checked_at",
        (base, ps, ysym, status, method, db.now_iso()))


def _verify_candidates(bases: list[str], base_to_portfolio: dict[str, str]) -> None:
    remaining = list(bases)
    any_data = False
    attempted_all_rounds = True
    max_rounds = max(len(_candidates(b)) for b in bases) if bases else 0
    for rnd in range(max_rounds):
        batch = {b: _candidates(b)[rnd] for b in remaining if rnd < len(_candidates(b))}
        if not batch:
            break
        try:
            raw = _yf_download(list(batch.values()), period="5d", interval="1d")
            frames = _split_download(raw, list(batch.values()))
        except Exception as exc:
            STATUS.fail(f"Symbol check failed: {exc}")
            attempted_all_rounds = False
            break
        if frames:
            any_data = True
        for base, ysym in batch.items():
            if ysym in frames:
                _save_map(base, base_to_portfolio[base], ysym, "Verified",
                          f"Yahoo returned data for {ysym}")
                remaining.remove(base)
    if not any_data:
        # Nothing at all came back – treat as connectivity problem, not "not found".
        STATUS.fail("Yahoo Finance returned no data while resolving symbols (network / rate limit).")
        return
    if attempted_all_rounds:
        for base in remaining:
            _save_map(base, base_to_portfolio[base], None, "Not found",
                      "No Yahoo data for: " + ", ".join(_candidates(base)))


def symbol_table() -> pd.DataFrame:
    rows = db.query("SELECT portfolio_symbol, base_symbol, yahoo_symbol, status, method, checked_at "
                    "FROM symbol_map ORDER BY base_symbol")
    return pd.DataFrame(rows)


# ================================================================ daily history (cached)
_HISTORY_LOCK = threading.Lock()
_HISTORY: dict[str, dict] = {}   # symbol -> {"df": DataFrame, "fetched_at": epoch, "error": str|None}


def get_history(symbols: list[str], force: bool = False) -> dict[str, pd.DataFrame]:
    """Daily OHLCV (HISTORY_PERIOD, default 1y) for each symbol – cached, one batch download."""
    symbols = [s for s in dict.fromkeys(symbols) if s]
    ttl = config.HISTORY_CACHE_MINUTES * 60
    with _HISTORY_LOCK:
        stale = [s for s in symbols if force or s not in _HISTORY
                 or time.time() - _HISTORY[s]["fetched_at"] > ttl
                 or (_HISTORY[s]["df"] is None and time.time() - _HISTORY[s]["fetched_at"] > 120)]
        if stale:
            try:
                raw = _yf_download(stale, period=config.HISTORY_PERIOD, interval="1d")
                frames = _split_download(raw, stale)
                if not frames:
                    raise RuntimeError("Yahoo Finance returned no history (network / rate limit)")
                for s in stale:
                    if s in frames:
                        df = frames[s][["Open", "High", "Low", "Close", "Volume"]].copy()
                        idx = pd.to_datetime(df.index)
                        if idx.tz is not None:
                            idx = idx.tz_convert(TZ).tz_localize(None)
                        df.index = idx.normalize()
                        _HISTORY[s] = {"df": df.dropna(subset=["Close"]), "fetched_at": time.time(), "error": None}
                    else:
                        prev = _HISTORY.get(s, {}).get("df")
                        _HISTORY[s] = {"df": prev, "fetched_at": time.time(), "error": "No Yahoo history"}
            except Exception as exc:
                for s in stale:
                    prev = _HISTORY.get(s, {}).get("df")
                    _HISTORY[s] = {"df": prev, "fetched_at": time.time() - ttl + 120, "error": str(exc)}
                STATUS.fail(f"History download failed: {exc}")
        return {s: _HISTORY[s]["df"] for s in symbols if _HISTORY.get(s, {}).get("df") is not None}


def history_errors() -> dict[str, str]:
    return {s: v["error"] for s, v in _HISTORY.items() if v.get("error")}


def _official_close(sym: str, session_date) -> float | None:
    """Today's daily-bar close from Yahoo (the exchange close) if Yahoo has published it."""
    entry = _HISTORY.get(sym)
    df = entry["df"] if entry else None
    if df is None or df.empty:
        return None
    same = df[df.index.date == session_date]
    return _f(same["Close"].iloc[-1]) if len(same) else None


def _previous_close(sym: str, session_date) -> float | None:
    entry = _HISTORY.get(sym)
    df = entry["df"] if entry else None
    if df is None or df.empty:
        return None
    before = df[df.index.date < session_date]
    return _f(before["Close"].iloc[-1]) if len(before) else None


# ================================================================ live quotes (10 s cache)
_QUOTE_LOCK = threading.Lock()
_HISTORY_AFTER_CLOSE: dict = {"date": None}
_QUOTE_CACHE: dict = {"symbols": frozenset(), "fetched_at": 0.0, "quotes": {}}
_LAST_GOOD: dict[str, dict] = {}
_INTRADAY: dict[str, pd.DataFrame] = {}


def _load_last_good_from_db(symbols: list[str]) -> None:
    """On a cold start, show the last stored Yahoo price as Stale rather than nothing."""
    missing = [s for s in symbols if s not in _LAST_GOOD]
    if not missing:
        return
    ph = ",".join("?" * len(missing))
    rows = db.query(
        f"SELECT p.* FROM price_snapshots p JOIN (SELECT yahoo_symbol, MAX(bucket) b FROM price_snapshots "
        f"WHERE yahoo_symbol IN ({ph}) GROUP BY yahoo_symbol) m ON p.yahoo_symbol=m.yahoo_symbol AND p.bucket=m.b",
        missing)
    for r in rows:
        ts = datetime.fromisoformat(r["data_timestamp"]) if r["data_timestamp"] else None
        _LAST_GOOD[r["yahoo_symbol"]] = {
            "symbol": r["yahoo_symbol"], "price": r["price"], "previous_close": r["previous_close"],
            "open": r["open"], "day_high": r["day_high"], "day_low": r["day_low"], "volume": r["volume"],
            "data_timestamp": ts, "fetched_at": datetime.fromisoformat(r["fetched_at"]),
            "status": stored_quote_status(ts), "error": "Loaded from local database (last Yahoo value)",
        }


def cached_quotes(symbols: list[str]) -> dict[str, dict]:
    """Non-blocking read of the latest quotes (used by the dashboard while the
    background monitor does the Yahoo requests). Never touches the network."""
    out = {}
    cache = _QUOTE_CACHE["quotes"]
    if any(s not in cache and s not in _LAST_GOOD for s in symbols):
        try:
            _load_last_good_from_db(symbols)
        except Exception:
            pass
    for s in symbols:
        if s in cache:
            out[s] = cache[s]
        elif s in _LAST_GOOD:
            q = dict(_LAST_GOOD[s])
            q["status"] = stored_quote_status(q.get("data_timestamp"))
            q["error"] = q.get("error") or "Waiting for next Yahoo refresh"
            out[s] = q
        else:
            out[s] = {"symbol": s, "price": None, "previous_close": None, "open": None, "day_high": None,
                      "day_low": None, "volume": None, "data_timestamp": None, "fetched_at": None,
                      "status": "UNAVAILABLE", "error": "Waiting for first Yahoo refresh"}
    return out


def get_quotes(symbols: list[str], force: bool = False) -> dict[str, dict]:
    """Latest quote per Yahoo symbol, refreshed at most every MARKET_REFRESH_SECONDS.

    Quote fields: price, previous_close, open, day_high, day_low, volume,
    data_timestamp (Yahoo bar time), fetched_at (local), status, error.
    status: Live | Market closed | Stale | Unavailable
    """
    symbols = sorted(s for s in dict.fromkeys(symbols) if s)
    if not symbols:
        return {}
    key = frozenset(symbols)
    with _QUOTE_LOCK:
        fresh = time.time() - _QUOTE_CACHE["fetched_at"] < config.MARKET_REFRESH_SECONDS
        if not force and fresh and key <= _QUOTE_CACHE["symbols"]:
            return {s: _QUOTE_CACHE["quotes"][s] for s in symbols if s in _QUOTE_CACHE["quotes"]}
        _load_last_good_from_db(symbols)
        now = now_ist()
        session = market_status(now)
        after_close = (session["trading_day"] and not session["is_open"]
                       and now.time() >= _parse_hhmm(config.MARKET_CLOSE_TIME))
        day_key = now.strftime("%Y-%m-%d")
        if after_close and _HISTORY_AFTER_CLOSE.get("date") != day_key:
            # Refresh daily bars once after the close so today's official close is available.
            get_history(symbols, force=True)
            _HISTORY_AFTER_CLOSE["date"] = day_key
        else:
            get_history(symbols)  # cached; supplies previous close
        quotes = _fetch_quotes(symbols)
        _QUOTE_CACHE.update(symbols=key, fetched_at=time.time(), quotes=quotes)
        return quotes


def _fetch_quotes(symbols: list[str]) -> dict[str, dict]:
    now = now_ist()
    session = market_status(now)
    quotes: dict[str, dict] = {}
    try:
        raw = _yf_download(symbols, period="1d", interval="1m", prepost=False)
        frames = _split_download(raw, symbols)
        if not frames:
            raise RuntimeError("Yahoo Finance returned no quote data (network / rate limit)")
        STATUS.ok()
        err = None
    except Exception as exc:
        frames, err = {}, str(exc)
        STATUS.fail(err)

    snapshot_rows = []
    bucket = now.strftime("%Y-%m-%d %H:%M")
    for s in symbols:
        fr = frames.get(s)
        if fr is not None:
            fr = _to_ist_index(fr).dropna(subset=["Close"])
        if fr is not None and not fr.empty:
            _INTRADAY[s] = fr
            last_ts = fr.index[-1].to_pydatetime()
            session_date = last_ts.date()
            price = _f(fr["Close"].iloc[-1])
            if price is None or price <= 0:
                fr = None  # invalid Yahoo value – handled below as missing, never as 0
        if fr is not None and not fr.empty:
            q = {
                "symbol": s, "price": price,
                "previous_close": _previous_close(s, session_date),
                "open": _f(fr["Open"].dropna().iloc[0]) if fr["Open"].notna().any() else None,
                "day_high": _f(fr["High"].max()), "day_low": _f(fr["Low"].min()),
                "volume": _f(fr["Volume"].fillna(0).sum()),
                "data_timestamp": last_ts, "fetched_at": now, "error": None,
            }
            age = (now - last_ts).total_seconds()
            if session["is_open"]:
                q["status"] = "Live" if age <= config.STALE_AFTER_SECONDS else "Stale"
                if q["status"] == "Stale":
                    q["error"] = f"Last Yahoo bar is {int(age // 60)} min old"
            else:
                q["status"] = "Market closed"
                close_px = None
                if (session["trading_day"] and session_date == now.date()
                        and now.time() >= _parse_hhmm(config.MARKET_CLOSE_TIME)):
                    close_px = _official_close(s, session_date)
                if close_px and close_px > 0:
                    q["price"], q["price_source"] = close_px, "Yahoo daily close"
                    q["status"] = "Closed"
                else:
                    q["price_source"] = "Last Yahoo 1-min bar"
            if q["previous_close"] is not None and q["previous_close"] <= 0:
                q["previous_close"] = None
            _LAST_GOOD[s] = q
            quotes[s] = q
            snapshot_rows.append((s, bucket, q["price"], q["previous_close"], q["open"], q["day_high"],
                                  q["day_low"], q["volume"], last_ts.isoformat(timespec="seconds"),
                                  now.isoformat(timespec="seconds")))
        elif s in _LAST_GOOD:
            q = dict(_LAST_GOOD[s])
            q["status"] = stored_quote_status(q.get("data_timestamp"), now)
            q["error"] = err or "No fresh Yahoo data for this symbol"
            quotes[s] = q
        else:
            quotes[s] = {"symbol": s, "price": None, "previous_close": _previous_close(s, now.date()),
                         "open": None, "day_high": None, "day_low": None, "volume": None,
                         "data_timestamp": None, "fetched_at": now, "status": "UNAVAILABLE",
                         "error": err or "No Yahoo data for this symbol"}
    fresh = {row[0] for row in snapshot_rows}          # symbols Yahoo priced in THIS refresh
    failed = sorted(set(symbols) - fresh)
    with STATUS.lock:
        STATUS.last_ok_count = len(symbols) - len(failed)
        STATUS.last_failed_count = len(failed)
        STATUS.failed_symbols = failed
    if failed:
        log.warning("Yahoo refresh: %d/%d symbols without a fresh price: %s",
                    len(failed), len(symbols), ", ".join(failed))
    if snapshot_rows:
        try:
            db.executemany(
                "INSERT INTO price_snapshots(yahoo_symbol, bucket, price, previous_close, open, day_high, "
                "day_low, volume, data_timestamp, fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(yahoo_symbol, bucket) DO UPDATE SET price=excluded.price, "
                "previous_close=excluded.previous_close, open=excluded.open, day_high=excluded.day_high, "
                "day_low=excluded.day_low, volume=excluded.volume, data_timestamp=excluded.data_timestamp, "
                "fetched_at=excluded.fetched_at", snapshot_rows)
        except Exception:
            log.exception("Could not store price snapshots")
    return quotes


def get_intraday(symbol: str) -> pd.DataFrame | None:
    """Today's 1-minute bars from the last quote refresh (used for VWAP)."""
    return _INTRADAY.get(symbol)


def latest_data_timestamp(quotes: dict[str, dict]) -> datetime | None:
    ts = [q["data_timestamp"] for q in quotes.values() if q.get("data_timestamp")]
    return max(ts) if ts else None


def price_basis(market: dict, yahoo_ts: datetime | None, now: datetime | None = None) -> dict:
    """How to label the prices in a snapshot/report so a 3:29 PM value is never called a close.

    LIVE / PRE-CLOSE     – market open: intraday prices, not closing prices
    CLOSED / LAST CLOSE  – session over and Yahoo data is from today's close
    LAST AVAILABLE       – session over but Yahoo's latest bar is earlier than the close
    PREVIOUS CLOSE       – before today's open / weekend / holiday: last session's prices
    UNAVAILABLE          – no Yahoo timestamp at all
    """
    now = now or now_ist()
    if yahoo_ts is None:
        return {"label": "UNAVAILABLE", "detail": "No Yahoo price data yet"}
    ts = yahoo_ts.astimezone(TZ) if yahoo_ts.tzinfo else yahoo_ts.replace(tzinfo=TZ)
    close_t = _parse_hhmm(config.MARKET_CLOSE_TIME)
    last_bar_t = (datetime.combine(ts.date(), close_t) - timedelta(minutes=1)).time()
    if market["status"] == "OPEN":
        return {"label": "LIVE / PRE-CLOSE",
                "detail": f"Intraday Yahoo prices as of {fmt_time(ts)} – not closing prices"}
    if market["trading_day"] and ts.date() == now.date() and now.time() >= close_t:
        if ts.time() >= last_bar_t:
            return {"label": "CLOSED / LAST CLOSE",
                    "detail": f"Session closed; Yahoo data through {fmt_datetime(ts)}"}
        return {"label": "CLOSED / LAST AVAILABLE",
                "detail": f"Session closed but latest Yahoo bar is {fmt_datetime(ts)}"}
    return {"label": "PREVIOUS CLOSE", "detail": f"Last session's Yahoo prices ({fmt_datetime(ts)})"}
