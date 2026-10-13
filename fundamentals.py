"""
Fundamental data from Yahoo Finance (yfinance Ticker.info + financial statements).

* Cached in SQLite for FUNDAMENTALS_CACHE_HOURS (default 24 h).
* Missing Yahoo fields are returned as None and displayed as "N/A".
  Nothing is estimated or back-filled from another source.
"""
from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timedelta

import pandas as pd

import config
from modules import database as db
from modules.market_data import now_ist, yf

_LOCK = threading.Lock()

FIELDS = [
    # (key, label, kind)  kind: text | money | ratio | pct | number
    ("company_name", "Company Name", "text"),
    ("sector", "Sector", "text"),
    ("industry", "Industry", "text"),
    ("market_cap", "Market Cap", "money"),
    ("revenue", "Revenue (TTM)", "money"),
    ("net_income", "Net Income", "money"),
    ("eps", "EPS (TTM)", "number"),
    ("pe", "P/E (TTM)", "ratio"),
    ("forward_pe", "Forward P/E", "ratio"),
    ("pb", "P/B", "ratio"),
    ("roe", "ROE", "pct"),
    ("total_debt", "Total Debt", "money"),
    ("total_assets", "Total Assets", "money"),
    ("debt_to_equity", "Debt/Equity (Yahoo)", "ratio"),
    ("revenue_growth", "Revenue Growth (YoY)", "pct"),
    ("earnings_growth", "Earnings Growth (YoY)", "pct"),
    ("dividend_yield", "Dividend Yield", "pct_points"),
    ("free_cash_flow", "Free Cash Flow", "money"),
    ("beta", "Beta (Yahoo)", "ratio"),
    ("fifty_two_week_high", "52-Week High", "number"),
    ("fifty_two_week_low", "52-Week Low", "number"),
]


def _num(x):
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def _statement_value(frame: pd.DataFrame | None, names: list[str]):
    if frame is None or getattr(frame, "empty", True):
        return None
    for n in names:
        if n in frame.index:
            row = frame.loc[n].dropna()
            if len(row):
                return _num(row.iloc[0])  # most recent period first
    return None


def fetch_from_yahoo(yahoo_symbol: str) -> dict:
    if yf is None:
        raise RuntimeError("yfinance is not installed")
    t = yf.Ticker(yahoo_symbol)
    info = t.info or {}
    if not info or (len(info) <= 2 and "trailingPegRatio" in info):
        raise RuntimeError("Yahoo returned no fundamentals")
    total_assets = None
    try:
        total_assets = _statement_value(t.balance_sheet, ["Total Assets"])
    except Exception:
        pass
    net_income = _num(info.get("netIncomeToCommon"))
    if net_income is None:
        try:
            net_income = _statement_value(t.financials, ["Net Income", "Net Income Common Stockholders"])
        except Exception:
            pass
    # Dividend yield: trailingAnnualDividendYield is a fraction (0.012 = 1.2 %).
    # yfinance's dividendYield is already in percent points in current versions.
    dy = _num(info.get("trailingAnnualDividendYield"))
    dy = dy * 100 if dy is not None else _num(info.get("dividendYield"))
    roe = _num(info.get("returnOnEquity"))
    return {
        "company_name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": info.get("currency") or info.get("financialCurrency"),
        "market_cap": _num(info.get("marketCap")),
        "revenue": _num(info.get("totalRevenue")),
        "net_income": net_income,
        "eps": _num(info.get("trailingEps")),
        "pe": _num(info.get("trailingPE")),
        "forward_pe": _num(info.get("forwardPE")),
        "pb": _num(info.get("priceToBook")),
        "roe": roe * 100 if roe is not None else None,
        "total_debt": _num(info.get("totalDebt")),
        "total_assets": total_assets,
        "debt_to_equity": _num(info.get("debtToEquity")),
        "revenue_growth": (_num(info.get("revenueGrowth")) * 100) if _num(info.get("revenueGrowth")) is not None else None,
        "earnings_growth": (_num(info.get("earningsGrowth")) * 100) if _num(info.get("earningsGrowth")) is not None else None,
        "dividend_yield": dy,
        "free_cash_flow": _num(info.get("freeCashflow")),
        "beta": _num(info.get("beta")),
        "fifty_two_week_high": _num(info.get("fiftyTwoWeekHigh")),
        "fifty_two_week_low": _num(info.get("fiftyTwoWeekLow")),
        "business_summary": info.get("longBusinessSummary"),
        "website": info.get("website"),
    }


def get_cached(yahoo_symbol: str) -> dict | None:
    row = db.query_one("SELECT * FROM fundamentals WHERE yahoo_symbol=?", (yahoo_symbol,))
    if not row:
        return None
    data = json.loads(row["data_json"]) if row["data_json"] else {}
    data["_status"] = row["status"]
    data["_fetched_at"] = row["fetched_at"]
    return data


def get_fundamentals(yahoo_symbol: str, force: bool = False) -> dict:
    """Return fundamentals (from cache when fresh). On Yahoo failure the previous
    cached copy is returned with _status='Stale', or {'_status': 'Unavailable'}."""
    cached = get_cached(yahoo_symbol)
    if cached and not force and cached.get("_status") == "OK":
        age = now_ist() - datetime.fromisoformat(cached["_fetched_at"])
        if age < timedelta(hours=config.FUNDAMENTALS_CACHE_HOURS):
            return cached
    if cached and not force and cached.get("_status") == "Error":
        age = now_ist() - datetime.fromisoformat(cached["_fetched_at"])
        if age < timedelta(minutes=30):          # don't hammer Yahoo after a failure
            return cached
    with _LOCK:
        try:
            data = fetch_from_yahoo(yahoo_symbol)
            db.execute(
                "INSERT INTO fundamentals(yahoo_symbol, data_json, status, fetched_at) VALUES(?,?,?,?) "
                "ON CONFLICT(yahoo_symbol) DO UPDATE SET data_json=excluded.data_json, status=excluded.status, "
                "fetched_at=excluded.fetched_at", (yahoo_symbol, json.dumps(data), "OK", db.now_iso()))
            data["_status"], data["_fetched_at"] = "OK", db.now_iso()
            return data
        except Exception as exc:
            if cached and cached.get("_status") in {"OK", "STALE", "Stale"} and any(
                    cached.get(k) is not None for k, _, _ in FIELDS):
                cached["_status"], cached["_error"] = "STALE", str(exc)
                return cached
            db.execute(
                "INSERT INTO fundamentals(yahoo_symbol, data_json, status, fetched_at) VALUES(?,?,?,?) "
                "ON CONFLICT(yahoo_symbol) DO UPDATE SET status=excluded.status, fetched_at=excluded.fetched_at",
                (yahoo_symbol, json.dumps({"_error": str(exc)}), "Error", db.now_iso()))
            return {"_status": "UNAVAILABLE", "_error": str(exc)}


def sector_map(yahoo_symbols: list[str]) -> dict[str, str]:
    """Sector per symbol from CACHED fundamentals only (no network). Unknown → 'Unknown'."""
    if not yahoo_symbols:
        return {}
    ph = ",".join("?" * len(yahoo_symbols))
    rows = db.query(f"SELECT yahoo_symbol, data_json FROM fundamentals WHERE yahoo_symbol IN ({ph})",
                    yahoo_symbols)
    out = {s: "Unknown" for s in yahoo_symbols}
    for r in rows:
        try:
            sec = json.loads(r["data_json"] or "{}").get("sector")
            if sec:
                out[r["yahoo_symbol"]] = sec
        except Exception:
            pass
    return out


def missing_or_stale(yahoo_symbols: list[str]) -> list[str]:
    out = []
    for s in yahoo_symbols:
        c = get_cached(s)
        if not c:
            out.append(s)
            continue
        age = now_ist() - datetime.fromisoformat(c["_fetched_at"])
        if c["_status"] == "OK" and age > timedelta(hours=config.FUNDAMENTALS_CACHE_HOURS):
            out.append(s)
        elif c["_status"] != "OK" and age > timedelta(minutes=30):
            out.append(s)
    return out


def format_value(value, kind: str, currency: str = config.CURRENCY_SYMBOL) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if kind == "text":
        return str(value)
    if kind == "money":
        return format_inr_compact(value, currency)
    if kind == "pct" or kind == "pct_points":
        return f"{value:.2f}%"
    if kind == "ratio":
        return f"{value:.2f}"
    return f"{value:,.2f}"


def format_inr_compact(value: float, currency: str = config.CURRENCY_SYMBOL) -> str:
    """₹ in Indian units: Cr (1e7) / L (1e5)."""
    if value is None:
        return "N/A"
    sign = "-" if value < 0 else ""
    v = abs(value)
    from modules.reports import indian_group
    if v >= 1e7:
        return f"{sign}{currency}{indian_group(v / 1e7, 2)} Cr"
    if v >= 1e5:
        return f"{sign}{currency}{indian_group(v / 1e5, 2)} L"
    return f"{sign}{currency}{indian_group(v, 0)}"


def as_table(data: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Metric": label, "Value": format_value(data.get(key), kind)}
                         for key, label, kind in FIELDS])
