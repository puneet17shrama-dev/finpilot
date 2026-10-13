"""
Portfolio scanner – analyses ONLY the stocks in the user's holdings file.

It does not scan the wider market and it does not produce BUY / SELL /
"best" / "worst" labels. For each holding it combines price, technicals,
fundamentals (from cache), volume, risk and news into:

  * a transparent CONDITION SCORE: the share of factual checks that are met,
    with every check listed so the user can see exactly what drove it;
  * a list of neutral FLAGS (things worth looking at, e.g. unusual volume).

A higher condition score only means more of the listed conditions are true
right now. It is not a forecast and not a recommendation.
"""
from __future__ import annotations

import pandas as pd

from modules import database as db
from modules import fundamentals as fnd
from modules import market_data as md
from modules import news as nws
from modules import technical as ta

CHECKS = [
    # (id, group, description)
    ("above_sma50", "Trend", "Price above 50-day SMA"),
    ("above_sma200", "Trend", "Price above 200-day SMA"),
    ("sma50_above_sma200", "Trend", "50-day SMA above 200-day SMA"),
    ("macd_above_signal", "Momentum", "MACD above signal line"),
    ("rsi_mid", "Momentum", "RSI(14) between 40 and 70"),
    ("pe_positive", "Fundamentals", "Positive trailing P/E reported (company profitable, TTM)"),
    ("roe_15", "Fundamentals", "ROE above 15%"),
    ("rev_growth_pos", "Fundamentals", "Positive revenue growth (YoY)"),
    ("no_negative_news", "News", "No negative-keyword headlines in last 7 days"),
]


def scan(positions: pd.DataFrame, history: dict, quotes: dict, benchmark=None) -> pd.DataFrame:
    news_df = nws.get_news([s for s in positions["yahoo_symbol"].dropna()], days=7)
    fetched = {r["yahoo_symbol"] for r in db.query(
        "SELECT yahoo_symbol FROM news_fetch_log WHERE status LIKE 'ok%'")}
    rows = []
    for r in positions.itertuples():
        sym = r.yahoo_symbol
        q = quotes.get(sym) if sym else None
        ind = ta.compute_indicators(history.get(sym), q, md.get_intraday(sym)) if sym else {"notes": []}
        f = fnd.get_cached(sym) if sym else None
        f = f if f and f.get("_status") in {"OK", "Stale"} else {}
        sym_news = news_df[news_df["yahoo_symbol"] == sym] if len(news_df) else pd.DataFrame()

        p = ind.get("price")
        results: dict[str, bool | None] = {}

        def chk(cid, cond):
            results[cid] = None if cond is None else bool(cond)

        chk("above_sma50", (p > ind["sma50"]) if p and ind.get("sma50") else None)
        chk("above_sma200", (p > ind["sma200"]) if p and ind.get("sma200") else None)
        chk("sma50_above_sma200", (ind["sma50"] > ind["sma200"]) if ind.get("sma50") and ind.get("sma200") else None)
        chk("macd_above_signal", (ind["macd"] > ind["macd_signal"])
            if ind.get("macd") is not None and ind.get("macd_signal") is not None else None)
        chk("rsi_mid", (40 <= ind["rsi14"] <= 70) if ind.get("rsi14") is not None else None)
        chk("pe_positive", (f["pe"] > 0) if f.get("pe") is not None else
            (False if f.get("eps") is not None and f["eps"] <= 0 else None))
        chk("roe_15", (f["roe"] > 15) if f.get("roe") is not None else None)
        chk("rev_growth_pos", (f["revenue_growth"] > 0) if f.get("revenue_growth") is not None else None)
        chk("no_negative_news", (not (sym_news["sentiment"] == "Negative").any()) if len(sym_news) else
            (True if sym in fetched else None))

        evaluated = {k: v for k, v in results.items() if v is not None}
        met = sum(evaluated.values())
        score = round(met / len(evaluated) * 100) if evaluated else None

        flags = []
        vr = ind.get("volume_ratio")
        if vr is not None and vr >= 2:
            flags.append(f"Volume {vr:.1f}× 20-day avg")
        if ind.get("rsi14") is not None and ind["rsi14"] > 70:
            flags.append(f"RSI {ind['rsi14']:.0f} (>70)")
        if ind.get("rsi14") is not None and ind["rsi14"] < 30:
            flags.append(f"RSI {ind['rsi14']:.0f} (<30)")
        if getattr(r, "day_change_pct", None) is not None and pd.notna(r.day_change_pct) and abs(r.day_change_pct) >= 5:
            flags.append(f"Day move {r.day_change_pct:+.1f}%")
        if pd.notna(r.return_pct) and r.return_pct <= -30:
            flags.append(f"Position {r.return_pct:+.0f}% vs cost")
        if len(sym_news) and (sym_news["category"] == "Important Event").any():
            flags.append("Important-event news (7d)")
        if r.has_zero_cost_lot:
            flags.append("Includes zero-cost lot")
        if not sym:
            flags.append("Yahoo symbol unresolved")

        rows.append({
            "Stock": r.base_symbol,
            "Yahoo Symbol": sym or "Unresolved",
            "Price": p,
            "Day %": r.day_change_pct if pd.notna(r.day_change_pct) else None,
            "Return %": r.return_pct if pd.notna(r.return_pct) else None,
            "RSI14": ind.get("rsi14"),
            "Vol Ratio": vr,
            "P/E": f.get("pe"),
            "ROE %": f.get("roe"),
            "News 7d": int(len(sym_news)),
            "Condition Score": score,
            "Checks Met": f"{met}/{len(evaluated)}" if evaluated else "0/0",
            "Flags": "; ".join(flags) if flags else "—",
            "_checks": results,
        })
    return pd.DataFrame(rows)


def explain(row: pd.Series) -> pd.DataFrame:
    """Per-check breakdown for one scanned stock."""
    res = row["_checks"]
    return pd.DataFrame([{
        "Group": g, "Condition": d,
        "Result": "Met" if res.get(cid) is True else "Not met" if res.get(cid) is False else "N/A (data unavailable)",
    } for cid, g, d in CHECKS])
