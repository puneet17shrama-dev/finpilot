"""
Technical indicators computed from cached Yahoo Finance daily history.

The one-year history is downloaded in a single cached batch (see
market_data.get_history), NOT every 10 seconds. The live Yahoo price is
spliced in as today's value so price-vs-average comparisons stay current.

Indicators describe past price/volume behaviour only. They do not predict
or guarantee future price movement.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from modules import database as db

MIN_BARS = {"sma20": 20, "sma50": 50, "sma200": 200, "ema20": 20, "ema50": 50,
            "rsi14": 15, "macd": 35, "bb": 20, "avg_vol": 20}


def _last(series: pd.Series) -> float | None:
    if series is None or len(series) == 0:
        return None
    v = series.iloc[-1]
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def sma(close: pd.Series, n: int) -> pd.Series:
    return close.rolling(n, min_periods=n).mean()


def ema(close: pd.Series, n: int) -> pd.Series:
    return close.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, n=20, k=2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return mid + k * sd, mid, mid - k * sd


def intraday_vwap(bars: pd.DataFrame | None) -> float | None:
    """Session VWAP from today's 1-minute Yahoo bars (typical price × volume)."""
    if bars is None or bars.empty or "Volume" not in bars:
        return None
    b = bars.dropna(subset=["High", "Low", "Close", "Volume"])
    vol = b["Volume"].astype(float)
    if vol.sum() <= 0:
        return None
    tp = (b["High"] + b["Low"] + b["Close"]) / 3
    return float((tp * vol).sum() / vol.sum())


def with_live_price(hist: pd.DataFrame, quote: dict | None) -> pd.DataFrame:
    """Replace/append today's bar using the live Yahoo quote (no invented values)."""
    if hist is None or hist.empty or not quote or quote.get("price") is None or not quote.get("data_timestamp"):
        return hist
    day = pd.Timestamp(quote["data_timestamp"].date())
    df = hist.copy()
    row = {"Open": quote.get("open") or quote["price"], "High": quote.get("day_high") or quote["price"],
           "Low": quote.get("day_low") or quote["price"], "Close": quote["price"],
           "Volume": quote.get("volume") if quote.get("volume") is not None else np.nan}
    df.loc[day, list(row)] = list(row.values())
    return df.sort_index()


def compute_indicators(hist: pd.DataFrame | None, quote: dict | None = None,
                       intraday: pd.DataFrame | None = None) -> dict:
    """Return a flat dict of indicator values. Missing / insufficient data → None."""
    out: dict = {"bars": 0, "notes": []}
    if hist is None or hist.empty:
        out["notes"].append("No Yahoo history available")
        return out
    df = with_live_price(hist, quote)
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    n = len(close)
    out["bars"] = n
    out["as_of"] = str(df.index[-1].date())
    price = _last(close)
    out["price"] = price

    for key, length in (("sma20", 20), ("sma50", 50), ("sma200", 200)):
        out[key] = _last(sma(close, length)) if n >= length else None
    for key, length in (("ema20", 20), ("ema50", 50)):
        out[key] = _last(ema(close, length)) if n >= length else None
    out["rsi14"] = _last(rsi(close, 14)) if n >= MIN_BARS["rsi14"] else None
    if n >= MIN_BARS["macd"]:
        line, sig, hist_ = macd(close)
        out["macd"], out["macd_signal"], out["macd_hist"] = _last(line), _last(sig), _last(hist_)
    else:
        out["macd"] = out["macd_signal"] = out["macd_hist"] = None
    if n >= 20:
        up, mid, lo = bollinger(close)
        out["bb_upper"], out["bb_middle"], out["bb_lower"] = _last(up), _last(mid), _last(lo)
        window = df.iloc[-20:]
        out["support_20d"] = float(window["Low"].min())
        out["resistance_20d"] = float(window["High"].max())
    else:
        out["bb_upper"] = out["bb_middle"] = out["bb_lower"] = None
        out["support_20d"] = out["resistance_20d"] = None
    # Classic floor pivots from the previous completed session
    if n >= 2:
        prev = df.iloc[-2]
        pivot = (prev["High"] + prev["Low"] + prev["Close"]) / 3
        out["pivot"] = float(pivot)
        out["pivot_s1"] = float(2 * pivot - prev["High"])
        out["pivot_r1"] = float(2 * pivot - prev["Low"])
    out["volume"] = _last(vol)
    prior_vol = vol.iloc[:-1]
    out["avg_volume_20d"] = float(prior_vol.iloc[-20:].mean()) if len(prior_vol.dropna()) >= 20 else None
    out["volume_ratio"] = (out["volume"] / out["avg_volume_20d"]
                           if out["volume"] is not None and out["avg_volume_20d"] else None)
    out["vwap"] = intraday_vwap(intraday)
    if out["vwap"] is None:
        out["notes"].append("VWAP needs today's intraday Yahoo bars")

    def pct_vs(ref):
        return (price / ref - 1) * 100 if price is not None and ref else None
    for key in ("sma20", "sma50", "sma200", "ema20", "ema50", "vwap"):
        out[f"price_vs_{key}_pct"] = pct_vs(out.get(key))
    need = {"sma20": 20, "sma50": 50, "sma200": 200, "ema20": 20, "ema50": 50, "rsi14": MIN_BARS["rsi14"],
            "macd": MIN_BARS["macd"], "macd_signal": MIN_BARS["macd"], "macd_hist": MIN_BARS["macd"],
            "bb_upper": 20, "bb_middle": 20, "bb_lower": 20, "support_20d": 20, "resistance_20d": 20,
            "avg_volume_20d": 21, "volume_ratio": 21}
    out["insufficient"] = {k: v for k, v in need.items() if n < v}
    if out["insufficient"]:
        out["notes"].append(f"Only {n} daily bars – insufficient data for: "
                            + ", ".join(sorted(out["insufficient"])))
    return out


def display_value(ind: dict, key: str, decimals: int = 2) -> str:
    """Formatted indicator value, or an explicit reason instead of a misleading number."""
    v = ind.get(key)
    if v is not None:
        return f"{v:,.{decimals}f}"
    if key in (ind.get("insufficient") or {}):
        return f"Insufficient data ({ind.get('bars', 0)}/{ind['insufficient'][key]} bars)"
    return "N/A"


def describe(ind: dict) -> list[str]:
    """Plain-language, factual observations (no predictions, no buy/sell)."""
    obs = []
    p = ind.get("price")
    if p is None:
        return ["Price unavailable – indicators cannot be compared."]
    for key, label in (("sma50", "50-day SMA"), ("sma200", "200-day SMA"), ("ema20", "20-day EMA")):
        v = ind.get(key)
        if v:
            obs.append(f"Price is {'above' if p > v else 'below'} its {label} ({ind[f'price_vs_{key}_pct']:+.1f}%).")
    r = ind.get("rsi14")
    if r is not None:
        zone = "above 70 (historically labelled 'overbought')" if r > 70 else \
               "below 30 (historically labelled 'oversold')" if r < 30 else "between 30 and 70"
        obs.append(f"RSI(14) is {r:.1f}, {zone}.")
    if ind.get("macd") is not None and ind.get("macd_signal") is not None:
        obs.append(f"MACD is {'above' if ind['macd'] > ind['macd_signal'] else 'below'} its signal line.")
    if ind.get("bb_upper") and ind.get("bb_lower"):
        if p > ind["bb_upper"]:
            obs.append("Price is above the upper Bollinger Band.")
        elif p < ind["bb_lower"]:
            obs.append("Price is below the lower Bollinger Band.")
        else:
            obs.append("Price is inside the Bollinger Bands.")
    vr = ind.get("volume_ratio")
    if vr is not None:
        obs.append(f"Today's volume is {vr:.1f}× the 20-day average.")
    return obs


def indicator_frame(hist: pd.DataFrame, quote: dict | None = None) -> pd.DataFrame:
    """Full indicator series for charting."""
    df = with_live_price(hist, quote).copy()
    c = df["Close"].astype(float)
    df["SMA20"], df["SMA50"], df["SMA200"] = sma(c, 20), sma(c, 50), sma(c, 200)
    df["EMA20"], df["EMA50"] = ema(c, 20), ema(c, 50)
    df["BB_upper"], df["BB_middle"], df["BB_lower"] = bollinger(c)
    df["RSI14"] = rsi(c)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(c)
    return df


def save_snapshot(yahoo_symbol: str, ind: dict) -> None:
    """One stored row per symbol per trading date (upsert – no duplicates from refreshes)."""
    if not ind.get("as_of"):
        return
    try:
        clean = {k: v for k, v in ind.items() if isinstance(v, (int, float, str, list, type(None)))}
        db.execute(
            "INSERT INTO technical_snapshots(yahoo_symbol, as_of_date, data_json, computed_at) VALUES(?,?,?,?) "
            "ON CONFLICT(yahoo_symbol, as_of_date) DO UPDATE SET data_json=excluded.data_json, "
            "computed_at=excluded.computed_at",
            (yahoo_symbol, ind["as_of"], json.dumps(clean, default=str), db.now_iso()))
    except Exception:
        pass
