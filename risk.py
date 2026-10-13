"""
Portfolio risk metrics from actual holdings + cached Yahoo daily history.

All numbers describe historical behaviour and current composition. They do
not label any stock as "safe" or "unsafe" and do not predict future risk.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _f(x):
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def concentration(positions: pd.DataFrame) -> dict:
    """Weights by current value (only priced holdings)."""
    p = positions[positions["current_value"].notna() & (positions["current_value"] > 0)]
    if p.empty:
        return {"available": False}
    w = (p.set_index("base_symbol")["current_value"] / p["current_value"].sum()).sort_values(ascending=False)
    hhi = float((w ** 2).sum())
    return {
        "available": True,
        "weights": w,
        "largest_symbol": w.index[0],
        "largest_weight_pct": float(w.iloc[0] * 100),
        "top5_weight_pct": float(w.iloc[:5].sum() * 100),
        "hhi": hhi,
        "effective_holdings": 1 / hhi if hhi else None,
        "holdings_over_10pct": [s for s, v in w.items() if v > 0.10],
        "priced_holdings": int(len(p)),
    }


def sector_exposure(positions: pd.DataFrame, sectors: dict[str, str]) -> pd.DataFrame:
    p = positions[positions["current_value"].notna()].copy()
    if p.empty:
        return pd.DataFrame(columns=["sector", "current_value", "weight_pct", "holdings"])
    p["sector"] = p["yahoo_symbol"].map(lambda s: sectors.get(s, "Unknown") if s else "Unknown")
    g = p.groupby("sector").agg(current_value=("current_value", "sum"), holdings=("base_symbol", "count"))
    g["weight_pct"] = g["current_value"] / g["current_value"].sum() * 100
    return g.sort_values("current_value", ascending=False).reset_index()


def returns_frame(history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = {s: df["Close"] for s, df in history.items() if df is not None and len(df) > 1}
    if not closes:
        return pd.DataFrame()
    return pd.DataFrame(closes).sort_index().pct_change(fill_method=None)


def max_drawdown(close: pd.Series) -> float | None:
    c = close.dropna()
    if len(c) < 2:
        return None
    dd = c / c.cummax() - 1
    return float(dd.min() * 100)


def stock_risk_table(positions: pd.DataFrame, history: dict[str, pd.DataFrame],
                     benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    bench_ret = benchmark["Close"].pct_change(fill_method=None) if benchmark is not None and len(benchmark) > 30 else None
    for r in positions.itertuples():
        h = history.get(r.yahoo_symbol) if r.yahoo_symbol else None
        if h is None or len(h) < 20:
            rows.append({"Stock": r.base_symbol, "Yahoo Symbol": r.yahoo_symbol, "Days of data": 0 if h is None else len(h),
                         "Volatility (ann.) %": None, "Max Drawdown %": None, "Beta vs Nifty": None,
                         "1Y Return %": None})
            continue
        ret = h["Close"].pct_change(fill_method=None).dropna()
        beta = None
        if bench_ret is not None:
            j = pd.concat([ret, bench_ret], axis=1, join="inner").dropna()
            if len(j) > 30 and j.iloc[:, 1].var() > 0:
                beta = float(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var())
        rows.append({
            "Stock": r.base_symbol, "Yahoo Symbol": r.yahoo_symbol, "Days of data": len(h),
            "Volatility (ann.) %": float(ret.std() * math.sqrt(TRADING_DAYS) * 100),
            "Max Drawdown %": max_drawdown(h["Close"]),
            "Beta vs Nifty": beta,
            "1Y Return %": float((h["Close"].iloc[-1] / h["Close"].iloc[0] - 1) * 100),
        })
    return pd.DataFrame(rows)


def constant_holdings_value(positions: pd.DataFrame, history: dict[str, pd.DataFrame]) -> pd.Series:
    """Historical value of TODAY's quantities priced at each past Yahoo close.

    This is a hypothetical 'what the current holdings were worth' series – it
    ignores when lots were actually bought. Stocks without history are excluded.
    """
    parts = []
    for r in positions.itertuples():
        h = history.get(r.yahoo_symbol) if r.yahoo_symbol else None
        if h is not None and len(h):
            parts.append(h["Close"].rename(r.base_symbol) * r.quantity)
    if not parts:
        return pd.Series(dtype=float)
    df = pd.concat(parts, axis=1).sort_index().ffill().dropna()
    return df.sum(axis=1)


def portfolio_metrics(positions: pd.DataFrame, history: dict[str, pd.DataFrame],
                      benchmark: pd.DataFrame | None = None) -> dict:
    value = constant_holdings_value(positions, history)
    out = {"available": len(value) > 20}
    if not out["available"]:
        return out
    ret = value.pct_change(fill_method=None).dropna()
    out["volatility_ann_pct"] = float(ret.std() * math.sqrt(TRADING_DAYS) * 100)
    out["max_drawdown_pct"] = max_drawdown(value)
    out["worst_day_pct"] = float(ret.min() * 100)
    out["best_day_pct"] = float(ret.max() * 100)
    # Historical 1-day Value-at-Risk (95 %) on current holdings, from actual past returns
    out["var_95_1d_pct"] = float(-np.percentile(ret, 5) * 100)
    out["var_95_1d_value"] = float(out["var_95_1d_pct"] / 100 * value.iloc[-1])
    out["days"] = int(len(value))
    out["stocks_in_series"] = int(sum(1 for r in positions.itertuples()
                                      if r.yahoo_symbol and history.get(r.yahoo_symbol) is not None))
    if benchmark is not None and len(benchmark) > 30:
        b = benchmark["Close"].pct_change(fill_method=None)
        j = pd.concat([ret, b], axis=1, join="inner").dropna()
        if len(j) > 30 and j.iloc[:, 1].var() > 0:
            out["beta_vs_benchmark"] = float(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var())
            out["correlation_vs_benchmark"] = float(j.iloc[:, 0].corr(j.iloc[:, 1]))
    out["value_series"] = value
    return out


def diversification_notes(conc: dict, sectors: pd.DataFrame) -> list[str]:
    """Factual statements about composition (no judgement of safety)."""
    notes = []
    if not conc.get("available"):
        return ["Concentration needs live Yahoo prices."]
    notes.append(f"Largest holding: {conc['largest_symbol']} at {conc['largest_weight_pct']:.1f}% of priced value.")
    notes.append(f"Top 5 holdings make up {conc['top5_weight_pct']:.1f}% of priced value.")
    if conc.get("effective_holdings"):
        notes.append(f"Effective number of holdings (1/HHI): {conc['effective_holdings']:.1f} "
                     f"out of {conc['priced_holdings']} priced stocks.")
    if conc["holdings_over_10pct"]:
        notes.append("Holdings above 10% weight: " + ", ".join(conc["holdings_over_10pct"]) + ".")
    if sectors is not None and len(sectors):
        known = sectors[sectors["sector"] != "Unknown"]
        if len(known):
            top = known.iloc[0]
            notes.append(f"Largest sector: {top['sector']} at {top['weight_pct']:.1f}% "
                         f"across {int(top['holdings'])} holding(s).")
        unk = sectors[sectors["sector"] == "Unknown"]
        if len(unk):
            notes.append(f"{unk['weight_pct'].sum():.1f}% of value has no sector yet (fundamentals not loaded / N/A on Yahoo).")
    return notes
