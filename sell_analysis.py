"""
Sell Analysis – CALCULATION ONLY.

This module never places, schedules or sends any order and has no broker
connection. It answers "what would the numbers look like if I sold N shares
at the current Yahoo price (or at a price I type in)?".

Estimates exclude brokerage, STT, exchange charges, stamp duty, DP charges
and capital-gains tax.
"""
from __future__ import annotations

import math

import pandas as pd

ESTIMATE_LABEL = "ESTIMATE ONLY – no order is placed. Excludes brokerage, taxes and charges."


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _selection_payload(position: pd.Series | dict) -> dict:
    pos = dict(position)
    return {
        "account": pos.get("account"),
        "stock": pos.get("base_symbol") or pos.get("stock") or pos.get("symbol"),
        "yahoo_symbol": pos.get("yahoo_symbol") or pos.get("base_symbol"),
        "held_quantity": float(pos.get("quantity") or 0),
        "avg_buy_price": pos.get("avg_price"),
    }


def estimate_lot(lot: pd.Series | dict, quantity: float, current_price: float | None,
                price_label: str = "Current Yahoo price") -> dict:
    """Estimate selling one exact holding lot (not a stock-level aggregate)."""
    pos = dict(lot)
    held = float(pos.get("quantity") or 0)
    avg = pos.get("avg_price")
    out = {
        "account": pos.get("account"),
        "stock": pos.get("base_symbol") or pos.get("stock") or pos.get("symbol"),
        "yahoo_symbol": pos.get("yahoo_symbol") or pos.get("base_symbol"),
        "held_quantity": held,
        "sell_quantity": float(quantity) if quantity is not None else None,
        "avg_buy_price": avg,
        "price_used": current_price,
        "price_label": price_label,
        "label": ESTIMATE_LABEL,
        "valid": True,
        "message": None,
    }

    if out["sell_quantity"] is None or out["sell_quantity"] <= 0:
        out.update(valid=False, message="Enter a quantity greater than zero.")
        return out
    if out["sell_quantity"] > held:
        out.update(valid=False, message=f"Quantity {out['sell_quantity']:.g} exceeds holding of {held:.g}.")
        return out
    if not _ok(current_price):
        out.update(valid=False, message="Price unavailable from Yahoo Finance – cannot estimate.")
        return out

    invested = out["sell_quantity"] * avg if _ok(avg) else None
    sale_value = out["sell_quantity"] * current_price
    pnl = sale_value - invested if invested is not None else None
    remaining_qty = held - out["sell_quantity"]

    out.update({
        "estimated_sale_value": sale_value,
        "invested_amount": invested,
        "estimated_pnl": pnl,
        "return_pct": (pnl / invested * 100) if invested not in (None, 0) else None,
        "remaining_quantity": remaining_qty,
        "remaining_invested": remaining_qty * avg if _ok(avg) else None,
        "remaining_estimated_value": remaining_qty * current_price,
    })
    return out


def aggregate_lots(lots: pd.DataFrame) -> pd.DataFrame:
    """Summarize lots by stock without inventing partial cost or quote values."""
    columns = ["base_symbol", "accounts", "quantity", "weighted_avg_buy", "invested_value",
               "current_price", "yahoo_symbol"]
    if lots.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for base_symbol, group in lots.groupby("base_symbol", sort=True):
        quantity = float(group["quantity"].sum())
        known_costs = (group["cost_known"].fillna(False).astype(bool).all()
                       if "cost_known" in group else group["avg_price"].notna().all())
        invested = float(group["invested_value"].sum()) if known_costs else None
        average = invested / quantity if invested is not None and quantity > 0 else None
        symbols = group["yahoo_symbol"].dropna().unique()
        yahoo_symbol = symbols[0] if len(symbols) == 1 and group["yahoo_symbol"].notna().all() else None
        prices = group["current_price"].dropna().unique()
        current_price = (prices[0] if yahoo_symbol and group["current_price"].notna().all()
                         and len(prices) == 1 else None)
        rows.append({
            "base_symbol": base_symbol,
            "accounts": ", ".join(sorted(group["account"].dropna().unique().tolist())),
            "quantity": quantity,
            "weighted_avg_buy": average,
            "invested_value": invested,
            "current_price": current_price,
            "yahoo_symbol": yahoo_symbol,
        })
    return pd.DataFrame(rows, columns=columns)


def estimate_sale(position: pd.Series | dict, quantity: float, sell_price: float | None,
                  price_label: str = "Current Yahoo price") -> dict:
    """Estimate the outcome of selling `quantity` of an aggregated position.

    Cost basis uses the position's weighted average buy price
    (total invested ÷ total quantity across all lots).
    """
    pos = dict(position)
    held = float(pos.get("quantity") or 0)
    avg = pos.get("avg_price")
    out = {
        "stock": pos.get("base_symbol"), "yahoo_symbol": pos.get("yahoo_symbol"),
        "held_quantity": held, "sell_quantity": quantity, "avg_buy_price": avg,
        "price_used": sell_price, "price_label": price_label, "label": ESTIMATE_LABEL,
        "valid": True, "message": None,
    }
    if quantity is None or quantity <= 0:
        out.update(valid=False, message="Enter a quantity greater than zero.")
        return out
    if quantity > held:
        out.update(valid=False, message=f"Quantity {quantity:g} exceeds holding of {held:g}.")
        return out
    if not _ok(sell_price):
        out.update(valid=False, message="Price unavailable from Yahoo Finance – cannot estimate.")
        return out
    invested = quantity * avg if _ok(avg) else None
    sale_value = quantity * sell_price
    pnl = sale_value - invested if invested is not None else None
    remaining_qty = held - quantity
    out.update({
        "estimated_sale_value": sale_value,
        "invested_amount": invested,
        "estimated_pnl": pnl,
        "return_pct": (pnl / invested * 100) if invested else None,   # None for zero-cost → N/A
        "remaining_quantity": remaining_qty,
        "remaining_invested": remaining_qty * avg if _ok(avg) else None,
        "remaining_estimated_value": remaining_qty * sell_price,
    })
    return out


def analyse(position: pd.Series | dict, quantity: float, current_price: float | None,
            custom_price: float | None = None) -> dict:
    """Current-price estimate plus an optional custom-price estimate."""
    res = {"current": estimate_sale(position, quantity, current_price, "Current Yahoo price")}
    if custom_price is not None and custom_price > 0:
        res["custom"] = estimate_sale(position, quantity, custom_price, "Your custom price")
    return res


def remaining_portfolio_value(summary: dict, sell_quantity: float, current_price: float | None) -> float | None:
    """Market value of the remaining holdings after the sold shares leave the portfolio
    (sold shares valued at the current Yahoo price; sale proceeds/cash not included)."""
    if summary.get("current_value") is None or not _ok(current_price) or not sell_quantity:
        return None
    return summary["current_value"] - sell_quantity * current_price
