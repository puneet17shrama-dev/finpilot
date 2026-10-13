"""
Portfolio loading and valuation.

* Reads ONE authoritative file, config.HOLDINGS_FILE (data/holdings.xlsx), dynamically –
  nothing is hard-coded and no other file (e.g. a holdings.csv) is ever picked up implicitly.
* Understands broker exports that stack several accounts in one sheet with
  repeated header rows and summary rows in between (like the Kotak/Zerodha
  style export this project ships with), as well as a plain single table.
* Every valid row becomes a LOT (duplicates are preserved). A stock-level
  aggregated view is built on top of the lots.
* Broker columns such as LTP / Current Value in the sheet are NEVER used for
  valuation: current prices come only from Yahoo Finance. They are kept as
  `broker_ltp` for the data-quality panel so the difference between the broker's
  export-time value and the live Yahoo value can be explained.
* build_snapshot() is the single source of truth: dashboard, Telegram, PDF,
  alerts, scanner, risk, agent and sell analysis all consume the same snapshot.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config
from modules import database as db

log = logging.getLogger("finpilot.portfolio")

# ---------------------------------------------------------------- column detection
COLUMN_ALIASES = {
    "symbol": ["instrument", "symbol", "stock", "ticker", "scrip", "trading symbol",
               "tradingsymbol", "stock symbol", "security", "stock name", "company"],
    "quantity": ["qty", "quantity", "shares", "units", "qty.", "no of shares", "holding qty",
                 "quantity available", "net qty"],
    "avg_price": ["avg price", "average price", "avg. price", "avg cost", "avg. cost",
                  "average cost", "buy price", "buy avg", "purchase price", "cost price",
                  "avg buy price", "average buy price", "buy average"],
    "account": ["userid", "user id", "account", "client id", "clientid", "account id", "portfolio"],
    "yahoo_symbol": ["yahoo symbol", "yahoo", "yahoo ticker", "yf symbol"],
    # informational only – never used for valuation
    "broker_ltp": ["ltp", "last price", "last traded price", "cmp"],
}

# NSE/BSE series suffixes that appear in broker exports (e.g. ZAGGLE-EQ, GANESHBE-BE).
# Only these exact trailing codes are stripped, so names such as BAJAJ-AUTO survive.
SERIES_SUFFIXES = {"EQ", "BE", "BZ", "BL", "SM", "ST", "IL", "GB", "GS", "TB", "IT",
                   "RR", "MF", "E1", "X1", "T", "W", "Z"}


def _norm(text) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower()) if text is not None else ""


def _match_column(header_value) -> str | None:
    h = _norm(header_value)
    if not h or h == "nan":
        return None
    for key, aliases in COLUMN_ALIASES.items():
        if h in aliases:
            return key
    return None


def parse_number(value) -> float | None:
    """Parse numbers like '1,01,610.98', '₹ 1,234', '(123.4)', 95101.8 → float. None if invalid."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else f
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "-", "--", "n/a", "na"}:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"[₹,\s()%]", "", s).replace("Rs.", "").replace("INR", "")
    try:
        f = float(s)
    except ValueError:
        return None
    return -f if neg else f


def clean_symbol(raw) -> tuple[str, str]:
    """Return (portfolio_symbol, base_symbol).

    portfolio_symbol – as written in the sheet (trimmed, upper case)
    base_symbol      – series suffix removed, used for Yahoo resolution/aggregation
    """
    ps = str(raw).strip().upper()
    base = ps
    for sep in ("-", "."):
        if sep in base:
            head, tail = base.rsplit(sep, 1)
            if tail in SERIES_SUFFIXES and head:
                base = head
                break
    return ps, base.strip()


# ---------------------------------------------------------------- loading
@dataclass
class LoadResult:
    lots: pd.DataFrame
    invalid_rows: list[dict] = field(default_factory=list)
    source_file: str = ""
    signature: str = ""
    header_rows: int = 0
    error: str | None = None


LOT_COLUMNS = ["lot_id", "account", "portfolio_symbol", "base_symbol", "quantity",
               "avg_price", "invested_value", "zero_cost", "cost_known", "source_row", "sheet",
               "yahoo_symbol_override", "broker_ltp"]


def file_signature(path: Path) -> str:
    stat = path.stat()
    h = hashlib.sha1(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _read_raw_sheets(path: Path) -> dict[str, pd.DataFrame]:
    if path.suffix.lower() == ".csv":
        return {"csv": pd.read_csv(path, header=None, dtype=object)}
    errors: list[Exception] = []
    for engine in ("openpyxl", None):
        try:
            return pd.read_excel(path, sheet_name=None, header=None, dtype=object, engine=engine) if engine else \
                pd.read_excel(path, sheet_name=None, header=None, dtype=object)
        except Exception as exc:  # try the default engine next if available, but keep the final root cause
            errors.append(exc)
    raise errors[-1]


def load_holdings(path: Path | None = None) -> LoadResult:
    path = Path(path or config.HOLDINGS_FILE)
    if path.suffix.lower() == ".csv" and not config.ALLOW_CSV_HOLDINGS:
        return LoadResult(pd.DataFrame(columns=LOT_COLUMNS), source_file=str(path),
                          error=f"{path.name} is a CSV. The authoritative portfolio is an Excel file "
                                "(HOLDINGS_FILE=data/holdings.xlsx). Set ALLOW_CSV_HOLDINGS=true to use a CSV "
                                "deliberately.")
    if not path.exists():
        return LoadResult(pd.DataFrame(columns=LOT_COLUMNS), error=f"Holdings file not found: {path}")
    try:
        sheets = _read_raw_sheets(path)
    except ImportError as exc:
        return LoadResult(pd.DataFrame(columns=LOT_COLUMNS),
                          error=f"Could not read {path.name}: Excel support is missing. Install the project requirements (pip install -r requirements.txt) or ensure openpyxl is installed.")
    except PermissionError as exc:
        return LoadResult(pd.DataFrame(columns=LOT_COLUMNS),
                          error=f"Could not read {path.name}: the workbook is probably open in Excel or locked by another process. Close it and retry.")
    except Exception as exc:  # corrupt / locked file
        msg = str(exc)
        if "openpyxl" in msg.lower() or "xlsx" in msg.lower() or "excel" in msg.lower():
            msg = "Excel engine issue or invalid workbook. Close the file and verify it is a valid .xlsx file."
        return LoadResult(pd.DataFrame(columns=LOT_COLUMNS), error=f"Could not read {path.name}: {msg}")

    lots, invalid, headers_found = [], [], 0
    for sheet_name, raw in sheets.items():
        colmap: dict[int, str] | None = None
        for ridx in range(len(raw)):
            row = raw.iloc[ridx].tolist()
            detected = {i: _match_column(v) for i, v in enumerate(row)}
            detected = {i: k for i, k in detected.items() if k}
            keys = set(detected.values())
            if {"symbol", "quantity"} <= keys:
                colmap = detected          # a (new) header row – next block starts
                headers_found += 1
                continue
            if colmap is None:
                continue
            values = {k: row[i] if i < len(row) else None for i, k in colmap.items()}
            excel_row = ridx + 1
            if all(v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == ""
                   for v in row):
                continue  # blank row – silently skipped
            sym_raw = values.get("symbol")
            qty = parse_number(values.get("quantity"))
            avg = parse_number(values.get("avg_price"))
            sym_ok = sym_raw is not None and str(sym_raw).strip() not in {"", "nan", "None"}
            reason = None
            sym_txt = str(sym_raw).strip() if sym_ok else ""
            is_label = sym_ok and (" " in sym_txt or parse_number(sym_txt) is not None
                                   or not re.search(r"[A-Za-z]", sym_txt))
            if not sym_ok and qty is None and avg is None:
                reason = "Not a holding (summary/label row)"
            elif not sym_ok:
                reason = "Missing symbol"
            elif is_label:
                reason = "Not a symbol (summary/label row)"
            elif qty is None:
                reason = "Missing/invalid quantity"
            elif qty <= 0:
                reason = "Quantity is zero or negative"
            elif avg is not None and avg < 0:
                reason = "Negative average price"
            if reason:
                invalid.append({"sheet": sheet_name, "row": excel_row, "reason": reason,
                                "content": " | ".join(str(v) for v in row if str(v) != "nan")[:160]})
                continue
            ps, base = clean_symbol(sym_raw)
            account = values.get("account")
            account = str(account).strip() if account is not None and str(account) != "nan" else "Default"
            yov = values.get("yahoo_symbol")
            yov = str(yov).strip() if yov is not None and str(yov).strip() not in {"", "nan"} else None
            lots.append({
                "lot_id": f"{sheet_name}:{excel_row}",
                "account": account,
                "portfolio_symbol": ps,
                "base_symbol": base,
                "quantity": float(qty),
                "avg_price": float(avg) if avg is not None else np.nan,
                "invested_value": float(qty) * float(avg) if avg is not None else np.nan,
                "zero_cost": avg == 0.0,
                "cost_known": avg is not None,
                "source_row": excel_row,
                "sheet": sheet_name,
                "yahoo_symbol_override": yov,
                "broker_ltp": parse_number(values.get("broker_ltp")),
            })
            if avg is None:
                log.warning("Row %s (%s): missing/invalid average price – lot kept, cost unknown, "
                            "excluded from invested value and P&L", excel_row, ps)

    df = pd.DataFrame(lots, columns=LOT_COLUMNS)
    result = LoadResult(df, invalid, str(path), file_signature(path), headers_found)
    if headers_found == 0:
        result.error = ("No header row found. The sheet needs at least a symbol column "
                        "(e.g. 'Instrument'/'Symbol') and a quantity column ('Qty'/'Quantity').")
    _persist_lots(result)
    return result


def _persist_lots(result: LoadResult) -> None:
    """Mirror the current Excel lots into SQLite (replaced atomically only when the file changed)."""
    if result.lots.empty or not result.signature:
        return
    try:
        if db.get_meta("holdings_signature") == result.signature:
            return
        now = db.now_iso()
        rows = [(r.lot_id, r.account, r.portfolio_symbol, r.base_symbol, r.quantity, r.avg_price,
                 r.invested_value, int(r.source_row), result.source_file, result.signature, now)
                for r in result.lots.itertuples()]
        with db.get_conn() as conn:
            conn.execute("DELETE FROM holdings_lots")
            conn.executemany(
                "INSERT INTO holdings_lots(lot_id, account, portfolio_symbol, base_symbol, quantity, "
                "avg_price, invested_value, source_row, source_file, file_signature, imported_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
        db.set_meta("holdings_signature", result.signature)
    except Exception:  # persistence must never break the dashboard
        log.exception("Could not mirror holdings into SQLite")


# ---------------------------------------------------------------- valuation
# Formulas (per lot, then summed – never re-multiplying an aggregated value):
#   invested_value = quantity × avg_price                    (only when the cost is known)
#   current_value  = quantity × Yahoo price                  (NaN when Yahoo has no price – never 0)
#   pnl            = current_value − invested_value          (only when both are known)
#   return_pct     = pnl ÷ invested_value × 100              (N/A when invested is 0 / unknown)
#   todays_pnl     = quantity × (price − previous_close)     (NaN when previous close is unknown)
#   day_change_pct = (price ÷ previous_close − 1) × 100
FLOAT_TOLERANCE = 0.05          # ₹ – allowed rounding difference in reconciliation checks
UNPRICED_STATUSES = {"UNAVAILABLE", "UNRESOLVED", "SYMBOL NOT FOUND", "PENDING"}


def _safe_div(a, b):
    if a is None or b is None:
        return None
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if b == 0 or math.isnan(a) or math.isnan(b):
        return None
    v = a / b
    return None if math.isnan(v) or math.isinf(v) else v


def _valid_price(x) -> float:
    """A usable Yahoo price is a finite number > 0. Anything else → NaN (never 0)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if math.isfinite(v) and v > 0 else np.nan


def default_series(df: pd.DataFrame, value) -> pd.Series:
    """A column filled with `value` that has exactly df's index/length.

    Fixes the old `pd.Series(["UNAVAILABLE"], index=df.index)` bug: a one-element list
    cannot be aligned to an index of length N ("Length of values (1) does not match
    length of index (N)"). A scalar is broadcast by pandas to every row instead."""
    return pd.Series(value, index=df.index, dtype=object if isinstance(value, str) else None)


VALUED_COLUMNS = ["yahoo_symbol", "current_price", "previous_close", "current_value", "pnl",
                  "return_pct", "todays_pnl", "day_change_pct", "price_status", "price_timestamp",
                  "is_priced"]


def value_lots(lots: pd.DataFrame, symbol_map: dict[str, str], quotes: dict[str, dict]) -> pd.DataFrame:
    """Attach Yahoo prices to each lot. Missing prices stay NaN and the lot is marked
    unavailable – quantity and invested value are always preserved."""
    df = lots.copy()
    if df.empty:
        for c in VALUED_COLUMNS:
            df[c] = pd.Series(dtype=object)
        return df
    if "cost_known" not in df:
        df["cost_known"] = df["avg_price"].notna()
    override = df["yahoo_symbol_override"] if "yahoo_symbol_override" in df else default_series(df, None)
    df["yahoo_symbol"] = [ov if isinstance(ov, str) and ov else symbol_map.get(b)
                          for ov, b in zip(override, df["base_symbol"])]

    def quote(sym) -> dict:
        return (quotes.get(sym) or {}) if sym else {}

    df["current_price"] = [_valid_price(quote(s).get("price")) for s in df["yahoo_symbol"]]
    df["previous_close"] = [_valid_price(quote(s).get("previous_close")) for s in df["yahoo_symbol"]]
    df["price_timestamp"] = [quote(s).get("data_timestamp") for s in df["yahoo_symbol"]]
    df["is_priced"] = df["current_price"].notna()
    statuses = []
    for sym, priced in zip(df["yahoo_symbol"], df["is_priced"]):
        if not sym:
            statuses.append("UNRESOLVED")
        elif not priced:
            statuses.append("UNAVAILABLE")
        else:
            statuses.append((quote(sym).get("status") or "Live").upper() if quote(sym).get("status") else "LIVE")
    df["price_status"] = statuses
    df["current_value"] = df["quantity"] * df["current_price"]                     # NaN if unpriced
    df["pnl"] = df["current_value"] - df["invested_value"]                         # NaN if either unknown
    df["return_pct"] = np.where(df["invested_value"] > 0, df["pnl"] / df["invested_value"] * 100, np.nan)
    df["todays_pnl"] = df["quantity"] * (df["current_price"] - df["previous_close"])
    df["day_change_pct"] = (df["current_price"] / df["previous_close"] - 1) * 100
    return df


def aggregate_positions(valued_lots: pd.DataFrame) -> pd.DataFrame:
    """One row per stock (across lots/accounts), built by SUMMING lot values.

    Money columns use sum(min_count=1) so a stock whose lots are all unpriced stays NaN
    instead of becoming ₹0. `avg_price` = invested ÷ quantity over lots with a known cost."""
    cols = ["base_symbol", "portfolio_symbol", "yahoo_symbol", "accounts", "lots", "quantity",
            "invested_value", "avg_price", "current_price", "previous_close", "current_value",
            "pnl", "return_pct", "todays_pnl", "day_change_pct", "has_zero_cost_lot", "price_status",
            "priced_lots", "price_timestamp"]
    if valued_lots.empty:
        return pd.DataFrame(columns=cols)
    df = valued_lots.copy()
    if "cost_known" not in df:
        df["cost_known"] = df["avg_price"].notna()
    df["_qty_cost_known"] = df["quantity"].where(df["cost_known"])
    g = df.groupby("base_symbol", sort=False)

    def first_valid(s: pd.Series):
        s = s.dropna()
        return s.iloc[0] if len(s) else np.nan

    def status(s: pd.Series):
        vals = list(dict.fromkeys(s.dropna()))
        return vals[0] if len(vals) == 1 else "Mixed: " + "/".join(vals)

    out = pd.DataFrame({
        "portfolio_symbol": g["portfolio_symbol"].agg(lambda s: ", ".join(sorted(set(s)))),
        "yahoo_symbol": g["yahoo_symbol"].agg(first_valid),
        "accounts": g["account"].agg(lambda s: ", ".join(sorted(set(s)))),
        "lots": g.size(),
        "priced_lots": g["is_priced"].sum(),
        "quantity": g["quantity"].sum(),
        "invested_value": g["invested_value"].sum(min_count=1),
        "_qty_cost_known": g["_qty_cost_known"].sum(min_count=1),
        "current_price": g["current_price"].agg(first_valid),
        "previous_close": g["previous_close"].agg(first_valid),
        "current_value": g["current_value"].sum(min_count=1),
        "pnl": g["pnl"].sum(min_count=1),
        "todays_pnl": g["todays_pnl"].sum(min_count=1),
        "has_zero_cost_lot": g["zero_cost"].any(),
        "price_status": g["price_status"].agg(status),
        "price_timestamp": g["price_timestamp"].agg(first_valid),
    }).reset_index()
    out["avg_price"] = out["invested_value"] / out["_qty_cost_known"].replace(0, np.nan)
    # Return % over the lots that have BOTH a price and a known cost (pnl basis).
    basis = df[df["pnl"].notna()].groupby("base_symbol")["invested_value"].sum()
    out["return_pct"] = [(_safe_div(p * 100, basis.get(b)) if pd.notna(p) else None)
                         for b, p in zip(out["base_symbol"], out["pnl"])]
    out["return_pct"] = out["return_pct"].astype(float)
    out["day_change_pct"] = (out["current_price"] / out["previous_close"] - 1) * 100
    return out[cols]


def compute_summary(valued_lots: pd.DataFrame) -> dict:
    """Portfolio totals from the LOTS – the only place totals are calculated.

    Unpriced lots are excluded from current value / P&L (never counted as zero) and
    reported in coverage fields, so a partial valuation is never shown as complete."""
    lots = valued_lots
    n = int(len(lots))
    empty = {"invested_value": 0.0, "invested_priced": 0.0, "current_value": None, "total_pnl": None,
             "return_pct": None, "todays_pnl": None, "todays_pct": None, "total_lots": 0,
             "total_holdings": 0, "unique_stocks": 0, "priced_holdings": 0, "unavailable_holdings": 0,
             "priced_stocks": 0, "unavailable_stocks": 0, "unavailable_symbols": [],
             "coverage_pct": None, "value_coverage_pct": None, "todays_pnl_holdings": 0, "stale_holdings": 0,
             "is_partial": False, "cost_unknown_lots": 0, "accounts": [], "status": "NO DATA"}
    if n == 0:
        return empty
    priced = lots[lots["is_priced"]]
    unpriced = lots[~lots["is_priced"]]
    cost_known = (lots["cost_known"].fillna(False).astype(bool)
                  if "cost_known" in lots else lots["avg_price"].notna())
    pnl_lots = lots[lots["pnl"].notna()]
    today_lots = lots[lots["todays_pnl"].notna()]
    invested_all = float(lots.loc[cost_known, "invested_value"].sum())
    invested_priced = float(pnl_lots["invested_value"].sum())
    current = float(priced["current_value"].sum()) if len(priced) else None
    total_pnl = float(pnl_lots["pnl"].sum()) if len(pnl_lots) else None
    todays = float(today_lots["todays_pnl"].sum()) if len(today_lots) else None
    prev_value = float((today_lots["quantity"] * today_lots["previous_close"]).sum()) if len(today_lots) else None
    stock_priced = lots.groupby("base_symbol")["is_priced"].all()
    unavailable_syms = sorted(unpriced["portfolio_symbol"].unique().tolist())
    coverage = len(priced) / n * 100
    status = "COMPLETE" if len(priced) == n else ("UNAVAILABLE" if len(priced) == 0 else "PARTIAL")
    return {
        "invested_value": invested_all,          # all lots with a known cost
        "invested_priced": invested_priced,      # cost basis of lots that also have a Yahoo price
        "current_value": current,                # priced lots only
        "total_pnl": total_pnl,
        "return_pct": _safe_div(total_pnl * 100 if total_pnl is not None else None, invested_priced),
        "todays_pnl": todays,
        "todays_pct": _safe_div(todays * 100 if todays is not None else None, prev_value),
        "todays_pnl_holdings": int(len(today_lots)),
        "total_lots": n,
        "total_holdings": n,                     # holdings = valid lots (rows) in holdings.xlsx
        "unique_stocks": int(lots["base_symbol"].nunique()),
        "priced_holdings": int(len(priced)),
        "unavailable_holdings": int(len(unpriced)),
        "priced_stocks": int(stock_priced.sum()),
        "unavailable_stocks": int((~stock_priced).sum()),
        "unavailable_symbols": unavailable_syms,
        "coverage_pct": coverage,
        "value_coverage_pct": _safe_div(invested_priced * 100, invested_all),
        "is_partial": len(priced) < n,
        "stale_holdings": int(priced["price_status"].astype(str).str.startswith("Stale").sum()),
        "cost_unknown_lots": int((~cost_known).sum()),
        "accounts": sorted(lots["account"].dropna().unique().tolist()),
        "status": status,
    }


def portfolio_summary(positions_or_lots: pd.DataFrame) -> dict:
    """Backward-compatible alias. Totals are always computed from lots when possible."""
    if "is_priced" in positions_or_lots.columns and "lot_id" in positions_or_lots.columns:
        return compute_summary(positions_or_lots)
    raise ValueError("portfolio_summary() now needs valued LOTS – use compute_summary(snapshot['lots'])")


def account_breakdown(valued_lots: pd.DataFrame) -> pd.DataFrame:
    """Per-account totals using exactly the same rules as compute_summary()."""
    cols = ["Account", "Invested Value", "Current Value", "P&L", "Return %", "Today's P&L",
            "Holdings", "Unique Stocks", "Priced", "Unavailable", "Coverage %"]
    if valued_lots.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for acct, g in valued_lots.groupby("account", sort=True):
        s = compute_summary(g)
        rows.append({"Account": acct, "Invested Value": s["invested_value"], "Current Value": s["current_value"],
                     "P&L": s["total_pnl"], "Return %": s["return_pct"], "Today's P&L": s["todays_pnl"],
                     "Holdings": s["total_lots"], "Unique Stocks": s["unique_stocks"],
                     "Priced": s["priced_holdings"], "Unavailable": s["unavailable_holdings"],
                     "Coverage %": s["coverage_pct"]})
    return pd.DataFrame(rows, columns=cols)


def lot_diagnostics(valued_lots: pd.DataFrame) -> dict:
    """Kept for the existing dashboard panels – values come from compute_summary()."""
    s = compute_summary(valued_lots)
    cols = ["account", "portfolio_symbol", "base_symbol", "yahoo_symbol", "quantity", "price_status"]
    unavailable = valued_lots.loc[~valued_lots["is_priced"], cols] if len(valued_lots) else pd.DataFrame(columns=cols)
    return {
        "holdings": s["total_lots"],
        "priced_holdings": s["priced_holdings"],
        "unavailable_holdings": s["unavailable_holdings"],
        "total_quantity": float(valued_lots["quantity"].sum()) if len(valued_lots) else 0.0,
        "total_value": s["current_value"],
        "unavailable": unavailable.to_dict("records"),
    }


def _display_path(path: str) -> str:
    if not path:
        return ""
    try:
        return Path(path).resolve().relative_to(config.BASE_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


def data_quality(load: LoadResult, valued_lots: pd.DataFrame, resolution: dict, quotes: dict) -> dict:
    """Facts for the data-validation panel – makes value discrepancies traceable."""
    reasons = [r["reason"] for r in load.invalid_rows]
    lots = valued_lots
    dup = (lots.groupby(["account", "base_symbol"]).size() if len(lots) else pd.Series(dtype=int))
    duplicate_groups = dup[dup > 1]
    multi_acct = (lots.groupby("base_symbol")["account"].nunique() if len(lots) else pd.Series(dtype=int))
    failed = sorted({s for s, q in quotes.items() if q.get("price") is None or pd.isna(_valid_price(q.get("price")))})
    broker_value = None
    if len(lots) and "broker_ltp" in lots and lots["broker_ltp"].notna().all():
        broker_value = float((lots["quantity"] * lots["broker_ltp"]).sum())
    return {
        "source_file": _display_path(load.source_file),
        "header_rows": load.header_rows,
        "excluded_rows": len(load.invalid_rows),
        "invalid_quantities": sum("quantity" in r.lower() for r in reasons),
        "invalid_buy_prices": sum("price" in r.lower() for r in reasons) + int((~lots["cost_known"]).sum() if len(lots) else 0),
        "missing_symbols": sum(r == "Missing symbol" for r in reasons),
        "label_rows": sum("summary/label" in r for r in reasons),
        "duplicate_rows": int((duplicate_groups - 1).sum()) if len(duplicate_groups) else 0,
        "duplicate_symbols_same_account": sorted(f"{a}:{b}" for a, b in duplicate_groups.index),
        "symbols_in_multiple_accounts": sorted(multi_acct[multi_acct > 1].index.tolist()),
        "unresolved_symbols": sorted(b for b, r in resolution.items() if not r.get("yahoo_symbol")),
        "yahoo_failed_symbols": failed,
        "yahoo_ok": len(quotes) - len(failed),
        "yahoo_failed": len(failed),
        "broker_file_value": broker_value,
    }


# ---------------------------------------------------------------- validation
def validate_snapshot(snap: dict) -> list[dict]:
    """Self-checks run on every snapshot. Any failure is shown prominently in the UI/reports."""
    checks: list[dict] = []
    lots, pos, s, acc = snap["lots"], snap["positions"], snap["summary"], snap["accounts_table"]
    tol = FLOAT_TOLERANCE

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    def close(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return abs(float(a) - float(b)) <= tol + 1e-9 * max(abs(float(a)), abs(float(b)))

    if len(lots):
        priced = lots[lots["is_priced"]]
        err = (priced["current_value"] - priced["quantity"] * priced["current_price"]).abs().max() if len(priced) else 0
        add("Stock current value = qty × price", (err or 0) <= tol, f"max error ₹{err or 0:.4f}")
        ck = lots[lots["cost_known"]]
        err = (ck["invested_value"] - ck["quantity"] * ck["avg_price"]).abs().max() if len(ck) else 0
        add("Stock invested value = qty × buy price", (err or 0) <= tol, f"max error ₹{err or 0:.4f}")
        pl = lots[lots["pnl"].notna()]
        err = (pl["pnl"] - (pl["current_value"] - pl["invested_value"])).abs().max() if len(pl) else 0
        add("Stock P&L = current − invested", (err or 0) <= tol, f"max error ₹{err or 0:.4f}")
        tp = lots[lots["todays_pnl"].notna()]
        err = (tp["todays_pnl"] - tp["quantity"] * (tp["current_price"] - tp["previous_close"])).abs().max() if len(tp) else 0
        add("Today's P&L = qty × (price − prev close)", (err or 0) <= tol, f"max error ₹{err or 0:.4f}")
        add("Unpriced lots are not valued at zero",
            lots.loc[~lots["is_priced"], "current_value"].isna().all()
            and not (lots["current_price"] == 0).any(), f"{s['unavailable_holdings']} unpriced")
    add("Invested total = Σ lot invested",
        close(s["invested_value"], float(lots.loc[lots["cost_known"], "invested_value"].sum()) if len(lots) else 0.0))
    add("Current total = Σ priced lot values",
        close(s["current_value"], float(lots.loc[lots["is_priced"], "current_value"].sum()) if s["priced_holdings"] else None))
    if s["total_pnl"] is not None and s["current_value"] is not None:
        add("Total P&L = current − invested (priced lots)",
            close(s["total_pnl"], s["current_value"] - s["invested_priced"]) or s["cost_unknown_lots"] > 0,
            f"{s['total_pnl']:.2f} vs {s['current_value'] - s['invested_priced']:.2f}")
    if s["return_pct"] is not None:
        add("Return % = P&L ÷ invested × 100",
            abs(s["return_pct"] - s["total_pnl"] / s["invested_priced"] * 100) < 1e-6)
    if len(pos):
        add("Stock-level totals = portfolio totals",
            close(pos["current_value"].sum(min_count=1) if s["priced_holdings"] else None, s["current_value"])
            and close(pos["invested_value"].sum(), s["invested_value"]))
        add("Quantities not duplicated by aggregation", close(pos["quantity"].sum(), lots["quantity"].sum()))
    if len(acc):
        def acc_sum(col):
            v = acc[col].dropna()
            return float(v.sum()) if len(v) else None
        add("Σ account invested = portfolio invested", close(acc_sum("Invested Value"), s["invested_value"]))
        add("Σ account current = portfolio current", close(acc_sum("Current Value"), s["current_value"]))
        add("Σ account P&L = portfolio P&L", close(acc_sum("P&L"), s["total_pnl"]))
        add("Σ account today's P&L = portfolio today's P&L", close(acc_sum("Today's P&L"), s["todays_pnl"]))
        add("Σ account holdings = portfolio holdings", int(acc["Holdings"].sum()) == s["total_lots"])
    loaded = snap["load"].lots
    loaded = loaded[loaded["account"].isin(s["accounts"])] if len(loaded) else loaded
    add("Holdings count = rows loaded from file", s["total_lots"] == len(loaded))
    return checks


def record_portfolio_snapshot(summary: dict, bucket: str) -> None:
    """One row per minute bucket – repeated 10 s refreshes overwrite, never duplicate."""
    if summary.get("current_value") is None:
        return
    try:
        db.execute(
            "INSERT INTO portfolio_snapshots(bucket, invested_value, current_value, total_pnl, todays_pnl, "
            "priced_holdings, total_holdings, recorded_at) VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(bucket) DO UPDATE SET invested_value=excluded.invested_value, "
            "current_value=excluded.current_value, total_pnl=excluded.total_pnl, "
            "todays_pnl=excluded.todays_pnl, priced_holdings=excluded.priced_holdings, "
            "total_holdings=excluded.total_holdings, recorded_at=excluded.recorded_at",
            (bucket, summary["invested_priced"], summary["current_value"], summary["total_pnl"],
             summary["todays_pnl"], summary["priced_holdings"], summary["total_holdings"], db.now_iso()))
    except Exception:
        log.exception("Could not record portfolio snapshot")


# ---------------------------------------------------------------- live snapshot (used by every component)
_HOLDINGS_CACHE: dict = {"signature": None, "result": None}


def get_holdings(force: bool = False) -> LoadResult:
    """Load HOLDINGS_FILE, re-reading whenever its size/modification time changes.
    The previous result is replaced, never appended to, so refreshes can't duplicate lots."""
    path = Path(config.HOLDINGS_FILE)
    sig = file_signature(path) if path.exists() else None
    if force or sig is None or sig != _HOLDINGS_CACHE["signature"]:
        _HOLDINGS_CACHE["result"] = load_holdings(path)
        _HOLDINGS_CACHE["signature"] = sig
        res = _HOLDINGS_CACHE["result"]
        log.info("Holdings loaded from %s: %d lots, %d unique stocks, %d rows excluded",
                 path, len(res.lots), res.lots["base_symbol"].nunique() if len(res.lots) else 0,
                 len(res.invalid_rows))
    return _HOLDINGS_CACHE["result"]


def finalize_positions(positions: pd.DataFrame, resolution: dict) -> pd.DataFrame:
    """Attach symbol-resolution status and label unpriced rows honestly."""
    positions = positions.copy()
    if positions.empty:
        positions["resolution_status"] = pd.Series(dtype=object)
        return positions
    positions["resolution_status"] = [resolution.get(b, {}).get("status", "Pending")
                                      for b in positions["base_symbol"]]
    mask = positions["yahoo_symbol"].isna()
    positions.loc[mask, "price_status"] = [
        "SYMBOL NOT FOUND" if st_ == "Not found" else "UNRESOLVED"
        for st_ in positions.loc[mask, "resolution_status"]]
    return positions


def _assemble(base: dict, lots: pd.DataFrame) -> dict:
    """Build summary / positions / accounts / validation from one lots frame."""
    positions = finalize_positions(aggregate_positions(lots), base["resolution"])
    summary = compute_summary(lots)
    snap = {**base, "lots": lots, "positions": positions, "summary": summary,
            "accounts_table": account_breakdown(lots), "diagnostics": lot_diagnostics(lots)}
    snap["quality"] = data_quality(base["load"], lots, base["resolution"], base["quotes"])
    snap["validation"] = validate_snapshot(snap)
    snap["valid"] = all(c["ok"] for c in snap["validation"])
    if not snap["valid"]:
        log.error("Snapshot validation FAILED: %s",
                  "; ".join(f"{c['check']} ({c['detail']})" for c in snap["validation"] if not c["ok"]))
    return snap


def filter_snapshot(snap: dict, accounts: list[str] | None) -> dict:
    """Same snapshot restricted to some accounts (same lots, same prices, same rules)."""
    all_accounts = snap.get("all_accounts") or snap["summary"].get("accounts", [])
    if not accounts or set(accounts) >= set(all_accounts):
        return snap
    lots = snap["lots"][snap["lots"]["account"].isin(accounts)]
    return _assemble(snap, lots)


def build_snapshot(force_quotes: bool = False, network: bool = True) -> dict:
    """THE portfolio snapshot (single source of truth) computed from HOLDINGS_FILE + Yahoo.

    Cheap to call every 10 s: holdings are re-read only when the file changes and quotes
    are cached for MARKET_REFRESH_SECONDS. network=False → read-only (symbol map from SQLite,
    quote cache filled by the background monitor) so the dashboard never blocks on Yahoo."""
    from modules import market_data as md

    load = get_holdings()
    lots = load.lots
    base_to_ps = (lots.drop_duplicates("base_symbol").set_index("base_symbol")["portfolio_symbol"].to_dict()
                  if not lots.empty else {})
    resolution = md.resolve_symbols(base_to_ps, allow_network=network) if base_to_ps else {}
    symbol_map = {b: r["yahoo_symbol"] for b, r in resolution.items() if r["yahoo_symbol"]}
    overrides = lots["yahoo_symbol_override"].dropna().tolist() if "yahoo_symbol_override" in lots else []
    yahoo_syms = sorted(set(symbol_map.values()) | set(overrides))
    if not yahoo_syms:
        quotes = {}
    elif network or force_quotes:
        quotes = md.get_quotes(yahoo_syms, force=force_quotes)
    else:
        quotes = md.cached_quotes(yahoo_syms)
    valued = value_lots(lots, symbol_map, quotes)
    now = md.now_ist()
    market = md.market_status(now)
    yahoo_ts = md.latest_data_timestamp(quotes)
    base = {"generated_at": now, "load": load, "resolution": resolution, "quotes": quotes,
            "market": market, "data_status": md.data_status(), "yahoo_timestamp": yahoo_ts,
            "price_basis": md.price_basis(market, yahoo_ts, now),
            "all_accounts": sorted(lots["account"].unique().tolist()) if not lots.empty else []}
    snap = _assemble(base, valued)
    s = snap["summary"]
    if network:
        record_portfolio_snapshot(s, now.strftime("%Y-%m-%d %H:%M"))
        log.info("Snapshot: %s lots=%d priced=%d unavailable=%d invested=%.2f current=%s pnl=%s today=%s "
                 "yahoo_ts=%s unavailable_symbols=%s", s["status"], s["total_lots"], s["priced_holdings"],
                 s["unavailable_holdings"], s["invested_value"], s["current_value"], s["total_pnl"],
                 s["todays_pnl"], yahoo_ts, ", ".join(s["unavailable_symbols"]) or "none")
    return snap


def symbol_price_log(snap: dict) -> pd.DataFrame:
    """Original symbol → Yahoo symbol → price → status, one row per stock."""
    p = snap["positions"]
    if p.empty:
        return pd.DataFrame(columns=["Original", "Yahoo symbol", "Price", "Previous close", "Status", "Resolution"])
    res = snap["resolution"]
    return pd.DataFrame({
        "Original": p["portfolio_symbol"], "Yahoo symbol": p["yahoo_symbol"].fillna("—"),
        "Price": p["current_price"], "Previous close": p["previous_close"], "Status": p["price_status"],
        "Resolution": [res.get(b, {}).get("method", "") for b in p["base_symbol"]],
    })


def find_position(positions: pd.DataFrame, text: str) -> pd.Series | None:
    """Find a holding by portfolio symbol, base symbol or Yahoo symbol (case-insensitive)."""
    if positions.empty or not text:
        return None
    t = text.strip().upper()
    for col in ("base_symbol", "yahoo_symbol"):
        hit = positions[positions[col].fillna("").str.upper() == t]
        if len(hit):
            return hit.iloc[0]
    hit = positions[positions["portfolio_symbol"].str.upper().str.split(", ").apply(lambda xs: t in xs)]
    return hit.iloc[0] if len(hit) else None


def top_movers(positions: pd.DataFrame, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gainers/decliners from the snapshot's stock rows, ranked by day_change_pct.

    Only stocks with a valid price AND previous close are ranked, and only prices from the
    most recent Yahoo session in the snapshot – an older (stale) day's move is never mixed
    with today's moves."""
    if positions.empty:
        return positions, positions
    ok = positions[positions["current_price"].notna() & positions["previous_close"].notna()
                   & positions["day_change_pct"].notna()
                   & ~positions["price_status"].astype(str).str.startswith(("Mixed", "UNAVAILABLE", "UNRESOLVED"))]
    if "price_timestamp" in ok and ok["price_timestamp"].notna().any():
        days = pd.to_datetime(ok["price_timestamp"], utc=True).dt.tz_convert(config.TIMEZONE).dt.date
        ok = ok[days == days.max()]
    ranked = ok.sort_values("day_change_pct", ascending=False)
    gainers = ranked[ranked["day_change_pct"] > 0].head(n)
    decliners = ranked[ranked["day_change_pct"] < 0].sort_values("day_change_pct").head(n)
    return gainers, decliners
