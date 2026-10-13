"""
Reports: Telegram message texts and PDF reports (ReportLab).

* Every number comes from the live snapshot (Excel holdings + Yahoo data).
  Nothing is hard-coded; unavailable values print as "N/A".
* PDFs are generated only on request or on schedule (never on the 10 s
  refresh) and saved in reports/ with metadata in SQLite.
"""
from __future__ import annotations

import html
import logging
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from modules import database as db
from modules import market_data as md

CUR = config.CURRENCY_SYMBOL
log = logging.getLogger("finpilot.reports")


# ================================================================ formatting helpers
def _isnum(x) -> bool:
    return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def indian_group(n: float, decimals: int = 2) -> str:
    """12345678.9 → '1,23,45,678.90' (Indian digit grouping)."""
    neg = n < 0
    s = f"{abs(n):.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        whole = ",".join(parts + [tail])
    return ("-" if neg else "") + whole + (f".{frac}" if decimals else "")


def inr(x, decimals: int = 2, signed: bool = False) -> str:
    if not _isnum(x):
        return "N/A"
    sign = "+" if signed and x > 0 else ""
    body = indian_group(x, decimals)
    if body.startswith("-"):
        return f"-{CUR}{body[1:]}"
    return f"{sign}{CUR}{body}"


def pct(x, signed: bool = True) -> str:
    if not _isnum(x):
        return "N/A"
    return f"{x:+.2f}%" if signed else f"{x:.2f}%"


def num(x, decimals: int = 2) -> str:
    return f"{x:,.{decimals}f}" if _isnum(x) else "N/A"


def _e(text) -> str:
    from modules.alerts import esc
    return esc(text)


def _arrow(x) -> str:
    return "🟢" if _isnum(x) and x > 0 else "🔴" if _isnum(x) and x < 0 else "⚪"


# ================================================================ Telegram texts
# Every message is built from ONE snapshot (portfolio.build_snapshot) – the same object the
# dashboard and the PDF use – so all channels show identical numbers.
def coverage_lines(snap: dict, html_mode: bool = True) -> list[str]:
    """Data-coverage / price-basis block shared by every report."""
    s, basis = snap["summary"], snap.get("price_basis") or {}
    b = (lambda t: f"<b>{t}</b>") if html_mode else (lambda t: t)
    esc = _e if html_mode else (lambda t: str(t))
    cov = f"{s['coverage_pct']:.1f}%" if s.get("coverage_pct") is not None else "N/A"
    lines = [f"Data: {b(s['status'])} · priced {s['priced_holdings']}/{s['total_lots']} holdings ({cov})",
             f"Prices: {esc(basis.get('label', 'N/A'))} – {esc(basis.get('detail', ''))}"]
    if s.get("unavailable_holdings"):
        lines.append(f"⚠ Unavailable ({s['unavailable_holdings']}): {esc(', '.join(s['unavailable_symbols']))} – "
                     "excluded from value/P&amp;L, NOT counted as zero" if html_mode else
                     f"Unavailable ({s['unavailable_holdings']}): {', '.join(s['unavailable_symbols'])} – "
                     "excluded from value/P&L, not counted as zero")
    if s.get("stale_holdings"):
        lines.append(f"⚠ {s['stale_holdings']} holding(s) use the last stored Yahoo price (stale).")
    if s["is_partial"]:
        lines.append(b("PARTIAL VALUATION – not a complete portfolio value."))
    if not snap.get("valid", True):
        bad = [c["check"] for c in snap.get("validation", []) if not c["ok"]]
        lines.append(b("⚠ Validation failed: ") + esc("; ".join(bad)))
    return lines


def startup_message(snap: dict) -> str:
    now, m, ds, s = snap["generated_at"], snap["market"], snap["data_status"], snap["summary"]
    return "\n".join([
        f"🟢 <b>{_e(config.PROJECT_NAME.upper())} STARTED</b>",
        "",
        f"Date: {now.strftime('%d %B %Y')} ({now.strftime('%A')})",
        f"Time: {now.strftime('%I:%M:%S %p')} IST",
        "",
        f"Market Status: <b>{m['status']}</b> ({_e(m['detail'])})",
        f"Portfolio: {s['total_lots']} holdings · {s['unique_stocks']} unique stocks · "
        f"{len(s.get('accounts', []))} accounts",
        f"Portfolio source: {_e(Path(snap['load'].source_file).name if snap['load'].source_file else 'missing')}",
        "",
        "Market Data: Yahoo Finance",
        f"Yahoo status: {ds['state'].title()}",
        *coverage_lines(snap),
        "",
        "Dashboard:",
        _e(config.DASHBOARD_URL),
        "",
        "<i>Monitoring &amp; analysis only – no trading.</i>",
    ])


def _movers(snap_or_positions, n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    from modules.portfolio import top_movers
    positions = snap_or_positions["positions"] if isinstance(snap_or_positions, dict) else snap_or_positions
    return top_movers(positions, n)


def _mover_lines(snap: dict, n: int = 5) -> list[str]:
    up, down = _movers(snap, n)
    out = []
    if len(up):
        out += ["", "<b>Top gainers today</b>"] + [
            f"🟢 {_e(r.base_symbol)} {num(r.current_price)} ({pct(r.day_change_pct)}) · day P&amp;L {inr(r.todays_pnl, 0, True)}"
            for r in up.itertuples()]
    if len(down):
        out += ["", "<b>Top decliners today</b>"] + [
            f"🔴 {_e(r.base_symbol)} {num(r.current_price)} ({pct(r.day_change_pct)}) · day P&amp;L {inr(r.todays_pnl, 0, True)}"
            for r in down.itertuples()]
    if not out:
        out = ["", "<i>Top movers unavailable (no fresh price + previous close).</i>"]
    return out


def _account_lines(snap: dict) -> list[str]:
    acc = snap.get("accounts_table")
    if acc is None or acc.empty:
        return []
    out = ["", "<b>Accounts</b>"]
    for r in acc.itertuples(index=False):
        out.append(f"• {_e(r[0])}: {inr(r[2], 0)} · P&amp;L {inr(r[3], 0, True)} ({pct(r[4])}) · "
                   f"today {inr(r[5], 0, True)} · {r[8]}/{r[6]} priced")
    return out


def _important_news_lines(snap: dict, hours: int = 24, limit: int = 5) -> list[str]:
    from modules import news as nws
    syms = snap["positions"]["yahoo_symbol"].dropna().tolist() if not snap["positions"].empty else []
    if not syms:
        return []
    try:
        df = nws.get_news(syms, days=max(1, hours // 24), category="Important Event", limit=limit)
    except Exception:  # news must never break a report
        log.exception("News lookup for report failed")
        return ["<i>News unavailable temporarily.</i>"]
    if df.empty:
        return []
    base = snap["positions"].set_index("yahoo_symbol")["base_symbol"].to_dict()
    return [f"• <b>{_e(base.get(r.yahoo_symbol, r.yahoo_symbol))}</b>: {_e(r.headline)} ({_e(r.publisher)})"
            for r in df.itertuples()]


def _value_lines(s: dict) -> list[str]:
    return [f"Portfolio Value: <b>{inr(s['current_value'], 0)}</b>" + (" (partial)" if s["is_partial"] else ""),
            f"Invested Value: {inr(s['invested_value'], 0)}",
            f"Total P&amp;L: {inr(s['total_pnl'], 0, True)}",
            f"Return: {pct(s['return_pct'])}",
            f"Today's P&amp;L: {inr(s['todays_pnl'], 0, True)} ({pct(s['todays_pct'])})"
            + (f" – from {s['todays_pnl_holdings']}/{s['total_lots']} holdings with a previous close"
               if s["todays_pnl_holdings"] < s["total_lots"] else "")]


def market_open_message(snap: dict) -> str:
    now, s, p = snap["generated_at"], snap["summary"], snap["positions"]
    lines = [f"🔔 <b>{_e(config.PROJECT_NAME.upper())} MARKET OPEN REPORT</b>",
             f"{now.strftime('%d %B %Y, %A')} · {now.strftime('%I:%M %p')} IST",
             f"Market: <b>{snap['market']['status']}</b>", ""] + _value_lines(s) + [""] + coverage_lines(snap)
    lines += _mover_lines(snap) + ["", "<b>Opening prices</b> (Yahoo)"]
    if not p.empty:
        for r in p.sort_values("current_value", ascending=False, na_position="last").itertuples():
            q = snap["quotes"].get(r.yahoo_symbol) or {}
            lines.append(f"{_arrow(r.day_change_pct)} {_e(r.base_symbol)}: open {num(q.get('open'))} · "
                         f"now {num(r.current_price)} ({pct(r.day_change_pct)})")
    news = _important_news_lines(snap)
    if news:
        lines += ["", "<b>Important news</b>"] + news
    lines += ["", f"Yahoo data time: {md.fmt_datetime(snap['yahoo_timestamp'])}",
              "<i>Monitoring only – no trades are placed.</i>"]
    return "\n".join(lines)


def closing_title(snap: dict) -> str:
    label = (snap.get("price_basis") or {}).get("label", "")
    if label.startswith("LIVE"):
        return "PRE-CLOSE REPORT (market still open – not closing prices)"
    if label == "CLOSED / LAST CLOSE":
        return "DAILY CLOSING REPORT"
    return f"DAILY REPORT – {label or 'DATA UNAVAILABLE'}"


def closing_message(snap: dict, history: dict | None = None) -> str:
    from modules import fundamentals as fnd
    from modules import risk
    now, s, p = snap["generated_at"], snap["summary"], snap["positions"]
    lines = [f"📊 <b>{_e(config.PROJECT_NAME.upper())} {_e(closing_title(snap))}</b>",
             f"Date: {now.strftime('%d %B %Y, %A')}",
             f"Market status: <b>{snap['market']['status']}</b>", ""] + _value_lines(s) + [""] + coverage_lines(snap)
    lines += _account_lines(snap)
    lines += _mover_lines(snap)
    if not p.empty:
        sectors = risk.sector_exposure(p, fnd.sector_map(p["yahoo_symbol"].dropna().tolist()))
        if len(sectors):
            lines += ["", "<b>Sector exposure</b>"] + [
                f"• {_e(r.sector)}: {r.weight_pct:.1f}%" for r in sectors.head(6).itertuples()]
    news = _important_news_lines(snap)
    if news:
        lines += ["", "<b>Important news / events</b>"] + news
    lines += ["", f"Yahoo data timestamp: {md.fmt_datetime(snap['yahoo_timestamp'])}",
              f"Report generated: {md.fmt_datetime(now)}",
              "<i>Figures are estimates from Yahoo Finance data. Not investment advice.</i>"]
    return "\n".join(lines)


def four_stock_message(snap: dict, symbols: list[str] | None = None) -> str:
    """Short performance report for up to four USER-CONFIGURED stocks (REPORT_STOCKS)."""
    from modules.portfolio import find_position
    symbols = [s.upper() for s in (symbols or config.REPORT_STOCKS)][:4]
    now = snap["generated_at"]
    if not symbols:
        return "No report stocks configured. Set REPORT_STOCKS=SYM1,SYM2,SYM3,SYM4 in .env."
    closed = (snap.get("price_basis") or {}).get("label") == "CLOSED / LAST CLOSE"
    lines = [f"📌 <b>{_e(config.PROJECT_NAME.upper())} – SELECTED HOLDINGS</b>",
             f"{now.strftime('%d %b %Y, %I:%M %p')} IST · Market {snap['market']['status']}", ""]
    for sym in symbols:
        r = find_position(snap["positions"], sym)
        if r is None:
            lines.append(f"• {_e(sym)}: not found in holdings")
            continue
        q = snap["quotes"].get(r["yahoo_symbol"]) or {}
        chg = (r["current_price"] - r["previous_close"]) if _isnum(r["current_price"]) and _isnum(r["previous_close"]) else None
        lines += [f"{_arrow(chg)} <b>{_e(r['base_symbol'])}</b> ({_e(r['yahoo_symbol'] or 'unresolved')})",
                  f"   Open: {num(q.get('open'))} | {'Close' if closed else 'Current'}: {num(r['current_price'])}",
                  f"   Today: {num(chg)} ({pct(r['day_change_pct'])})",
                  f"   Portfolio P&amp;L: {inr(r['pnl'], 0, True)} ({pct(r['return_pct'])})", ""]
    lines.append("<i>Monitoring only – no trades are placed.</i>")
    return "\n".join(lines)


# ================================================================ PDF
def generate_pdf(snap: dict, trigger: str = "manual", include_sell: dict | None = None,
                 include_sections: set[str] | None = None) -> Path:
    """Build a PDF report from the current snapshot. Returns the file path."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)
    from modules import fundamentals as fnd
    from modules import news as nws
    from modules import risk
    from modules import scanner
    from modules import technical as ta

    sections = include_sections or {"summary", "holdings", "technical", "fundamentals", "news", "risk", "scanner"}
    now = snap["generated_at"]
    path = config.REPORTS_DIR / f"FinPilot_Report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=landscape(A4), leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm,
                            title=f"{config.PROJECT_NAME} Portfolio Report", author=config.PROJECT_NAME)
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, textColor=colors.HexColor("#0f1b33"), alignment=TA_LEFT)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=colors.HexColor("#1c5cab"), spaceBefore=8)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#52514e"))
    rupee = "Rs."  # Helvetica has no ₹ glyph

    def money(x, d=0, signed=False):
        return inr(x, d, signed).replace(CUR, rupee)

    def table(data, widths=None, numeric_from=1):
        t = Table(data, colWidths=widths, repeatRows=1)
        style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f1b33")),
                 ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                 ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                 ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                 ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f5f9")]),
                 ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d5d9e2")),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
        if numeric_from is not None:
            style.append(("ALIGN", (numeric_from, 1), (-1, -1), "RIGHT"))
        t.setStyle(TableStyle(style))
        return t

    s, p = snap["summary"], snap["positions"]
    story = [Paragraph(f"{config.PROJECT_NAME} · Portfolio Report", h1),
             Paragraph(f"{config.PROJECT_TAGLINE} &nbsp;|&nbsp; Generated {md.fmt_datetime(now)} &nbsp;|&nbsp; "
                       f"Market {snap['market']['status']} &nbsp;|&nbsp; Yahoo data time "
                       f"{md.fmt_datetime(snap['yahoo_timestamp'])}", small),
             Spacer(1, 6)]

    if "summary" in sections:
        story += [Paragraph("Portfolio summary", h2), table([
            ["Portfolio value", "Invested", "Total P&L", "Return", "Today's P&L", "Holdings (priced)"],
            [money(s["current_value"]) + (" (partial)" if s["is_partial"] else ""), money(s["invested_value"]),
             money(s["total_pnl"], 0, True), pct(s["return_pct"]),
             f"{money(s['todays_pnl'], 0, True)} ({pct(s['todays_pct'])})",
             f"{s['total_lots']} ({s['priced_holdings']})"]], numeric_from=0)]
        story.append(Spacer(1, 4))
        for line in coverage_lines(snap, html_mode=False):
            story.append(Paragraph(html.escape(line).replace("⚠", "!"), small))
        acc = snap.get("accounts_table")
        if acc is not None and len(acc):
            rows = [["Account", "Invested", "Current value", "P&L", "Return", "Today's P&L", "Holdings", "Priced",
                     "Unavailable"]]
            for r in acc.itertuples(index=False):
                rows.append([r[0], money(r[1]), money(r[2]), money(r[3], 0, True), pct(r[4]), money(r[5], 0, True),
                             str(r[6]), str(r[8]), str(r[9])])
            rows.append(["TOTAL", money(s["invested_value"]), money(s["current_value"]), money(s["total_pnl"], 0, True),
                         pct(s["return_pct"]), money(s["todays_pnl"], 0, True), str(s["total_lots"]),
                         str(s["priced_holdings"]), str(s["unavailable_holdings"])])
            story += [Paragraph("Account-wise breakdown", h2), table(rows)]
        up, down = _movers(snap, 5)
        if len(up) or len(down):
            rows = [["", "Stock", "Price", "Day %", "Today's P&L"]]
            rows += [["Gainer", r.base_symbol, num(r.current_price), pct(r.day_change_pct), money(r.todays_pnl, 0, True)]
                     for r in up.itertuples()]
            rows += [["Decliner", r.base_symbol, num(r.current_price), pct(r.day_change_pct), money(r.todays_pnl, 0, True)]
                     for r in down.itertuples()]
            story += [Paragraph("Top gainers / decliners", h2), table(rows, numeric_from=2)]
        if not snap.get("valid", True):
            story.append(Paragraph("VALIDATION FAILED: " + html.escape("; ".join(
                c["check"] for c in snap["validation"] if not c["ok"])), body))

    if "holdings" in sections and not snap["lots"].empty:
        rows = [["Account", "Stock", "Yahoo", "Qty", "Avg buy", "Price", "Invested", "Value", "P&L", "Return",
                 "Today P&L", "Status"]]
        for r in snap["lots"].sort_values(["account", "current_value"], ascending=[True, False],
                                          na_position="last").itertuples():
            rows.append([r.account, r.base_symbol, r.yahoo_symbol or "—", num(r.quantity, 0), num(r.avg_price),
                         num(r.current_price), money(r.invested_value), money(r.current_value), money(r.pnl, 0, True),
                         pct(r.return_pct), money(r.todays_pnl, 0, True), r.price_status])
        story += [Paragraph("Stock-wise holdings (each lot, by account)", h2), table(rows, numeric_from=3)]

    syms = p["yahoo_symbol"].dropna().tolist() if not p.empty else []
    history = md.get_history(syms + [config.BENCHMARK_SYMBOL]) if syms else {}

    if "technical" in sections and syms:
        rows = [["Stock", "Price", "SMA20", "SMA50", "SMA200", "EMA20", "RSI14", "MACD", "Signal", "BB up", "BB low",
                 "Support", "Resist.", "Vol ratio"]]
        for r in p.itertuples():
            if not r.yahoo_symbol:
                continue
            ind = ta.compute_indicators(history.get(r.yahoo_symbol), snap["quotes"].get(r.yahoo_symbol),
                                        md.get_intraday(r.yahoo_symbol))
            rows.append([r.base_symbol] + [num(ind.get(k)) for k in
                        ("price", "sma20", "sma50", "sma200", "ema20", "rsi14", "macd", "macd_signal",
                         "bb_upper", "bb_lower", "support_20d", "resistance_20d", "volume_ratio")])
        story += [PageBreak(), Paragraph("Technical indicators (Yahoo daily history)", h2), table(rows),
                  Paragraph("Indicators describe past price behaviour and do not predict future movement.", small)]

    if "fundamentals" in sections and syms:
        rows = [["Stock", "Sector", "Mkt cap", "P/E", "Fwd P/E", "P/B", "ROE", "Rev growth", "Div yield", "EPS"]]
        for r in p.itertuples():
            f = fnd.get_cached(r.yahoo_symbol) if r.yahoo_symbol else None
            f = f or {}
            rows.append([r.base_symbol, (f.get("sector") or "N/A")[:22],
                         fnd.format_value(f.get("market_cap"), "money").replace(CUR, rupee),
                         num(f.get("pe")), num(f.get("forward_pe")), num(f.get("pb")),
                         fnd.format_value(f.get("roe"), "pct"), fnd.format_value(f.get("revenue_growth"), "pct"),
                         fnd.format_value(f.get("dividend_yield"), "pct"), num(f.get("eps"))])
        story += [Paragraph("Fundamentals (Yahoo Finance, cached)", h2), table(rows),
                  Paragraph("N/A = not provided by Yahoo Finance or not yet loaded. No values are estimated.", small)]

    if "risk" in sections and not p.empty:
        conc = risk.concentration(p)
        sectors = risk.sector_exposure(p, fnd.sector_map(syms))
        pm = risk.portfolio_metrics(p, history, history.get(config.BENCHMARK_SYMBOL))
        story += [PageBreak(), Paragraph("Risk & diversification", h2)]
        for note in risk.diversification_notes(conc, sectors):
            story.append(Paragraph("• " + html.escape(note), body))
        if pm.get("available"):
            story.append(Spacer(1, 4))
            story.append(table([["Volatility (ann.)", "Max drawdown", "1-day VaR 95%", "Beta vs Nifty", "Days"],
                                [pct(pm["volatility_ann_pct"], False), pct(pm["max_drawdown_pct"]),
                                 f"{pct(pm['var_95_1d_pct'], False)} ({money(pm['var_95_1d_value'])})",
                                 num(pm.get("beta_vs_benchmark")), str(pm["days"])]], numeric_from=0))
            story.append(Paragraph("Based on current quantities priced at historical Yahoo closes (hypothetical, "
                                   "ignores actual purchase dates).", small))
        if len(sectors):
            story.append(Spacer(1, 4))
            story.append(table([["Sector", "Value", "Weight", "Holdings"]] + [
                [r.sector, money(r.current_value), pct(r.weight_pct, False), str(r.holdings)]
                for r in sectors.itertuples()]))

    if "scanner" in sections and not p.empty:
        sc = scanner.scan(p, history, snap["quotes"])
        rows = [["Stock", "Price", "Day %", "Return %", "RSI", "Vol ratio", "Score", "Checks", "Flags"]]
        for _, r in sc.iterrows():
            score = r["Condition Score"]
            rows.append([r["Stock"], num(r["Price"]), pct(r["Day %"]), pct(r["Return %"]), num(r["RSI14"]),
                         num(r["Vol Ratio"]), "N/A" if score is None or pd.isna(score) else f"{int(score)}",
                         r["Checks Met"], Paragraph(html.escape(str(r["Flags"])), small)])
        story += [Paragraph("Portfolio scanner (holdings only)", h2), table(rows),
                  Paragraph("Condition score = share of listed factual checks currently met. It is descriptive, "
                            "not a buy/sell recommendation.", small)]

    if "news" in sections and syms:
        nd = nws.get_news(syms, days=7, limit=25)
        story += [Paragraph("Recent news (Yahoo Finance)", h2)]
        if nd.empty:
            story.append(Paragraph("No Yahoo Finance news stored for the last 7 days.", body))
        else:
            nd = nd.astype(object).where(nd.notna(), None)
            base = p.set_index("yahoo_symbol")["base_symbol"].to_dict()
            rows = [["Stock", "Published", "Headline", "Publisher", "Tone", "Category"]]
            for r in nd.itertuples():
                pub = datetime.fromisoformat(r.published_at).strftime("%d %b %H:%M") if r.published_at else "N/A"
                rows.append([base.get(r.yahoo_symbol, r.yahoo_symbol), pub, Paragraph(html.escape(str(r.headline)), small),
                             r.publisher, r.sentiment, r.category])
            story.append(table(rows, widths=[60, 60, 380, 90, 50, 70], numeric_from=None))

    if include_sell:
        story += [Paragraph("Sell analysis – ESTIMATE ONLY (no order placed)", h2)]
        for key, est in include_sell.items():
            if not est.get("valid"):
                story.append(Paragraph(html.escape(f"{est.get('price_label')}: {est.get('message')}"), body))
                continue
            story.append(table([
                ["Basis", "Stock", "Qty", "Avg buy", "Price", "Sale value", "Invested", "Est. P&L", "Return", "Remaining qty",
                 "Remaining value"],
                [est["price_label"], est["stock"], num(est["sell_quantity"], 0), num(est["avg_buy_price"]),
                 num(est["price_used"]), money(est["estimated_sale_value"]), money(est["invested_amount"]),
                 money(est["estimated_pnl"], 0, True), pct(est["return_pct"]), num(est["remaining_quantity"], 0),
                 money(est["remaining_estimated_value"])]]))
        story.append(Paragraph("Excludes brokerage, STT, exchange charges, stamp duty and taxes.", small))

    story += [Spacer(1, 10), Paragraph(config.DISCLAIMER, small)]
    doc.build(story)
    db.execute("INSERT OR IGNORE INTO reports(report_type, file_path, trigger, created_at) VALUES(?,?,?,?)",
               ("portfolio_pdf", str(path), trigger, db.now_iso()))
    return path


def list_reports(limit: int = 50) -> pd.DataFrame:
    rows = db.query("SELECT * FROM reports ORDER BY created_at DESC LIMIT ?", (limit,))
    return pd.DataFrame([r for r in rows if Path(r["file_path"]).exists()])
