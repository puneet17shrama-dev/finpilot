"""
FinPilot – Streamlit dashboard.

    streamlit run app.py

* Market-dependent sections refresh about every MARKET_REFRESH_SECONDS (10 s)
  with st.fragment(run_every=...) – no `while True` inside Streamlit.
* Quotes come from one cached Yahoo batch request per refresh window; daily
  history, fundamentals and news are cached separately (see modules/).
* A background Monitor thread (started once per server process) evaluates
  alerts and sends scheduled Telegram reports.
* FinPilot never places orders. Sell Analysis is a calculation only.
"""
from __future__ import annotations

import math
from html import escape
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
import config
from agent import FinPilotAgent
from modules import alerts as al, database as db, fundamentals as fnd, market_data as md, news as nws
from modules import portfolio as pf, reports as rp, risk, scanner, sell_analysis as sa, technical as ta, telegram as tg

st.set_page_config(page_title=config.PROJECT_NAME, layout="wide")
db.init_db()

REFRESH = config.MARKET_REFRESH_SECONDS
C = {
    "bg": "#f7f8fa", "text": "#202631", "muted": "#697586", "border": "#d9dee7",
    "pos": "#16845b", "neg": "#c44747", "warn": "#b7791f", "accent": "#2878a5",
    "series": ["#2878a5", "#d48435", "#16845b", "#8d65a9", "#c44747", "#4b8b8b", "#7a8290", "#b35c84"],
}
@st.cache_resource
def get_monitor():
    return al.Monitor().start()


MONITOR = get_monitor()


def num(value, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def inr(value, decimals: int = 2, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    number = float(value)
    sign = "+" if signed and number > 0 else ""
    return f"{sign}{config.CURRENCY_SYMBOL}{number:,.{decimals}f}"


def compact(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    number = float(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e7, "Cr"), (1e5, "L")):
        if abs(number) >= threshold:
            return f"{config.CURRENCY_SYMBOL}{number / threshold:.2f}{suffix}"
    return inr(number, 0)


def pct(value, signed: bool = True) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    number = float(value)
    sign = "+" if signed and number > 0 else ""
    return f"{sign}{number:.2f}%"


def cls(value) -> str:
    if value is None or pd.isna(value) or value == 0:
        return ""
    return "pos" if value > 0 else "neg"


def card(title: str, value, subtitle: str = "", tone: str = "") -> str:
    return (f'<div class="fp-card {escape(tone)}"><div class="fp-card-title">{escape(str(title))}</div>'
            f'<div class="fp-card-value">{escape(str(value))}</div>'
            f'<div class="fp-card-subtitle">{escape(str(subtitle))}</div></div>')


def plotly_layout(fig, height: int, title: str):
    fig.update_layout(height=height, title=title, template="plotly_white", paper_bgcolor="white",
                      plot_bgcolor="white", font=dict(color=C["text"]), margin=dict(l=16, r=16, t=50, b=16))
    return fig


def snapshot() -> dict:
    snap = pf.build_snapshot(network=not MONITOR.status()["running"])
    return pf.filter_snapshot(snap, st.session_state.get("accounts"))


st.markdown("""<style>
.fp-card{border:1px solid #d9dee7;border-radius:6px;background:#fff;padding:14px 16px;min-height:92px}
.fp-card-title{color:#697586;font-size:12px}.fp-card-value{color:#202631;font-size:22px;font-weight:650;margin:5px 0}
.fp-card-subtitle{color:#697586;font-size:11px}.fp-card.pos .fp-card-value,.pos{color:#16845b}
.fp-card.neg .fp-card-value,.neg{color:#c44747}.fp-brand{font-weight:750;color:#202631}.fp-tag{color:#697586;font-size:12px}
.fp-section{font-size:16px;font-weight:700;margin:16px 0 8px}.fp-note,.fp-status{color:#697586;font-size:12px}
.fp-estimate{padding:10px 12px;border-left:3px solid #b7791f;background:#fff8e8;color:#644d1e}
.fp-pill{display:inline-block;border:1px solid #d9dee7;border-radius:12px;padding:3px 8px;margin-right:5px;font-size:12px}
</style>""", unsafe_allow_html=True)

def history_for(symbols: list[str]) -> dict[str, pd.DataFrame]:
    return md.get_history([s for s in symbols if s] + [config.BENCHMARK_SYMBOL])


# ================================================================ sidebar
base_snap = pf.build_snapshot(network=not MONITOR.status()["running"])
with st.sidebar:
    st.markdown(f'<div class="fp-brand" style="font-size:22px">{config.PROJECT_NAME.upper()}</div>'
                f'<div class="fp-tag">{config.PROJECT_TAGLINE}</div>', unsafe_allow_html=True)
    st.write("")
    PAGES = ["Overview", "Technical Analysis", "Fundamentals", "News", "Risk", "Scanner", "Sell Analysis",
             "Alerts", "AI Assistant", "Reports", "Data & Settings"]
    page = st.radio("Navigate", PAGES, label_visibility="collapsed", key="page")
    st.divider()
    all_accounts = base_snap["summary"].get("accounts", [])
    if len(all_accounts) > 1:
        st.multiselect("Accounts", all_accounts, default=all_accounts, key="accounts",
                       help="Filter the dashboard to selected accounts from holdings.xlsx")
    if st.button("↻ Refresh market data now", width="stretch"):
        md.get_quotes(base_snap["positions"]["yahoo_symbol"].dropna().tolist(), force=True)
        st.rerun()
    st.caption(f"Live sections refresh every ~{REFRESH}s. Market data: Yahoo Finance only.")
    st.caption("⚠️ " + config.DISCLAIMER)

# ================================================================ header
h1, h2, h3 = st.columns([1.25, 1.1, 1.9], vertical_alignment="center")
with h1:
    st.markdown(f'<div class="fp-brand">{config.PROJECT_NAME.upper()}</div>'
                f'<div class="fp-tag">{config.PROJECT_TAGLINE}</div>', unsafe_allow_html=True)
with h2:
    # Local app clock – ticks every second in the browser, shown in Asia/Kolkata.
    components.html(f"""
    <style>body {{ margin: 0; }}</style>
    <div style="font-family: 'Source Sans Pro', system-ui, sans-serif; color:{C['text']}; line-height:1.25">
      <div id="fp-date" style="font-size:14px;color:{C['muted']}"></div>
      <div id="fp-time" style="font-size:26px;font-weight:700;font-variant-numeric:tabular-nums"></div>
      <div style="font-size:11px;color:{C['muted']}">Local clock ({config.TIMEZONE}) · not exchange time</div>
    </div>
    <script>
      const tz = "{config.TIMEZONE}";
      function tick() {{
        const d = new Date();
        const date = d.toLocaleDateString('en-GB', {{timeZone: tz, day: '2-digit', month: 'long', year: 'numeric'}});
        const day = d.toLocaleDateString('en-GB', {{timeZone: tz, weekday: 'long'}});
        const time = d.toLocaleTimeString('en-US', {{timeZone: tz, hour: '2-digit', minute: '2-digit',
                                                     second: '2-digit', hour12: true}});
        document.getElementById('fp-date').textContent = date + ' · ' + day;
        document.getElementById('fp-time').textContent = time + ' IST';
      }}
      tick(); setInterval(tick, 1000);
    </script>""", height=86)


@st.fragment(run_every=REFRESH)
def header_status():
    snap = snapshot()
    m, ds = snap["market"], snap["data_status"]
    mcol = {"OPEN": "green", "PRE-OPEN": "amber"}.get(m["status"], "gray")
    dcol = {"CONNECTED": "green", "STALE": "amber", "ERROR": "red"}.get(ds["state"], "gray")
    nxt = ds["next_refresh_in"]
    err = f'<br><span class="neg">⚠ {ds["last_error"][:140]}</span>' if ds["last_error"] else ""
    st.markdown(f"""
    <div>
      <span class="fp-pill {mcol}">Market: {m['status']}</span>
      <span class="fp-pill {dcol}">Yahoo Data: {ds['state']}</span>
      <span class="fp-pill gray">Monitor: {'running' if MONITOR.status()['running'] else 'stopped'}</span>
    </div>
    <div class="fp-status">
      Market: <b>{m['detail']}</b> (schedule-based) ·
      Last Market Data Refresh: <b>{md.fmt_time(ds['last_success'])}</b> ·
      Next Refresh: <b>~{int(nxt) if nxt is not None else REFRESH} s</b><br>
      Yahoo Data Timestamp: <b>{md.fmt_datetime(snap['yahoo_timestamp'])}</b> ·
      Server Clock: <b>{md.fmt_time(snap['generated_at'])}</b>{err}
    </div>""", unsafe_allow_html=True)


with h3:
    header_status()
st.markdown(f'<hr style="border:none;border-top:1px solid {C["border"]};margin:6px 0 10px">', unsafe_allow_html=True)


def fmt_df(df: pd.DataFrame, fmt: dict, color_cols: tuple = (), status_col: str | None = None):
    """Format a numeric frame for display. Missing values show as N/A (Streamlit renders raw
    nulls as 'None', so values are pre-formatted); colours are computed from the numeric data."""
    disp = df.copy().astype(object)
    css = pd.DataFrame("", index=df.index, columns=df.columns)
    for col in df.columns:
        f = fmt.get(col)
        vals = df[col]
        disp[col] = [("N/A" if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA
                      else (f(v) if f else v)) for v in vals]
        if f:
            css[col] = "text-align: right"
    for col in color_cols:
        css[col] = [f"text-align: right; color: {C['pos']}" if isinstance(v, (int, float)) and not math.isnan(v) and v > 0
                    else f"text-align: right; color: {C['neg']}" if isinstance(v, (int, float)) and not math.isnan(v) and v < 0
                    else f"text-align: right; color: {C['muted']}" for v in df[col]]
    if status_col:
        css[status_col] = [f"color: {C['pos']}" if v in {"Live", "Closed", "Stored close"} else f"color: {C['warn']}" if v in {"Stale", "Market closed"}
                           else f"color: {C['neg']}" for v in df[status_col]]
    return disp.style.apply(lambda _: css, axis=None)


# ================================================================ table styling
def style_positions(df: pd.DataFrame):
    def safe_col(name: str, default=None):
        if name in df.columns:
            return df[name]
        return pd.Series([default] * len(df), index=df.index)

    cols = {
        "Stock": safe_col("base_symbol", ""),
        "Yahoo Symbol": safe_col("yahoo_symbol", "Unresolved").fillna("Unresolved"),
        "Quantity": safe_col("quantity", np.nan),
        "Avg Buy Price": safe_col("avg_price", np.nan),
        "Current Price": safe_col("current_price", np.nan),
        "Current Value": safe_col("current_value", np.nan),
        "Invested Value": safe_col("invested_value", np.nan),
        "P&L": safe_col("pnl", np.nan),
        "Return %": safe_col("return_pct", np.nan),
        "Day %": safe_col("day_change_pct", np.nan),
        "Today's P&L": safe_col("todays_pnl", np.nan),
        "Status": safe_col("price_status", "UNAVAILABLE"),
    }
    if "account" in df.columns:
        cols = {"Account": df["account"].fillna("Default"), **cols}
    view = pd.DataFrame(cols)
    fmt = {"Quantity": lambda v: num(v, 0), "Avg Buy Price": lambda v: inr(v), "Current Price": lambda v: inr(v),
           "Current Value": lambda v: inr(v, 0), "Invested Value": lambda v: inr(v, 0),
           "P&L": lambda v: inr(v, 0, True), "Return %": lambda v: pct(v), "Day %": lambda v: pct(v),
           "Today's P&L": lambda v: inr(v, 0, True)}
    return fmt_df(view, fmt, ("P&L", "Return %", "Day %", "Today's P&L"), "Status")


# ================================================================ pages
def render_data_warnings(snap: dict) -> None:
    """Prominent banners whenever the valuation is partial, stale or fails validation."""
    s = snap["summary"]
    if snap["load"].error:
        st.error(snap["load"].error)
    if not snap.get("valid", True):
        bad = [f"{c['check']} {c['detail']}".strip() for c in snap["validation"] if not c["ok"]]
        st.error("❌ Portfolio validation failed – figures may be inconsistent: " + "; ".join(bad))
    if s["total_lots"] and s["priced_holdings"] == 0:
        st.warning("Yahoo Finance prices are currently unavailable – values show N/A and will retry on the "
                   "next refresh. No prices are estimated.")
    elif s["is_partial"]:
        st.warning(f"PARTIAL VALUATION – {s['priced_holdings']}/{s['total_lots']} holdings priced "
                   f"({s['coverage_pct']:.1f}% coverage). Unavailable: {', '.join(s['unavailable_symbols'])}. "
                   "Their current value is excluded (not counted as ₹0); invested value is still shown in full.")
    hol = md.holiday_calendar_status(snap["generated_at"])
    if not hol["covered"]:
        st.warning(f"No exchange holidays configured for {hol['year']} – market status may show OPEN on a holiday. "
                   "Add the official NSE dates to MARKET_HOLIDAYS in .env.")
    if MONITOR.token_scan and MONITOR.token_scan["hits"]:
        st.error("Telegram bot token found outside .env in: " + ", ".join(MONITOR.token_scan["hits"])
                 + ". Revoke it in @BotFather, remove it from those files and put the new token in .env.")
    if s.get("stale_holdings"):
        st.info(f"{s['stale_holdings']} holding(s) are using the last stored Yahoo price (stale) until the next "
                "successful refresh.")


def page_overview():
    @st.fragment(run_every=REFRESH)
    def live():
        snap = snapshot()
        all_accounts = snap["summary"].get("accounts", [])
        account_options = ["ALL ACCOUNTS"] + all_accounts
        selected_account = st.selectbox("Portfolio account", account_options, index=0, key="portfolio_account_filter")
        selected_accounts = all_accounts if selected_account == "ALL ACCOUNTS" else [selected_account]
        filtered_snap = pf.filter_snapshot(snap, selected_accounts if selected_account != "ALL ACCOUNTS" else None)
        s, p = filtered_snap["summary"], filtered_snap["positions"]
        lots = filtered_snap["lots"].copy()
        partial = (f"PARTIAL · {s['priced_holdings']}/{s['total_lots']} priced ({s['coverage_pct']:.1f}%)"
                   if s["is_partial"] and s["coverage_pct"] is not None else
                   f"All {s['total_lots']} holdings priced · {(filtered_snap.get('price_basis') or {}).get('label', '')}")
        unique_stock_count = int(s.get("unique_stocks", 0))
        cols = st.columns(6)
        total_lot_count = int(s.get("total_lots", s.get("total_holdings", 0)))
        cards = [
            card("Invested Value", inr(s["invested_value"], 0), f"{total_lot_count} valid lots · {len(s.get('accounts', []))} accounts"),
            card("Current Portfolio Value", inr(s["current_value"], 0), partial),
            card("Total P&L", inr(s["total_pnl"], 0, True), "vs invested (priced holdings)", cls(s["total_pnl"])),
            card("Return %", pct(s["return_pct"]), "Zero-cost lots excluded from % only", cls(s["return_pct"])),
            card("Today's P&L", inr(s["todays_pnl"], 0, True), f"{pct(s['todays_pct'])} vs previous close",
                 cls(s["todays_pnl"])),
            card("TOTAL HOLDING LOTS", f"{total_lot_count}", f"{unique_stock_count} UNIQUE STOCKS"),
        ]
        for c, html in zip(cols, cards):
            c.markdown(html, unsafe_allow_html=True)
        render_data_warnings(filtered_snap)

        st.markdown('<div class="fp-section">Account-wise portfolio</div>', unsafe_allow_html=True)
        acc = filtered_snap["accounts_table"]
        if len(acc):
            total_row = pd.DataFrame([{"Account": "TOTAL", "Invested Value": s["invested_value"],
                                       "Current Value": s["current_value"], "P&L": s["total_pnl"],
                                       "Return %": s["return_pct"], "Today's P&L": s["todays_pnl"],
                                       "Holdings": s["total_lots"], "Unique Stocks": s["unique_stocks"],
                                       "Priced": s["priced_holdings"], "Unavailable": s["unavailable_holdings"],
                                       "Coverage %": s["coverage_pct"]}])
            st.dataframe(fmt_df(pd.concat([acc, total_row], ignore_index=True),
                                {"Invested Value": lambda v: inr(v, 0), "Current Value": lambda v: inr(v, 0),
                                 "P&L": lambda v: inr(v, 0, True), "Return %": lambda v: pct(v),
                                 "Today's P&L": lambda v: inr(v, 0, True), "Coverage %": lambda v: f"{v:.1f}%"},
                                ("P&L", "Return %", "Today's P&L")), hide_index=True)
            recon = [c for c in filtered_snap["validation"] if c["check"].startswith("Σ account")]
            st.caption(("✅ Accounts reconcile with the portfolio total. " if all(c["ok"] for c in recon)
                        else "❌ Account totals do NOT reconcile – see Data & Settings. ")
                       + "Current value / P&L include priced holdings only; unpriced holdings are never counted as ₹0.")

        st.caption(f"Portfolio totals use all valid holding lots ({total_lot_count}) from "
                   f"{Path(snap['load'].source_file).as_posix().split('/')[-1] if snap['load'].source_file else 'holdings file'}; "
                   f"stock-level charts aggregate by unique symbol ({unique_stock_count}).")
        st.markdown('<div class="fp-section">Portfolio holdings</div>', unsafe_allow_html=True)
        if lots.empty:
            st.info("No valid holdings found in data/holdings.xlsx.")
            return
        order = lots.sort_values("current_value", ascending=False, na_position="last")
        st.dataframe(style_positions(order), hide_index=True, height=min(38 * len(order) + 40, 620))
        st.caption("Each lot is shown separately; duplicate stock symbols are preserved per account/lot. "
                   "Return % is N/A when cost is zero. Prices: Yahoo Finance (may be delayed).")

        c1, c2 = st.columns(2)
        with c1:
            pp = p[p["pnl"].notna()].sort_values("pnl")
            if len(pp):
                fig = go.Figure(go.Bar(
                    x=pp["pnl"], y=pp["base_symbol"], orientation="h",
                    marker=dict(color=[C["pos"] if v >= 0 else C["neg"] for v in pp["pnl"]], cornerradius=4),
                    customdata=np.stack([pp["return_pct"].fillna(np.nan)], axis=-1),
                    hovertemplate="<b>%{y}</b><br>P&L ₹%{x:,.0f}<br>Return %{customdata[0]:.2f}%<extra></extra>"))
                st.plotly_chart(plotly_layout(fig, max(320, 18 * len(pp)), "P&L by stock (₹)"),
                                config={"displayModeBar": False})
        with c2:
            g_, d_ = pf.top_movers(p, n=len(p))
            dp = pd.concat([g_, d_]).sort_values("day_change_pct")
            if len(dp):
                fig = go.Figure(go.Bar(
                    x=dp["day_change_pct"], y=dp["base_symbol"], orientation="h",
                    marker=dict(color=[C["pos"] if v >= 0 else C["neg"] for v in dp["day_change_pct"]], cornerradius=4),
                    customdata=np.stack([dp["todays_pnl"]], axis=-1),
                    hovertemplate="<b>%{y}</b><br>%{x:+.2f}% today<br>Today's P&L ₹%{customdata[0]:,.0f}<extra></extra>"))
                st.plotly_chart(plotly_layout(fig, max(320, 18 * len(dp)), "Today's gainers & losers (%)"),
                                config={"displayModeBar": False})

        c3, c4 = st.columns(2)
        priced = p[p["current_value"].notna() & (p["current_value"] > 0)]
        with c3:
            if len(priced):
                fig = go.Figure(go.Treemap(
                    labels=priced["base_symbol"], parents=[""] * len(priced), values=priced["current_value"],
                    root_color="rgba(0,0,0,0)", tiling=dict(pad=2),
                    marker=dict(colors=priced["current_value"], colorscale=[[0, "#184f95"], [1, "#6da7ec"]],
                                line=dict(color=C["bg"], width=2)),
                    texttemplate="<b>%{label}</b><br>%{percentRoot:.1%}",
                    hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percentRoot:.2%} of priced value<extra></extra>"))
                st.plotly_chart(plotly_layout(fig, 380, "Stock allocation (current value)"),
                                config={"displayModeBar": False})
        with c4:
            sectors = risk.sector_exposure(p, fnd.sector_map(p["yahoo_symbol"].dropna().tolist()))
            if len(sectors):
                top = sectors.head(7)
                if len(sectors) > 7:
                    top = pd.concat([top, pd.DataFrame([{"sector": "Other", "current_value": sectors.iloc[7:]["current_value"].sum(),
                                                         "weight_pct": sectors.iloc[7:]["weight_pct"].sum(),
                                                         "holdings": sectors.iloc[7:]["holdings"].sum()}])])
                colors = [("#5b6478" if s == "Unknown" else C["series"][i % 8]) for i, s in enumerate(top["sector"])]
                fig = go.Figure(go.Pie(labels=top["sector"], values=top["current_value"], hole=.6, sort=False,
                                       marker=dict(colors=colors, line=dict(color=C["bg"], width=2)),
                                       hovertemplate="<b>%{label}</b><br>₹%{value:,.0f} (%{percent})<extra></extra>"))
                st.plotly_chart(plotly_layout(fig, 380, "Sector exposure"), config={"displayModeBar": False})
                if (sectors["sector"] == "Unknown").any():
                    st.caption("‘Unknown’ = fundamentals not loaded yet (fetched gradually in the background) "
                               "or sector not provided by Yahoo.")

    live()

    # Historical value (cached history – not re-downloaded every 10 s)
    snap = snapshot()
    p = snap["positions"]
    syms = p["yahoo_symbol"].dropna().tolist()
    if syms:
        hist = history_for(syms)
        series = risk.constant_holdings_value(p, hist)
        if len(series) > 5:
            covered = p[p["yahoo_symbol"].isin([s for s in syms if s in hist])]
            inv = float(covered["invested_value"].sum())
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index, y=series.values, name="Value of current holdings",
                                     line=dict(color=C["accent"], width=2),
                                     hovertemplate="%{x|%d %b %Y}<br>₹%{y:,.0f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[series.index[0], series.index[-1]], y=[inv, inv], name="Invested (same stocks)",
                                     line=dict(color=C["muted"], width=1.5, dash="dot"),
                                     hovertemplate="Invested ₹%{y:,.0f}<extra></extra>"))
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(plotly_layout(fig, 340, "Portfolio value – current quantities at past Yahoo closes (₹)"),
                            config={"displayModeBar": False})
            st.caption(f"Hypothetical series: today's quantities × historical daily closes for {len(covered)} stocks "
                       "with Yahoo history. It ignores actual purchase dates.")


def stock_picker(p: pd.DataFrame, key: str, label: str = "Stock") -> pd.Series | None:
    opts = p.sort_values("base_symbol")
    if opts.empty:
        st.info("No holdings loaded.")
        return None
    labels = [f"{r.base_symbol}  ·  {r.yahoo_symbol or 'unresolved'}" for r in opts.itertuples()]
    choice = st.selectbox(label, labels, key=key)
    return opts.iloc[labels.index(choice)]


def page_technical():
    snap = snapshot()
    p = snap["positions"]
    c1, c2 = st.columns([2, 1])
    with c1:
        row = stock_picker(p, "tech_stock")
    with c2:
        rng = st.segmented_control("Range", ["3M", "6M", "1Y"], default="6M", key="tech_range")
    if row is None:
        return
    sym = row["yahoo_symbol"]
    if not sym:
        st.warning(f"{row['base_symbol']}: Yahoo symbol not resolved yet – technicals unavailable.")
        return
    hist = history_for([sym]).get(sym)
    if hist is None or hist.empty:
        err = md.history_errors().get(sym)
        st.warning(f"Yahoo history unavailable for {sym}. {err or ''} Will retry automatically.")
        return

    @st.fragment(run_every=REFRESH)
    def live_indicators():
        s2 = snapshot()
        q = s2["quotes"].get(sym)
        ind = ta.compute_indicators(hist, q, md.get_intraday(sym))
        ta.save_snapshot(sym, ind)
        cols = st.columns(6)
        vals = [("Price", inr(ind.get("price")), f"{q['status'] if q else 'Unavailable'}"),
                ("RSI 14", num(ind.get("rsi14")), "30 / 70 reference levels"),
                ("MACD", num(ind.get("macd")), f"signal {num(ind.get('macd_signal'))} · hist {num(ind.get('macd_hist'))}"),
                ("SMA 50 / 200", f"{num(ind.get('sma50'), 1)} / {num(ind.get('sma200'), 1)}",
                 f"price vs SMA50 {pct(ind.get('price_vs_sma50_pct'))}"),
                ("20d Support / Resist.", f"{num(ind.get('support_20d'), 1)} / {num(ind.get('resistance_20d'), 1)}",
                 f"pivot {num(ind.get('pivot'), 1)}"),
                ("Volume ratio", num(ind.get("volume_ratio")), f"avg 20d {num(ind.get('avg_volume_20d'), 0)}")]
        for c, (l, v, s_) in zip(cols, vals):
            c.markdown(card(l, v, s_), unsafe_allow_html=True)
        st.write("")
        a, b = st.columns([1.3, 1])
        with a:
            items = [("SMA 20", "sma20", "price_vs_sma20_pct"), ("SMA 50", "sma50", "price_vs_sma50_pct"),
                     ("SMA 200", "sma200", "price_vs_sma200_pct"), ("EMA 20", "ema20", "price_vs_ema20_pct"),
                     ("EMA 50", "ema50", "price_vs_ema50_pct"), ("VWAP (today)", "vwap", "price_vs_vwap_pct"),
                     ("RSI 14", "rsi14", None), ("MACD", "macd", None), ("MACD signal", "macd_signal", None),
                     ("MACD histogram", "macd_hist", None), ("Bollinger upper", "bb_upper", None),
                     ("Bollinger middle", "bb_middle", None), ("Bollinger lower", "bb_lower", None),
                     ("20d support", "support_20d", None), ("20d resistance", "resistance_20d", None),
                     ("Volume (today)", "volume", None), ("Avg volume (20d)", "avg_volume_20d", None),
                     ("Volume ratio", "volume_ratio", None)]
            table = pd.DataFrame([{"Indicator": label, "Value": ta.display_value(ind, key),
                                   "Price vs indicator %": pct(ind.get(vs)) if vs else ""}
                                  for label, key, vs in items])
            st.dataframe(table, hide_index=True)
        with b:
            st.markdown("**Factual observations**")
            for o in ta.describe(ind):
                st.markdown(f"- {o}")
            for n_ in ind.get("notes", []):
                st.caption("ℹ️ " + n_)
            st.caption("Indicators summarise past prices and volumes. They do not predict or guarantee future moves.")

    live_indicators()

    df = ta.indicator_frame(hist, snap["quotes"].get(sym))
    days = {"3M": 63, "6M": 126, "1Y": 252}[rng or "6M"]
    df = df.iloc[-days:]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.52, 0.14, 0.17, 0.17])
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 name="Price", increasing_line_color=C["pos"], decreasing_line_color=C["neg"]), 1, 1)
    for col, color, dash in (("SMA20", C["series"][3], None), ("SMA50", C["series"][0], None),
                             ("SMA200", C["series"][6], None)):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=color, width=1.6, dash=dash)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="Bollinger", line=dict(color=C["muted"], width=1, dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="Bollinger", showlegend=False,
                             line=dict(color=C["muted"], width=1, dash="dot"), fill="tonexty",
                             fillcolor="rgba(139,151,176,.06)"), 1, 1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="rgba(57,135,229,.45)"), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI 14", line=dict(color=C["series"][4], width=1.6)), 3, 1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line=dict(color=C["muted"], width=1, dash="dot"), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="MACD hist",
                         marker_color=[C["pos"] if v >= 0 else C["neg"] for v in df["MACD_hist"].fillna(0)]), 4, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color=C["series"][0], width=1.4)), 4, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color=C["series"][1], width=1.4)), 4, 1)
    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(plotly_layout(fig, 760, f"{row['base_symbol']} ({sym}) – daily, Yahoo Finance"),
                    config={"displayModeBar": False})


def page_fundamentals():
    snap = snapshot()
    p = snap["positions"]
    syms = p["yahoo_symbol"].dropna().unique().tolist()
    c1, c2 = st.columns([2, 1])
    with c1:
        row = stock_picker(p, "fund_stock")
    with c2:
        st.write("")
        missing = fnd.missing_or_stale(syms)
        if st.button(f"Load fundamentals for all holdings ({len(missing)} to fetch)", disabled=not missing,
                     width="stretch"):
            bar = st.progress(0.0, "Fetching from Yahoo Finance…")
            for i, s in enumerate(missing):
                fnd.get_fundamentals(s)
                bar.progress((i + 1) / len(missing), f"{s} ({i + 1}/{len(missing)})")
            st.rerun()
    if row is not None and row["yahoo_symbol"]:
        with st.spinner("Loading Yahoo fundamentals…"):
            f = fnd.get_fundamentals(row["yahoo_symbol"])
        status = (f.get("_status") or "").upper()
        if status == "UNAVAILABLE":
            st.warning(f"Fundamentals unavailable from Yahoo Finance for {row['yahoo_symbol']}: {f.get('_error')}. "
                       "Will retry later.")
        else:
            if status == "STALE":
                st.info(f"Showing last saved Yahoo data (refresh failed: {f.get('_error')}).")
            st.markdown(f"### {f.get('company_name') or row['base_symbol']}")
            st.caption(f"{f.get('sector') or 'N/A'} · {f.get('industry') or 'N/A'} · Yahoo symbol {row['yahoo_symbol']} "
                       f"· fetched {f.get('_fetched_at')}")
            t = fnd.as_table(f)
            a, b = st.columns(2)
            half = (len(t) + 1) // 2
            a.dataframe(t.iloc[:half], hide_index=True)
            b.dataframe(t.iloc[half:], hide_index=True)
            if f.get("business_summary"):
                with st.expander("Business description (Yahoo Finance)"):
                    st.write(f["business_summary"])
            st.caption("N/A = not provided by Yahoo Finance. FinPilot never estimates missing values.")
    elif row is not None:
        st.warning("Yahoo symbol not resolved – fundamentals unavailable.")

    st.markdown('<div class="fp-section">All holdings (cached fundamentals)</div>', unsafe_allow_html=True)
    rows = []
    for r in p.itertuples():
        f = (fnd.get_cached(r.yahoo_symbol) if r.yahoo_symbol else None) or {}
        rows.append({"Stock": r.base_symbol, "Company": f.get("company_name"), "Sector": f.get("sector"),
                     "Market Cap": f.get("market_cap"), "P/E": f.get("pe"), "Fwd P/E": f.get("forward_pe"),
                     "P/B": f.get("pb"), "ROE %": f.get("roe"), "Rev Growth %": f.get("revenue_growth"),
                     "Div Yield %": f.get("dividend_yield"), "Status": f.get("_status", "Not loaded")})
    tbl = pd.DataFrame(rows)
    st.dataframe(fmt_df(tbl, {"Market Cap": compact, "P/E": lambda v: num(v), "Fwd P/E": lambda v: num(v),
                                   "P/B": lambda v: num(v), "ROE %": lambda v: num(v), "Rev Growth %": lambda v: num(v),
                                   "Div Yield %": lambda v: num(v)}), hide_index=True)


def page_news():
    snap = snapshot()
    p = snap["positions"]
    syms = p["yahoo_symbol"].dropna().unique().tolist()
    base = p.set_index("yahoo_symbol")["base_symbol"].to_dict() if len(p) else {}
    c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
    stock = c1.selectbox("Stock", ["All holdings"] + sorted(base.values()), key="news_stock")
    cat = c2.selectbox("Category", ["All", "Important Event", "General"], key="news_cat")
    tone = c3.selectbox("Keyword tone", ["All", "Positive", "Negative", "Neutral"], key="news_tone")
    days = c4.selectbox("Period", [3, 7, 14, 30], index=1, format_func=lambda d: f"Last {d} days", key="news_days")
    sel = syms if stock == "All holdings" else [s for s, b in base.items() if b == stock]
    stale = nws.stale_symbols(sel)
    if st.button(f"Fetch latest Yahoo news ({len(stale)} symbol(s) due)", disabled=not stale):
        bar = st.progress(0.0)
        errors = 0
        for i, s in enumerate(stale):
            res = nws.refresh_news(s)
            errors += res["status"] == "error"
            bar.progress((i + 1) / len(stale), s)
        if errors:
            st.warning(f"Yahoo news request failed for {errors} symbol(s); will retry later.")
        st.rerun()
    df = nws.get_news(sel, days=days, category=None if cat == "All" else cat, limit=300)
    if len(df) and tone != "All":
        df = df[df["sentiment"] == tone]
    st.caption("Tone and category come from transparent keyword rules applied to the headline/summary "
               "(matched words shown). They describe wording only and do not predict stock performance. "
               "News is also fetched gradually in the background and stored once (deduplicated).")
    if df.empty:
        st.info("No Yahoo Finance news stored for this selection yet.")
        return
    df = df.astype(object).where(df.notna(), None)
    for r in df.itertuples():
        pub = md.fmt_datetime(datetime.fromisoformat(r.published_at)) if r.published_at else "N/A"
        tone_cls = {"Positive": "green", "Negative": "red"}.get(r.sentiment, "gray")
        cat_cls = "amber" if r.category == "Important Event" else "gray"
        head = escape(r.headline)
        url = r.url if r.url and str(r.url).startswith(("http://", "https://")) else None
        link = f'<a href="{escape(url, quote=True)}" target="_blank">{head}</a>' if url else f"<b>{head}</b>"
        summary = f'<div class="fp-note" style="margin-top:4px">{escape(r.summary[:300])}</div>' if r.summary else ""
        st.markdown(f"""<div class="fp-news">
            <span class="fp-pill gray">{base.get(r.yahoo_symbol, r.yahoo_symbol)}</span>
            <span class="fp-pill {cat_cls}">{r.category}</span><span class="fp-pill {tone_cls}">{r.sentiment}</span>
            <div style="margin-top:6px">{link}</div>{summary}
            <div class="fp-note" style="margin-top:6px">{escape(str(r.publisher))} · {pub}
            {(' · matched: ' + escape(r.matched_keywords)) if r.matched_keywords else ''}</div></div>""",
                    unsafe_allow_html=True)


def page_risk():
    snap = snapshot()
    p = snap["positions"]
    syms = p["yahoo_symbol"].dropna().unique().tolist()
    hist = history_for(syms) if syms else {}
    bench = hist.get(config.BENCHMARK_SYMBOL)
    conc = risk.concentration(p)
    sectors = risk.sector_exposure(p, fnd.sector_map(syms))
    pm = risk.portfolio_metrics(p, hist, bench)
    cols = st.columns(6)
    items = [
        ("Largest holding", f"{conc['largest_weight_pct']:.1f}%" if conc.get("available") else "N/A",
         conc.get("largest_symbol", "")),
        ("Top 5 weight", f"{conc['top5_weight_pct']:.1f}%" if conc.get("available") else "N/A", "of priced value"),
        ("Effective holdings", num(conc.get("effective_holdings"), 1), "1 / HHI"),
        ("Volatility (ann.)", pct(pm.get("volatility_ann_pct"), False), "1y daily, current holdings"),
        ("Max drawdown", pct(pm.get("max_drawdown_pct")), "1y, current holdings"),
        ("1-day VaR 95%", pct(pm.get("var_95_1d_pct"), False), inr(pm.get("var_95_1d_value"), 0) + " historical"),
    ]
    for c, (l, v, s_) in zip(cols, items):
        c.markdown(card(l, v, s_), unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([1, 1])
    with a:
        st.markdown("**Diversification facts**")
        for n_ in risk.diversification_notes(conc, sectors):
            st.markdown(f"- {n_}")
        if pm.get("beta_vs_benchmark") is not None:
            st.markdown(f"- Beta vs Nifty 50 ({config.BENCHMARK_SYMBOL}): {pm['beta_vs_benchmark']:.2f}; "
                        f"correlation {pm.get('correlation_vs_benchmark', float('nan')):.2f}")
        st.caption("These describe composition and historical behaviour only – they don't label any holding safe or unsafe.")
    with b:
        if conc.get("available"):
            w = conc["weights"].head(15).iloc[::-1] * 100
            fig = go.Figure(go.Bar(x=w.values, y=w.index, orientation="h",
                                   marker=dict(color=C["accent"], cornerradius=4),
                                   hovertemplate="<b>%{y}</b><br>%{x:.2f}% of priced value<extra></extra>"))
            fig.add_vline(x=10, line=dict(color=C["muted"], dash="dot", width=1))
            st.plotly_chart(plotly_layout(fig, 380, "Concentration – top 15 weights (%)"), config={"displayModeBar": False})
    if pm.get("available"):
        v = pm["value_series"]
        dd = (v / v.cummax() - 1) * 100
        fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", line=dict(color=C["neg"], width=1.5),
                                   fillcolor="rgba(230,103,103,.15)",
                                   hovertemplate="%{x|%d %b %Y}<br>%{y:.2f}% from peak<extra></extra>"))
        st.plotly_chart(plotly_layout(fig, 260, "Drawdown of current holdings (hypothetical, %)"),
                        config={"displayModeBar": False})
    st.markdown('<div class="fp-section">Per-stock risk (Yahoo daily history)</div>', unsafe_allow_html=True)
    rt = risk.stock_risk_table(p, hist, bench)
    if len(rt):
        st.dataframe(fmt_df(rt, {c: (lambda v: num(v)) for c in rt.columns if c not in ("Stock", "Yahoo Symbol", "Days of data")}), hide_index=True)


def page_scanner():
    snap = snapshot()
    p = snap["positions"]
    syms = p["yahoo_symbol"].dropna().unique().tolist()
    st.markdown(f'<div class="fp-note">Scans ONLY the {len(p)} stocks in your holdings file. The condition score is the '
                "share of the factual checks below that are currently met (N/A checks are excluded). It is descriptive "
                "and is <b>not</b> a buy/sell recommendation or a ranking of best/worst stocks.</div>",
                unsafe_allow_html=True)
    hist = history_for(syms) if syms else {}
    sc = scanner.scan(p, hist, snap["quotes"])
    if sc.empty:
        st.info("No holdings to scan.")
        return
    view = sc.drop(columns=["_checks"]).sort_values("Stock")
    st.dataframe(fmt_df(view, {"Price": lambda v: num(v), "Day %": lambda v: pct(v), "Return %": lambda v: pct(v),
                                    "RSI14": lambda v: num(v, 1), "Vol Ratio": lambda v: num(v), "P/E": lambda v: num(v),
                                    "ROE %": lambda v: num(v), "Condition Score": lambda v: "N/A" if v is None or pd.isna(v) else f"{int(v)}"}), hide_index=True, height=min(38 * len(view) + 40, 640))
    with st.expander("Checks used by the condition score", expanded=False):
        st.dataframe(pd.DataFrame([{"Group": g, "Condition": d} for _, g, d in scanner.CHECKS]), hide_index=True)
    pick = st.selectbox("Explain a stock", view["Stock"].tolist(), key="scan_pick")
    row = sc[sc["Stock"] == pick].iloc[0]
    st.dataframe(scanner.explain(row), hide_index=True)
    st.caption(f"Flags for {pick}: {row['Flags']}. Fundamentals/news use cached Yahoo data – load them on the "
               "Fundamentals/News pages if they show N/A.")


def page_sell():
    snap = snapshot()
    lots = snap["lots"].copy()
    all_accounts = sorted(snap.get("all_accounts") or snap["summary"].get("accounts", []))
    st.markdown('<div class="fp-estimate">ESTIMATE ONLY – calculation tool. No order is placed and FinPilot has no broker '
                'connection.</div>', unsafe_allow_html=True)
    st.write("")

    account_scope = st.selectbox("Account Selection", ["All Accounts", *all_accounts], index=0,
                                 key="sell_account_scope")
    account_filter = all_accounts if account_scope == "All Accounts" else [account_scope]
    scope_lots = lots[lots["account"].isin(account_filter)] if account_filter else lots.iloc[0:0]
    valid_lots = scope_lots.loc[(scope_lots["quantity"].notna()) & (scope_lots["quantity"] > 0) & scope_lots["base_symbol"].notna()].copy()
    if valid_lots.empty:
        st.warning(f"No valid holding lots are available for {account_scope}.")
        return

    mode = st.radio("Analysis type", ["Individual holding lot", "Combined Position Across All Accounts"],
                    index=0, key="sell_analysis_mode")

    if mode == "Combined Position Across All Accounts":
        combo = sa.aggregate_lots(valid_lots)
        if combo.empty:
            st.warning(f"No combined stock positions are available for {account_scope}.")
            return
        labels = [
            f"{row.base_symbol} | Accounts: {row.accounts} | Qty: {num(row.quantity, 0)} | Weighted Avg Buy: {inr(row.weighted_avg_buy)}"
            for row in combo.itertuples(index=False)
        ]
        combo_ix = st.selectbox("Combined stock selection", range(len(combo)), format_func=lambda i: labels[i],
                                key="sell_combined_stock")
        row = combo.iloc[int(combo_ix)]
        current_price = row["current_price"]
        if current_price is None:
            st.warning("No current Yahoo price is available for this combined stock position.")
            return
        custom = st.number_input("Custom combined sell price (optional)", min_value=0.0, value=0.0, step=0.05,
                                 key="sell_combined_custom",
                                 help="Leave 0 to use the live Yahoo price.")
        qty_total = float(row["quantity"])
        invested_total = row["invested_value"]
        sell_price = custom if custom > 0 else current_price
        estimate = sa.estimate_sale(
            {"base_symbol": row["base_symbol"], "yahoo_symbol": row["yahoo_symbol"],
             "quantity": qty_total, "avg_price": row["weighted_avg_buy"]},
            qty_total, sell_price,
            "Custom estimated sell price" if custom > 0 else "Current Yahoo price",
        )
        sale_value = estimate.get("estimated_sale_value")
        invested_total = estimate.get("invested_amount")
        pnl = estimate.get("estimated_pnl")
        return_pct = estimate.get("return_pct")
        st.caption(f"Combined position across {row['accounts']} · weighted average cost {inr(row['weighted_avg_buy'])} · Yahoo price {inr(current_price)} · sell price used {inr(sell_price)}")
        cols = st.columns(4)
        cards = [
            card("Total Quantity", num(qty_total, 0), f"across {row['accounts']}"),
            card("Weighted Avg Buy", inr(row["weighted_avg_buy"]), "total invested ÷ total quantity"),
            card("Invested Amount", inr(invested_total, 0), "sum of all lot costs"),
            card("Estimated Sale Value", inr(sale_value, 0), f"{num(qty_total, 0)} shares × {inr(sell_price)}", cls(sale_value)),
        ]
        for c, h in zip(cols, cards):
            c.markdown(h, unsafe_allow_html=True)
        st.write("")
        cols = st.columns(4)
        cards = [
            card("Estimated P&L", inr(pnl, 0, True), f"Return {pct(return_pct)}", cls(pnl)),
            card("Return %", pct(return_pct), "profit/loss ÷ total invested"),
            card("Sell Price Used", inr(sell_price), row["yahoo_symbol"] or "symbol unresolved"),
            card("Scope", account_scope, "combined across selected account(s)"),
        ]
        for c, h in zip(cols, cards):
            c.markdown(h, unsafe_allow_html=True)
        st.caption(sa.ESTIMATE_LABEL + " Combined analysis aggregates all relevant lots for the selected stock in the current scope.")
        return

    lot_rows = valid_lots.sort_values(["base_symbol", "account", "source_row"], kind="mergesort").reset_index(drop=True)
    lot_labels = [
        f"{row.base_symbol} | Account: {row.account} | Qty: {num(row.quantity, 0)} | Avg Buy: {inr(row.avg_price)}"
        for row in lot_rows.itertuples(index=False)
    ]
    lot_ix = st.selectbox("Select holding lot", range(len(lot_rows)), format_func=lambda idx: lot_labels[idx],
                          key="sell_lot_choice")
    chosen = lot_rows.iloc[int(lot_ix)]

    c1, c2, c3 = st.columns([1.6, 1, 1])
    with c1:
        st.caption(f"Selected lot: {chosen['base_symbol']} · Account {chosen['account']} · Qty {num(chosen['quantity'], 0)} · Avg Buy {inr(chosen['avg_price'])}")
    with c2:
        qty = st.number_input(f"Quantity to sell (held {chosen['quantity']:,.0f})", min_value=0.0, max_value=float(chosen['quantity']),
                              value=min(float(chosen['quantity']), 100.0), step=1.0, key=f"sell_qty_{chosen['lot_id']}")
    with c3:
        custom = st.number_input("Custom estimated sell price (optional)", min_value=0.0, value=0.0, step=0.05,
                                 key=f"sell_custom_{chosen['lot_id']}",
                                 help="Leave 0 to use only the Yahoo price")

    @st.fragment(run_every=REFRESH)
    def live():
        s2 = snapshot()
        scope_lots_now = s2["lots"] if account_scope == "All Accounts" else s2["lots"][s2["lots"]["account"] == account_scope]
        lot_now = scope_lots_now[scope_lots_now["lot_id"] == chosen["lot_id"]]
        if lot_now.empty:
            st.warning("This holding lot is no longer available in the selected account scope.")
            return
        current_lot = lot_now.iloc[0]
        price = current_lot["current_price"] if pd.notna(current_lot["current_price"]) else None
        res = sa.estimate_lot(current_lot, qty, price, "Current Yahoo price")
        if custom > 0:
            custom_est = sa.estimate_lot(current_lot, qty, custom, "Custom estimated sell price")
            res = {"current": res, "custom": custom_est}
        else:
            res = {"current": res}
        st.session_state["last_sell_analysis"] = res
        q = s2["quotes"].get(current_lot["yahoo_symbol"]) or {}
        st.caption(f"Yahoo price {inr(price)} · status {q.get('status', 'Unavailable')} · Yahoo data time "
                   f"{md.fmt_time(q.get('data_timestamp'))} · recalculated {md.fmt_time(s2['generated_at'])}")
        for est_key in ["current", "custom"] if custom > 0 else ["current"]:
            est = res[est_key]
            st.markdown(f"**{est['price_label']}: {inr(est['price_used'])}**")
            if not est["valid"]:
                st.warning(est["message"])
                continue
            cols = st.columns(4)
            cards = [
                card("Account", est.get("account") or "N/A", f"stock {est['stock']}"),
                card("Avg Buy Price", inr(est["avg_buy_price"]), f"held {num(est['held_quantity'], 0)} shares"),
                card("Estimated Sale Value", inr(est["estimated_sale_value"], 0), f"{num(est['sell_quantity'], 0)} shares"),
                card("Invested Amount", inr(est["invested_amount"], 0), "cost of shares sold"),
            ]
            for c, h in zip(cols, cards):
                c.markdown(h, unsafe_allow_html=True)
            st.write("")
            cols = st.columns(4)
            cards = [
                card("Estimated P&L", inr(est["estimated_pnl"], 0, True), f"Return {pct(est['return_pct'])}", cls(est["estimated_pnl"])),
                card("Return %", pct(est["return_pct"]), "P&L ÷ invested × 100"),
                card("Remaining Quantity", num(est["remaining_quantity"], 0), f"of {num(est['held_quantity'], 0)} held"),
                card("Remaining Estimated Value", inr(est["remaining_estimated_value"], 0), "at price used"),
            ]
            for c, h in zip(cols, cards):
                c.markdown(h, unsafe_allow_html=True)
            st.write("")
        st.caption(sa.ESTIMATE_LABEL + " Return % is N/A when the cost basis is zero.")

    live()

    if st.button("📩 Send Sell Analysis to Telegram", key="send_sell_analysis_to_telegram"):
        s2 = snapshot()
        scope_lots_now = s2["lots"] if account_scope == "All Accounts" else s2["lots"][s2["lots"]["account"] == account_scope]
        lot_now = scope_lots_now[scope_lots_now["lot_id"] == chosen["lot_id"]]
        if lot_now.empty:
            st.warning("Sell analysis cannot be sent because the selected holding lot is not available in the current scope.")
        else:
            lot_row = lot_now.iloc[0]
            est = st.session_state.get("last_sell_analysis", {}).get("custom") or st.session_state.get("last_sell_analysis", {}).get("current")
            if not est or not est.get("valid"):
                st.warning("Create a valid sell analysis before sending it to Telegram.")
            elif lot_row.get("current_price") is None or pd.isna(lot_row.get("current_price")):
                st.warning("Sell analysis cannot be sent because current market data is unavailable.")
            else:
                selected_sell_price = est.get("price_used")
                price_label = "Estimated Sell Price: Current Yahoo LTP" if custom <= 0 else "Estimated Sell Price"
                ok, msg = tg.send_sell_analysis_report(
                    account=lot_row.get("account"),
                    stock=lot_row.get("base_symbol"),
                    yahoo_symbol=lot_row.get("yahoo_symbol"),
                    quantity=float(est.get("sell_quantity", qty)),
                    avg_buy_price=est.get("avg_buy_price"),
                    current_price=float(lot_row["current_price"]),
                    estimated_sale_value=est.get("estimated_sale_value"),
                    invested_value=est.get("invested_amount"),
                    return_pct=est.get("return_pct"),
                    estimated_pnl=est.get("estimated_pnl"),
                    market_status=(s2.get("market", {}).get("status", "Unavailable")
                                   if isinstance(s2.get("market"), dict) else "Unavailable"),
                    data_status=(s2.get("data_status", {}).get("state", "Unavailable")
                                 if isinstance(s2.get("data_status"), dict) else "Unavailable"),
                    estimated_sell_price=("Current Yahoo LTP" if custom <= 0 else
                                          float(selected_sell_price) if selected_sell_price is not None else None),
                    price_label=price_label,
                )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg or "Telegram report failed. Check Telegram configuration.")

    if st.button("Generate PDF with this sell analysis"):
        path = rp.generate_pdf(snapshot(), trigger="manual-sell", include_sell=st.session_state.get("last_sell_analysis"))
        st.success(f"Saved {path.name} in reports/")
        st.download_button("Download PDF", path.read_bytes(), file_name=path.name, mime="application/pdf")


def page_alerts():
    snap = snapshot()
    p = snap["positions"]
    tg = "Configured" if config.TELEGRAM_ENABLED else "Not configured"
    cols = st.columns(4)
    cols[0].markdown(card("Telegram", tg, "notifications only – never trading",
                          "pos" if config.TELEGRAM_ENABLED else "neg"), unsafe_allow_html=True)
    cols[1].markdown(card("Day-move alert", f"±{config.ALERT_DAY_MOVE_PCT:g}%", "per step, once per day"), unsafe_allow_html=True)
    cols[2].markdown(card("Volume alert", f"{config.ALERT_VOLUME_RATIO:g}× avg", "once per stock per day"), unsafe_allow_html=True)
    cols[3].markdown(card("Portfolio alert", f"±{config.ALERT_PORTFOLIO_MOVE_PCT:g}%", f"P&L step ±{config.ALERT_PNL_MOVE_PCT:g}%"),
                     unsafe_allow_html=True)
    st.caption("Thresholds are set in .env. Alerts are deduplicated in the database and combined into one Telegram "
               "message per cycle; price-based alerts use live Yahoo data only.")

    st.markdown('<div class="fp-section">Target price alerts</div>', unsafe_allow_html=True)
    with st.form("target_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1.6])
        stock = c1.selectbox("Stock", sorted(p["base_symbol"].tolist()) if len(p) else [])
        direction = c2.selectbox("When price is", ["above", "below"])
        price = c3.number_input("Target price (₹)", min_value=0.0, step=0.5)
        note = c4.text_input("Note (optional)")
        if st.form_submit_button("Add target alert") and stock and price > 0:
            al.add_target_rule(stock, direction, price, note)
            st.success(f"Alert added: {stock} {direction} ₹{price:,.2f}")
    rules = al.list_rules()
    if len(rules):
        for r in rules.itertuples():
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.markdown(f"**{r.base_symbol}** {r.direction} ₹{r.threshold:,.2f} {('· ' + r.note) if r.note else ''} "
                        f"<span class='fp-note'>({'active' if r.active else 'paused'})</span>", unsafe_allow_html=True)
            if c2.button("Pause" if r.active else "Resume", key=f"tog_{r.rule_id}"):
                al.set_rule_active(r.rule_id, not r.active)
                st.rerun()
            if c3.button("Delete", key=f"del_{r.rule_id}"):
                al.delete_rule(r.rule_id)
                st.rerun()
    else:
        st.caption("No target alerts yet.")

    @st.fragment(run_every=REFRESH)
    def live_alerts():
        st.markdown('<div class="fp-section">Recent alerts</div>', unsafe_allow_html=True)
        df = al.recent_alerts(100)
        if df.empty:
            st.caption("No alerts raised yet.")
        else:
            show = df[["created_at", "alert_type", "base_symbol", "message", "telegram_sent"]].rename(columns={
                "created_at": "Time", "alert_type": "Type", "base_symbol": "Stock", "message": "Details",
                "telegram_sent": "Telegram"})
            show["Telegram"] = show["Telegram"].map({1: "Sent", 0: "Not sent"})
            st.dataframe(show, hide_index=True, height=320)
        ms = MONITOR.status()
        st.caption(f"Background monitor: {'running' if ms['running'] else 'stopped'} · last cycle "
                   f"{md.fmt_time(ms['last_tick'])}" + (f" · error: {ms['last_error']}" if ms["last_error"] else ""))
        if ms["events"]:
            with st.expander("Monitor activity log"):
                st.code("\n".join(ms["events"]))

    live_alerts()


def page_ai():
    st.caption("Answers come from your holdings file and Yahoo Finance data calculated by FinPilot. "
               "If a number is unavailable it says so. It never places trades.")
    if "chat" not in st.session_state:
        st.session_state.chat = [("assistant", "Hi! Ask me about your portfolio – e.g. *What is my current portfolio "
                                               "value?* or *What is the estimated P&L if I sell 100 NMDC?*")]
    ex = st.columns(4)
    examples = ["What is my current portfolio value?", "Which stocks have negative P&L?",
                "Show my portfolio sector exposure.", "Show recent important news."]
    clicked = None
    for c, e in zip(ex, examples):
        if c.button(e, width="stretch"):
            clicked = e
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)
    q = st.chat_input("Ask FinPilot…") or clicked
    if q:
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Checking your data…"):
                ans = FinPilotAgent(snapshot()).answer(q)
            st.markdown(ans)
        st.session_state.chat.append(("assistant", ans))


def page_reports():
    snap = snapshot()
    st.markdown('<div class="fp-section">PDF report</div>', unsafe_allow_html=True)
    secs = st.multiselect("Sections", ["summary", "holdings", "technical", "fundamentals", "news", "risk", "scanner"],
                          default=["summary", "holdings", "technical", "fundamentals", "news", "risk", "scanner"])
    c1, c2 = st.columns(2)
    if c1.button("Generate PDF report", type="primary", width="stretch"):
        with st.spinner("Building PDF…"):
            path = rp.generate_pdf(snap, trigger="manual", include_sections=set(secs))
        st.session_state["last_pdf"] = str(path)
        st.success(f"Saved {path.name} in reports/")
    if c2.button("Generate PDF and send to Telegram", width="stretch", disabled=not config.TELEGRAM_ENABLED):
        path = rp.generate_pdf(snap, trigger="manual-telegram", include_sections=set(secs))
        ok, d = al.send_telegram_document(path, f"{config.PROJECT_NAME} report {md.fmt_datetime(snap['generated_at'])}")
        (st.success if ok else st.error)(f"Telegram: {d}")
    reps = rp.list_reports()
    if len(reps):
        st.markdown("**Saved reports**")
        for r in reps.head(15).itertuples():
            a, b = st.columns([4, 1])
            a.markdown(f"{Path(r.file_path).name} <span class='fp-note'>· {r.trigger} · {r.created_at}</span>",
                       unsafe_allow_html=True)
            b.download_button("Download", Path(r.file_path).read_bytes(), file_name=Path(r.file_path).name,
                              mime="application/pdf", key=f"dl_{r.report_id}")

    st.markdown('<div class="fp-section">Telegram reports (sent on request)</div>', unsafe_allow_html=True)
    if not config.TELEGRAM_ENABLED:
        st.info("Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env to enable Telegram. The group invite link is "
                "not a token – see README steps in .env.example.")
    syms = snap["positions"]["yahoo_symbol"].dropna().tolist()
    builders = {
        "Market-open report": lambda: rp.market_open_message(snap),
        "Closing report": lambda: rp.closing_message(snap, history_for(syms) if syms else {}),
        "Four-stock performance": lambda: rp.four_stock_message(snap),
        "Startup message": lambda: rp.startup_message(snap),
    }
    choice = st.selectbox("Report", list(builders), key="tg_choice")
    text = builders[choice]()
    with st.expander("Preview"):
        st.code(text.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "").replace("&amp;", "&"))
    if choice == "Four-stock performance" and not config.REPORT_STOCKS:
        st.caption("Configure REPORT_STOCKS=SYM1,SYM2,SYM3,SYM4 in .env – FinPilot doesn't pick them for you.")
    if st.button("Send to Telegram now", disabled=not config.TELEGRAM_ENABLED):
        ok, d = al.send_telegram(text, choice)
        (st.success if ok else st.error)(d)
    st.caption("Scheduled: startup (once per start), market-open report "
               f"({config.TELEGRAM_OPEN_REPORT_TIME} IST) and closing report ({config.TELEGRAM_CLOSE_REPORT_TIME} IST) "
               "once per trading day, plus deduplicated alerts. Never every 10 seconds.")


def render_validation_panel(snap: dict) -> None:
    """Developer / data-quality panel – makes any future value discrepancy traceable."""
    s, q, ds = snap["summary"], snap["quality"], snap["data_status"]
    st.markdown('<div class="fp-section">Data validation</div>', unsafe_allow_html=True)
    cov = f"{s['coverage_pct']:.2f}%" if s["coverage_pct"] is not None else "N/A"
    rows = [
        ("Portfolio source", q["source_file"] or "missing"),
        ("Total holdings (valid lots)", s["total_lots"]),
        ("Unique stocks", s["unique_stocks"]),
        ("Priced holdings", s["priced_holdings"]),
        ("Unavailable holdings", s["unavailable_holdings"]),
        ("Unavailable symbols", ", ".join(s["unavailable_symbols"]) or "none"),
        ("Stale (last stored) prices", s["stale_holdings"]),
        ("Coverage", cov),
        ("Status", s["status"]),
        ("Duplicate rows (same account + symbol)", q["duplicate_rows"]),
        ("Symbols held in several accounts", ", ".join(q["symbols_in_multiple_accounts"]) or "none"),
        ("Invalid quantities", q["invalid_quantities"]),
        ("Invalid / missing buy prices", q["invalid_buy_prices"]),
        ("Missing symbols", q["missing_symbols"]),
        ("Summary / label rows skipped", q["label_rows"]),
        ("Unresolved Yahoo symbols", ", ".join(q["unresolved_symbols"]) or "none"),
        ("Yahoo prices OK / failed (snapshot)", f"{q['yahoo_ok']} / {q['yahoo_failed']}"),
        ("Yahoo failed symbols", ", ".join(q["yahoo_failed_symbols"]) or "none"),
        ("Yahoo last refresh OK / failed", f"{ds.get('last_ok_count', 0)} / {ds.get('last_failed_count', 0)}"),
        ("Yahoo data timestamp", md.fmt_datetime(snap["yahoo_timestamp"])),
        ("Price basis", (snap.get("price_basis") or {}).get("label", "N/A")),
        ("Exchange holiday calendar", (lambda h: f"{h['count_this_year']} NSE holidays loaded for {h['year']}"
                                       if h["covered"] else f"NOT configured for {h['year']} – add MARKET_HOLIDAYS in .env")
         (md.holiday_calendar_status())),
        ("Telegram token exposure", (MONITOR.token_scan or {}).get("note", "scan runs at startup")
         + (": " + ", ".join(MONITOR.token_scan["hits"]) if MONITOR.token_scan and MONITOR.token_scan["hits"] else "")),
        ("Invested value", inr(s["invested_value"])),
        ("Current value (priced holdings)", inr(s["current_value"])),
        ("Broker file value (Qty × LTP column in Excel – NOT used)",
         inr(q["broker_file_value"]) if q["broker_file_value"] is not None else "no LTP column"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Item", "Value"]).astype(str), hide_index=True)
    if q["broker_file_value"] is not None and s["current_value"] is not None and not s["is_partial"]:
        st.caption(f"Difference Yahoo vs broker-file value: {inr(s['current_value'] - q['broker_file_value'], 0, True)}. "
                   "The broker LTP column is the price at the time the Excel was exported; FinPilot values holdings "
                   "at the latest Yahoo price, so the two differ by the price movement since the export.")
    checks = pd.DataFrame(snap["validation"])
    if len(checks):
        checks["Result"] = checks["ok"].map({True: "PASS", False: "FAIL"})
        st.markdown("**Automated checks (run on every snapshot)**")
        st.dataframe(checks[["check", "Result", "detail"]], hide_index=True)
    st.markdown("**Symbol → Yahoo → price → status**")
    st.dataframe(fmt_df(pf.symbol_price_log(snap), {"Price": lambda v: num(v), "Previous close": lambda v: num(v)}),
                 hide_index=True)


def page_data():

    snap = snapshot()
    load = snap["load"]
    c = st.columns(4)
    c[0].markdown(card("Holdings file", Path(load.source_file).name if load.source_file else "Missing",
                       f"{load.header_rows} header row(s) detected"), unsafe_allow_html=True)
    c[1].markdown(card("Valid lots", str(len(load.lots)), f"{snap['summary']['unique_stocks']} unique stocks"),
                  unsafe_allow_html=True)
    c[2].markdown(card("Excluded rows", str(len(load.invalid_rows)), "reason shown below"), unsafe_allow_html=True)
    ds = snap["data_status"]
    c[3].markdown(card("Yahoo Finance", ds["state"], f"failures in a row: {ds['failures']}"), unsafe_allow_html=True)
    if load.error:
        st.error(load.error)
    st.markdown('<div class="fp-section">Yahoo symbol resolution</div>', unsafe_allow_html=True)
    st.caption("Each symbol is verified against Yahoo (NSE `.NS` first, then BSE `.BO`) – never blindly suffixed. "
               "Use SYMBOL_OVERRIDES in .env to force a mapping.")
    res = pd.DataFrame([{"Portfolio symbol": v["portfolio_symbol"], "Base": k, "Yahoo symbol": v["yahoo_symbol"] or "—",
                         "Status": v["status"], "How": v["method"]} for k, v in snap["resolution"].items()])
    st.dataframe(res, hide_index=True)
    if st.button("Re-check unresolved symbols now"):
        base_to_ps = {k: v["portfolio_symbol"] for k, v in snap["resolution"].items()}
        md.resolve_symbols(base_to_ps, force=True)
        st.rerun()
    if load.invalid_rows:
        st.markdown('<div class="fp-section">Rows excluded from holdings input</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(load.invalid_rows), hide_index=True)
    render_validation_panel(snap)
    st.markdown('<div class="fp-section">Lots (as read from Excel)</div>', unsafe_allow_html=True)
    lots = snap["lots"][["account", "portfolio_symbol", "yahoo_symbol", "quantity", "avg_price", "invested_value",
                         "current_price", "current_value", "pnl", "return_pct", "zero_cost", "source_row",
                         "price_status"]]
    st.dataframe(fmt_df(lots, {"quantity": lambda v: num(v, 0), "avg_price": lambda v: num(v),
                                    "invested_value": lambda v: inr(v, 0), "current_price": lambda v: num(v),
                                    "current_value": lambda v: inr(v, 0), "pnl": lambda v: inr(v, 0, True),
                                    "return_pct": lambda v: pct(v)}), hide_index=True)
    st.markdown('<div class="fp-section">Configuration</div>', unsafe_allow_html=True)
    st.json({"PROJECT_NAME": config.PROJECT_NAME, "MARKET_REFRESH_SECONDS": REFRESH, "TIMEZONE": config.TIMEZONE,
             "MARKET_DATA": config.MARKET_DATA_PROVIDER, "HOLDINGS_FILE": str(config.HOLDINGS_FILE),
             "DB_PATH": str(config.DB_PATH), "HISTORY_PERIOD": config.HISTORY_PERIOD,
             "MARKET_HOURS": f"{config.MARKET_OPEN_TIME}–{config.MARKET_CLOSE_TIME} IST",
             "MARKET_HOLIDAYS_CONFIGURED": len(config.MARKET_HOLIDAYS), "HOLIDAY_YEARS": config.HOLIDAY_YEARS,
             "TELEGRAM": "configured" if config.TELEGRAM_ENABLED else "not configured",
             "REPORT_STOCKS": config.REPORT_STOCKS, "LLM_FALLBACK": config.LLM_ENABLED})
    counts = {t: db.query_one(f"SELECT COUNT(*) n FROM {t}")["n"] for t in
              ("holdings_lots", "symbol_map", "price_snapshots", "portfolio_snapshots", "technical_snapshots",
               "fundamentals", "news", "alerts", "telegram_log", "reports")}
    st.caption("Database rows: " + " · ".join(f"{k} {v}" for k, v in counts.items()))


ROUTES = {"Overview": page_overview, "Technical Analysis": page_technical, "Fundamentals": page_fundamentals,
          "News": page_news, "Risk": page_risk, "Scanner": page_scanner, "Sell Analysis": page_sell,
          "Alerts": page_alerts, "AI Assistant": page_ai, "Reports": page_reports, "Data & Settings": page_data}
ROUTES[page]()
