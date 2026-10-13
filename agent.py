"""
FinPilot AI agent – answers questions from ACTUAL calculated data.

How it works
------------
1. Every answer starts from a fresh snapshot (Excel holdings + Yahoo data).
2. Built-in intent rules answer the common questions directly from those
   numbers (portfolio value, P&L, positions, prices, sell estimates, sector
   exposure, news, technicals, fundamentals, risk).
3. Optionally (ANTHROPIC_API_KEY + LLM_MODEL in .env) questions the rules
   don't cover are sent to an LLM together with FinPilot's calculated data
   ONLY, with instructions not to invent numbers.
4. If data is unavailable the agent says so. It never fabricates a figure and
   it never places or suggests executing trades.

CLI
---
    python agent.py                      interactive Q&A in the terminal
    python agent.py "what is my portfolio value?"
    python agent.py --monitor            headless monitor (alerts + Telegram reports)
    python agent.py --telegram-chat-id   list chats your bot can see (setup helper)
    python agent.py --telegram-check     verify the bot token + scan for token leaks (token never printed)
    python agent.py --send startup|open|close|four|pdf|test
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import pandas as pd

import config
from modules import database as db
from modules import fundamentals as fnd
from modules import market_data as md
from modules import news as nws
from modules import portfolio as pf
from modules import reports as rp
from modules import risk
from modules import sell_analysis as sa
from modules import technical as ta

inr, pct, num = rp.inr, rp.pct, rp.num

HELP = """I answer from your holdings file and Yahoo Finance data. Try:
- What is my current portfolio value?
- Which stocks have negative P&L?
- What is my COFFEEDAY position?
- What is the current price of NMDC?
- What is the estimated P&L if I sell 100 shares of NMDC? (add "at 90" for a custom price)
- Show my portfolio sector exposure.
- Show recent important news.
- Today's top gainers / losers
- Technicals for KAYNES · Fundamentals of DMART · Portfolio risk / concentration
I never place trades."""


class FinPilotAgent:
    def __init__(self, snapshot: dict | None = None):
        self.snap = snapshot

    # ------------------------------------------------------------ helpers
    def _snapshot(self) -> dict:
        if self.snap is None:
            self.snap = pf.build_snapshot()
        return self.snap

    def _positions(self) -> pd.DataFrame:
        return self._snapshot()["positions"]

    def _find_symbols(self, question: str) -> list[pd.Series]:
        p = self._positions()
        if p.empty:
            return []
        q_up = question.upper()
        tokens = set(re.findall(r"[A-Z0-9&\-\.]+", q_up))
        found, seen = [], set()
        for _, r in p.iterrows():
            names = {r["base_symbol"], *(r["portfolio_symbol"].split(", "))}
            if r["yahoo_symbol"]:
                names.add(r["yahoo_symbol"])
            if names & tokens and r["base_symbol"] not in seen:
                found.append(r)
                seen.add(r["base_symbol"])
        if not found:  # try company names from cached fundamentals
            for _, r in p.iterrows():
                f = fnd.get_cached(r["yahoo_symbol"]) if r["yahoo_symbol"] else None
                name = (f or {}).get("company_name") or ""
                first = re.sub(r"(?i)\b(limited|ltd\.?|india|the)\b", "", name).strip().upper()
                if first and len(first) > 3 and first.split()[0] in q_up and r["base_symbol"] not in seen:
                    found.append(r)
                    seen.add(r["base_symbol"])
        return found

    def _data_note(self) -> str:
        s, ds = self._snapshot()["summary"], self._snapshot()["data_status"]
        notes = []
        if s["is_partial"]:
            notes.append(f"PARTIAL: Yahoo prices for {s['priced_holdings']}/{s['total_lots']} holdings; "
                         f"unavailable: {', '.join(s['unavailable_symbols'])} (excluded, not counted as zero)")
        if ds["state"] != "CONNECTED":
            notes.append(f"Yahoo Finance status: {ds['state']}" + (f" ({ds['last_error']})" if ds["last_error"] else ""))
        ts = self._snapshot()["yahoo_timestamp"]
        notes.append(f"Yahoo data time: {md.fmt_datetime(ts)}")
        return "\n\n_" + " · ".join(notes) + "_"

    # ------------------------------------------------------------ main entry
    def answer(self, question: str) -> str:
        q = question.strip()
        ql = q.lower()
        if not q or ql in {"help", "?", "what can you do", "hi", "hello"}:
            return HELP
        if re.search(r"place (an? )?(buy |sell )?(order|trade)|\bexecute\b|\b(buy|sell)\b.*\bfor me\b|^buy\b", ql):
            return ("I can't place, modify or execute any trade – FinPilot is monitoring and analysis only. "
                    "I can show a sell *estimate* (e.g. 'estimated P&L if I sell 100 NMDC').")
        snap = self._snapshot()
        if snap["load"].error and snap["positions"].empty:
            return f"I couldn't load your holdings: {snap['load'].error}"
        syms = self._find_symbols(q)

        if re.search(r"\bsell\b", ql):
            return self._sell(ql, syms)
        if "account" in ql:
            return self._accounts()
        if "sector" in ql and not syms:
            return self._sectors()
        if "news" in ql:
            return self._news(ql, syms)
        if any(w in ql for w in ("technical", "rsi", "sma", "ema", "macd", "bollinger", "support", "resistance", "vwap")):
            return self._technical(syms)
        if any(w in ql for w in ("fundamental", "p/e", " pe ", "pe ratio", "roe", "market cap", "eps", "dividend", "sector")):
            return self._fundamentals(syms)
        if any(w in ql for w in ("risk", "concentration", "diversif", "volatility", "drawdown")):
            return self._risk()
        if "today" in ql and any(w in ql for w in ("gainer", "loser", "mover", "top", "worst", "best", "up", "down")):
            return self._movers()
        if any(w in ql for w in ("negative", "loss", "losing", "in red", "red ")):
            return self._pnl_list(negative=True)
        if any(w in ql for w in ("positive p&l", "profit", "in green", "gaining", "positive")) and not syms:
            return self._pnl_list(negative=False)
        if "market" in ql and any(w in ql for w in ("open", "status", "closed")):
            m = snap["market"]
            return (f"Market (schedule-based, IST): **{m['status']}** – {m['detail']}. "
                    f"Local clock: {md.fmt_datetime(snap['generated_at'])}. "
                    f"Yahoo data: {snap['data_status']['state']}.")
        if syms and any(w in ql for w in ("price", "trading at", "quote", "ltp")) and not any(
                w in ql for w in ("position", "average", "avg", "buy price", "holding", "p&l", "pnl", "invested")):
            return "\n\n".join(self._price(r) for r in syms)
        if syms:
            return "\n\n".join(self._position(r) for r in syms) + self._data_note()
        if "price" in ql:
            return "Which stock? For example: *What is the current price of NMDC?*"
        if any(w in ql for w in ("portfolio", "value", "worth", "invested", "p&l", "pnl", "return", "today", "summary",
                                 "holding", "how am i")):
            return self._summary()
        return self._llm(q) or ("I couldn't match that to FinPilot data. " + HELP)

    # ------------------------------------------------------------ intents
    def _summary(self) -> str:
        s = self._snapshot()["summary"]
        return (f"**Portfolio summary**\n\n"
                f"- Current value: **{inr(s['current_value'])}**\n"
                f"- Invested value: {inr(s['invested_value'])}\n"
                f"- Total P&L: {inr(s['total_pnl'], signed=True)} ({pct(s['return_pct'])})\n"
                f"- Today's P&L: {inr(s['todays_pnl'], signed=True)} ({pct(s['todays_pct'])})\n"
                f"- Holdings: {s['total_lots']} lots ({s['unique_stocks']} unique stocks) across "
                f"{len(s.get('accounts', []))} account(s)" + self._data_note())

    def _accounts(self) -> str:
        acc = self._snapshot()["accounts_table"]
        if acc.empty:
            return "No accounts loaded."
        lines = ["**Account-wise portfolio**"]
        for r in acc.to_dict("records"):
            today = r["Today's P&L"]
            lines.append(f"- {r['Account']}: value {inr(r['Current Value'], 0)} · invested {inr(r['Invested Value'], 0)} · "
                         f"P&L {inr(r['P&L'], 0, True)} ({pct(r['Return %'])}) · today {inr(today, 0, True)} · "
                         f"{r['Priced']}/{r['Holdings']} priced")
        return "\n".join(lines) + self._data_note()

    def _position(self, r: pd.Series) -> str:
        zero = " (includes a zero-cost lot)" if r["has_zero_cost_lot"] else ""
        return (f"**{r['base_symbol']}** ({r['yahoo_symbol'] or 'Yahoo symbol unresolved'}) – accounts: {r['accounts']}\n"
                f"- Quantity: {num(r['quantity'], 0)} in {r['lots']} lot(s){zero}\n"
                f"- Average buy price: {inr(r['avg_price'])}\n"
                f"- Current price: {inr(r['current_price'])} ({r['price_status']})\n"
                f"- Invested: {inr(r['invested_value'])} · Current value: {inr(r['current_value'])}\n"
                f"- P&L: {inr(r['pnl'], signed=True)} ({pct(r['return_pct'])})\n"
                f"- Today: {pct(r['day_change_pct'])} · Today's P&L {inr(r['todays_pnl'], signed=True)}")

    def _price(self, r: pd.Series) -> str:
        qd = self._snapshot()["quotes"].get(r["yahoo_symbol"]) if r["yahoo_symbol"] else None
        if not qd or qd.get("price") is None:
            return f"{r['base_symbol']}: current price is **unavailable** from Yahoo Finance right now."
        return (f"**{r['base_symbol']}** ({r['yahoo_symbol']}): {inr(qd['price'])} "
                f"({pct(r['day_change_pct'])} vs prev close {inr(qd.get('previous_close'))}). "
                f"Open {num(qd.get('open'))}, High {num(qd.get('day_high'))}, Low {num(qd.get('day_low'))}, "
                f"Volume {num(qd.get('volume'), 0)}. Status: {qd['status']}. "
                f"Yahoo data time: {md.fmt_time(qd.get('data_timestamp'))}.")

    def _sell(self, ql: str, syms: list[pd.Series]) -> str:
        if not syms:
            return "Which stock? e.g. *estimated P&L if I sell 100 shares of NMDC*."
        r = syms[0]
        m = re.search(r"sell\s+(all|[\d,]+)", ql)
        qty = r["quantity"] if (m and m.group(1) == "all") or " all " in f" {ql} " else \
            float(m.group(1).replace(",", "")) if m else None
        if qty is None:
            m2 = re.search(r"([\d,]+)\s*(shares|qty|units)", ql)
            qty = float(m2.group(1).replace(",", "")) if m2 else None
        if qty is None:
            return f"How many {r['base_symbol']} shares? You hold {num(r['quantity'], 0)}."
        custom = None
        m3 = re.search(r"\bat\s*(?:rs\.?|₹|inr)?\s*([\d,]+(?:\.\d+)?)", ql)
        if m3:
            custom = float(m3.group(1).replace(",", ""))
        res = sa.analyse(r, qty, r["current_price"] if pd.notna(r["current_price"]) else None, custom)
        out = [f"**Sell analysis – {r['base_symbol']}** · _{sa.ESTIMATE_LABEL}_"]
        for est in res.values():
            if not est["valid"]:
                out.append(f"- {est['price_label']}: {est['message']}")
                continue
            out.append(f"\n**{est['price_label']}: {inr(est['price_used'])}**\n"
                       f"- Selling {num(est['sell_quantity'], 0)} of {num(est['held_quantity'], 0)} · avg buy {inr(est['avg_buy_price'])}\n"
                       f"- Estimated sale value: {inr(est['estimated_sale_value'])}\n"
                       f"- Invested amount: {inr(est['invested_amount'])}\n"
                       f"- Estimated P&L: **{inr(est['estimated_pnl'], signed=True)}** ({pct(est['return_pct'])})\n"
                       f"- Remaining: {num(est['remaining_quantity'], 0)} shares ≈ {inr(est['remaining_estimated_value'])}")
        rem = sa.remaining_portfolio_value(self._snapshot()["summary"], qty,
                                           r["current_price"] if pd.notna(r["current_price"]) else None)
        if rem is not None and res["current"]["valid"]:
            out.append(f"\nRemaining portfolio value (at current prices): {inr(rem)}")
        return "\n".join(out)

    def _pnl_list(self, negative: bool) -> str:
        p = self._positions()
        p = p[p["pnl"].notna()]
        sel = p[p["pnl"] < 0].sort_values("pnl") if negative else p[p["pnl"] > 0].sort_values("pnl", ascending=False)
        label = "negative" if negative else "positive"
        if sel.empty:
            unpriced = int(self._positions()["pnl"].isna().sum())
            return f"No holdings with {label} P&L among priced holdings." + (
                f" ({unpriced} holdings have no Yahoo price.)" if unpriced else "")
        lines = [f"**{len(sel)} holdings with {label} P&L** (total {inr(sel['pnl'].sum(), signed=True)}):"]
        lines += [f"- {r.base_symbol}: {inr(r.pnl, 0, True)} ({pct(r.return_pct)})" for r in sel.itertuples()]
        return "\n".join(lines) + self._data_note()

    def _movers(self) -> str:
        top, bottom = pf.top_movers(self._positions(), 5)
        if top.empty and bottom.empty:
            return "Today's change is unavailable (no fresh Yahoo price / previous close)."
        return ("**Top gainers today**\n" + "\n".join(f"- {r.base_symbol}: {pct(r.day_change_pct)} ({inr(r.todays_pnl, 0, True)})"
                                                     for r in top.itertuples()) +
                "\n\n**Top decliners today**\n" + "\n".join(f"- {r.base_symbol}: {pct(r.day_change_pct)} ({inr(r.todays_pnl, 0, True)})"
                                                         for r in bottom.itertuples()) + self._data_note())

    def _sectors(self) -> str:
        p = self._positions()
        sectors = risk.sector_exposure(p, fnd.sector_map(p["yahoo_symbol"].dropna().tolist()))
        if sectors.empty:
            return "Sector exposure needs Yahoo prices, which are unavailable right now."
        lines = ["**Sector exposure** (by current value)"] + [
            f"- {r.sector}: {r.weight_pct:.1f}% ({inr(r.current_value, 0)}, {r.holdings} holdings)" for r in sectors.itertuples()]
        if (sectors["sector"] == "Unknown").any():
            lines.append("\n_'Unknown' = fundamentals not loaded yet or sector not provided by Yahoo._")
        return "\n".join(lines)

    def _news(self, ql: str, syms: list[pd.Series]) -> str:
        yahoo = [r["yahoo_symbol"] for r in syms if r["yahoo_symbol"]] or \
            self._positions()["yahoo_symbol"].dropna().tolist()
        for s in yahoo[:5] if syms else []:
            nws.refresh_news(s)
        cat = "Important Event" if "important" in ql else None
        df = nws.get_news(yahoo, days=14, category=cat, limit=10)
        if df.empty:
            return "No Yahoo Finance news stored for these holdings in the last 14 days" + \
                   (" (news is fetched gradually in the background)." if not syms else ".")
        df = df.astype(object).where(df.notna(), None)
        base = self._positions().set_index("yahoo_symbol")["base_symbol"].to_dict()
        lines = [f"**Recent {'important ' if cat else ''}news** (Yahoo Finance; tone = keyword rule, not a prediction)"]
        for r in df.itertuples():
            pub = pd.to_datetime(r.published_at).strftime("%d %b %H:%M") if r.published_at else "N/A"
            lines.append(f"- **{base.get(r.yahoo_symbol, r.yahoo_symbol)}** · {pub} · {r.headline} "
                         f"— {r.publisher} [{r.category}, {r.sentiment}]" + (f" ([link]({r.url}))" if r.url else ""))
        return "\n".join(lines)

    def _technical(self, syms: list[pd.Series]) -> str:
        if not syms:
            return "Which stock? e.g. *technicals for NMDC*."
        out = []
        for r in syms:
            if not r["yahoo_symbol"]:
                out.append(f"{r['base_symbol']}: Yahoo symbol unresolved.")
                continue
            hist = md.get_history([r["yahoo_symbol"]]).get(r["yahoo_symbol"])
            ind = ta.compute_indicators(hist, self._snapshot()["quotes"].get(r["yahoo_symbol"]),
                                        md.get_intraday(r["yahoo_symbol"]))
            if not ind.get("bars"):
                out.append(f"{r['base_symbol']}: Yahoo history unavailable.")
                continue
            out.append(f"**{r['base_symbol']} technicals** (as of {ind.get('as_of')}, {ind['bars']} daily bars)\n"
                       f"- Price {num(ind.get('price'))} · SMA20 {num(ind.get('sma20'))} · SMA50 {num(ind.get('sma50'))} · "
                       f"SMA200 {num(ind.get('sma200'))}\n"
                       f"- EMA20 {num(ind.get('ema20'))} · EMA50 {num(ind.get('ema50'))} · RSI14 {num(ind.get('rsi14'))}\n"
                       f"- MACD {num(ind.get('macd'))} / signal {num(ind.get('macd_signal'))} / hist {num(ind.get('macd_hist'))}\n"
                       f"- Bollinger {num(ind.get('bb_lower'))} – {num(ind.get('bb_upper'))} · "
                       f"20d support {num(ind.get('support_20d'))} / resistance {num(ind.get('resistance_20d'))}\n"
                       f"- Volume ratio {num(ind.get('volume_ratio'))} · VWAP {num(ind.get('vwap'))}\n"
                       + "\n".join(f"- {o}" for o in ta.describe(ind)))
        return "\n\n".join(out) + "\n\n_Indicators describe past behaviour; they don't predict prices._"

    def _fundamentals(self, syms: list[pd.Series]) -> str:
        if not syms:
            return self._sectors()
        out = []
        for r in syms:
            if not r["yahoo_symbol"]:
                out.append(f"{r['base_symbol']}: Yahoo symbol unresolved.")
                continue
            f = fnd.get_fundamentals(r["yahoo_symbol"])
            status = (f.get("_status") or "").upper()
            if status == "UNAVAILABLE":
                out.append(f"{r['base_symbol']}: fundamentals unavailable from Yahoo ({f.get('_error')}).")
                continue
            rows = "\n".join(f"- {label}: {fnd.format_value(f.get(k), kind)}" for k, label, kind in fnd.FIELDS)
            out.append(f"**{r['base_symbol']} fundamentals** (Yahoo, {status or 'UNKNOWN'}, fetched {f.get('_fetched_at')})\n{rows}")
        return "\n\n".join(out)

    def _risk(self) -> str:
        p = self._positions()
        syms = p["yahoo_symbol"].dropna().tolist()
        conc = risk.concentration(p)
        sectors = risk.sector_exposure(p, fnd.sector_map(syms))
        hist = md.get_history(syms + [config.BENCHMARK_SYMBOL]) if syms else {}
        pm = risk.portfolio_metrics(p, hist, hist.get(config.BENCHMARK_SYMBOL))
        lines = ["**Portfolio risk (factual)**"] + [f"- {n}" for n in risk.diversification_notes(conc, sectors)]
        if pm.get("available"):
            lines += [f"- Annualised volatility: {pct(pm['volatility_ann_pct'], False)}",
                      f"- Max drawdown (1y, current holdings): {pct(pm['max_drawdown_pct'])}",
                      f"- Historical 1-day VaR 95%: {pct(pm['var_95_1d_pct'], False)} ≈ {inr(pm['var_95_1d_value'], 0)}",
                      f"- Beta vs Nifty 50: {num(pm.get('beta_vs_benchmark'))}"]
        else:
            lines.append("- Volatility/drawdown need Yahoo history (unavailable right now).")
        return "\n".join(lines)

    # ------------------------------------------------------------ optional LLM
    def _context_json(self) -> str:
        snap = self._snapshot()
        p = snap["positions"]
        cols = ["base_symbol", "yahoo_symbol", "quantity", "avg_price", "current_price", "invested_value",
                "current_value", "pnl", "return_pct", "todays_pnl", "day_change_pct", "price_status"]
        data = {
            "generated_at": snap["generated_at"].isoformat(),
            "market": {k: v for k, v in snap["market"].items() if k != "as_of"},
            "yahoo_status": snap["data_status"]["state"],
            "summary": {k: v for k, v in snap["summary"].items()},
            "positions": json.loads(p[cols].round(2).to_json(orient="records")) if not p.empty else [],
        }
        return json.dumps(data, default=str)

    def _llm(self, question: str) -> str | None:
        if not config.LLM_ENABLED:
            return None
        import requests
        system = ("You are FinPilot, a portfolio monitoring assistant. Answer ONLY from the JSON data provided. "
                  "Never invent numbers; if a value is null or missing say it is unavailable. Never give buy/sell "
                  "recommendations and never offer to place trades. Currency is INR. Be concise.")
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
                "x-api-key": config.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                "content-type": "application/json"}, json={
                "model": config.LLM_MODEL, "max_tokens": 800, "system": system,
                "messages": [{"role": "user", "content": f"DATA:\n{self._context_json()}\n\nQUESTION: {question}"}]})
            data = r.json()
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            return text or None
        except Exception:
            return None


# ================================================================ CLI
def _cli():
    ap = argparse.ArgumentParser(description=f"{config.PROJECT_NAME} agent (analysis only – never trades)")
    ap.add_argument("question", nargs="*", help="ask a single question")
    ap.add_argument("--monitor", action="store_true", help="run the headless monitor (alerts + Telegram reports)")
    ap.add_argument("--telegram-chat-id", action="store_true", help="list chats visible to your Telegram bot")
    ap.add_argument("--telegram-check", action="store_true",
                    help="verify the bot token (getMe) and scan the project for token leaks – never prints the token")
    ap.add_argument("--send", choices=["startup", "open", "close", "four", "pdf", "test"],
                    help="send a report to Telegram now")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):   # Windows consoles: print ₹ and emoji safely
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    db.init_db()

    if args.telegram_check:
        from modules.alerts import check_telegram_token, token_exposure_scan
        ok, msg = check_telegram_token()
        print(("OK: " if ok else "PROBLEM: ") + msg)
        scan = token_exposure_scan()
        print(scan["note"] + (": " + ", ".join(scan["hits"]) if scan["hits"] else ""))
        return

    if args.telegram_chat_id:
        from modules.alerts import discover_chat_ids
        chats = discover_chat_ids()
        if not chats:
            print("No chats yet. Add the bot to your group, send a message in the group, then run this again.")
        for c in chats:
            print(f"chat_id={c['chat_id']}  type={c['type']}  title={c['title']}")
        return

    if args.monitor:
        import time
        from modules.alerts import Monitor
        print(f"{config.PROJECT_NAME} monitor running every {config.MARKET_REFRESH_SECONDS}s. Ctrl+C to stop.")
        mon = Monitor(source="cli").start()
        try:
            while mon.thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            mon.stop()
        return

    if args.send:
        from modules import alerts
        snap = pf.build_snapshot()
        syms = snap["positions"]["yahoo_symbol"].dropna().tolist()
        hist = md.get_history(syms + [config.BENCHMARK_SYMBOL]) if syms else {}
        if args.send == "pdf":
            path = rp.generate_pdf(snap, trigger="cli")
            print("PDF saved:", path)
            print(alerts.send_telegram_document(path, f"{config.PROJECT_NAME} report"))
            return
        text = {"startup": rp.startup_message, "open": rp.market_open_message,
                "close": lambda s: rp.closing_message(s, hist), "four": rp.four_stock_message,
                "test": lambda s: f"✅ {config.PROJECT_NAME} Telegram test – {md.fmt_datetime(md.now_ist())}"}[args.send](snap)
        print(alerts.send_telegram(text, args.send))
        return

    agent = FinPilotAgent()
    if args.question:
        print(agent.answer(" ".join(args.question)))
        return
    print(f"{config.PROJECT_NAME} agent – type a question, 'help', or 'quit'.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        agent.snap = None  # fresh data for every question
        print("\nFinPilot:", agent.answer(q))


if __name__ == "__main__":
    sys.exit(_cli())
