"""
Alerts, Telegram notifications and the background monitor.

Telegram is used ONLY for notifications / reports / alerts. Nothing received
from or sent to Telegram can place or execute a trade – FinPilot has no
trading capability at all.

Anti-spam rules
---------------
* Every alert has a deterministic `alert_key` (type + symbol + date + level)
  stored with a UNIQUE constraint → the same alert is never raised twice.
* All new alerts found in one cycle are combined into ONE Telegram message.
* Scheduled reports (market-open, closing) are claimed once per trading day
  in `telegram_log`, so multiple dashboards/monitors can't double-send.
* The dashboard refreshes every ~10 s; Telegram is NOT messaged every 10 s.
"""
from __future__ import annotations

import html
import os
import re
import logging
import math
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

import config
from modules import database as db
from modules import market_data as md

log = logging.getLogger("finpilot.alerts")

# ================================================================ Telegram
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
MAX_TG_LEN = 4000


_TOKEN_RE = re.compile(r"bot\d+:[A-Za-z0-9_-]+")


def redact(text) -> str:
    """Remove the bot token from any text (request errors include the full API URL)."""
    out = _TOKEN_RE.sub("bot<redacted>", str(text))
    if config.TELEGRAM_BOT_TOKEN:
        out = out.replace(config.TELEGRAM_BOT_TOKEN, "<redacted>")
    return out


SCAN_SKIP_DIRS = {".venv", "venv", "__pycache__", ".git", "node_modules"}


def token_exposure_scan() -> dict:
    """Look for the bot token anywhere in the project except .env (source, README, logs, PDFs,
    and the SQLite database). Returns file names only – never the token itself."""
    tok = config.TELEGRAM_BOT_TOKEN
    if not tok:
        return {"checked": False, "hits": [], "note": "No token configured"}
    needle, hits = tok.encode(), []
    for root, dirs, files in os.walk(config.BASE_DIR):
        dirs[:] = [d for d in dirs if d not in SCAN_SKIP_DIRS]
        for name in files:
            if name == ".env":
                continue
            path = Path(root) / name
            try:
                if path.stat().st_size > 50_000_000:
                    continue
                if needle in path.read_bytes():
                    hits.append(str(path.relative_to(config.BASE_DIR)))
            except OSError:
                continue
    if hits:
        log.warning("Telegram bot token found outside .env in: %s – revoke it in @BotFather", ", ".join(hits))
    return {"checked": True, "hits": hits,
            "note": "Token found outside .env – revoke it in @BotFather and remove it from these files"
            if hits else "Token appears only in .env"}


def check_telegram_token() -> tuple[bool, str]:
    """Verify the configured token with Telegram's getMe (read-only). Prints the bot name, never the token."""
    if not config.TELEGRAM_BOT_TOKEN:
        return False, "TELEGRAM_BOT_TOKEN is empty in .env"
    try:
        r = requests.get(TELEGRAM_API.format(token=config.TELEGRAM_BOT_TOKEN, method="getMe"), timeout=20)
        data = r.json()
    except Exception as exc:
        return False, f"Could not reach Telegram: {redact(exc)}"
    if not data.get("ok"):
        return False, f"Token rejected by Telegram ({redact(data.get('description', 'unknown error'))}) – " \
                      "generate a new one in @BotFather and update .env"
    bot = data["result"]
    return True, f"Token valid for bot @{bot.get('username')} ({bot.get('first_name')})"


def telegram_configured() -> bool:
    return bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID)


def esc(text) -> str:
    return html.escape(str(text), quote=False)


def send_telegram(text: str, message_type: str = "message") -> tuple[bool, str]:
    """Send an HTML-formatted message to the configured chat. Splits long messages."""
    if not config.TELEGRAM_ENABLED:
        return False, "Telegram not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env)"
    chunks, buf = [], ""
    for line in text.splitlines(keepends=True):
        if len(buf) + len(line) > MAX_TG_LEN:
            chunks.append(buf)
            buf = ""
        buf += line
    if buf:
        chunks.append(buf)
    url = TELEGRAM_API.format(token=config.TELEGRAM_BOT_TOKEN, method="sendMessage")
    for chunk in chunks:
        try:
            r = requests.post(url, data={"chat_id": config.TELEGRAM_CHAT_ID, "text": chunk,
                                         "parse_mode": "HTML", "disable_web_page_preview": "true"},
                              timeout=20)
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            if not data.get("ok"):
                return False, f"Telegram error: {data.get('description') or r.status_code}"
        except Exception as exc:
            return False, f"Telegram request failed: {redact(exc)}"
    return True, f"sent {len(chunks)} message(s)"


def send_telegram_document(path: Path, caption: str = "") -> tuple[bool, str]:
    if not config.TELEGRAM_ENABLED:
        return False, "Telegram not configured"
    url = TELEGRAM_API.format(token=config.TELEGRAM_BOT_TOKEN, method="sendDocument")
    try:
        with open(path, "rb") as fh:
            r = requests.post(url, data={"chat_id": config.TELEGRAM_CHAT_ID, "caption": caption[:1000]},
                              files={"document": (Path(path).name, fh, "application/pdf")}, timeout=60)
        data = r.json()
        return (True, "sent") if data.get("ok") else (False, data.get("description", "error"))
    except Exception as exc:
        return False, redact(exc)


def discover_chat_ids() -> list[dict]:
    """Helper for setup: list chats the bot has seen (add the bot to your group and
    post any message there first). The group invite link itself is NOT a chat id."""
    if not config.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN in .env first")
    r = requests.get(TELEGRAM_API.format(token=config.TELEGRAM_BOT_TOKEN, method="getUpdates"), timeout=20)
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(data.get("description", "Telegram error"))
    chats = {}
    for upd in data.get("result", []):
        msg = upd.get("message") or upd.get("channel_post") or upd.get("my_chat_member") or {}
        chat = msg.get("chat")
        if chat:
            chats[chat["id"]] = {"chat_id": chat["id"], "type": chat.get("type"),
                                 "title": chat.get("title") or chat.get("username")}
    return list(chats.values())


# ================================================================ alert rules (user-defined targets)
def add_target_rule(base_symbol: str, direction: str, price: float, note: str = "") -> None:
    direction = "above" if direction.lower().startswith("a") else "below"
    db.execute("INSERT INTO alert_rules(base_symbol, rule_type, direction, threshold, active, note, created_at) "
               "VALUES(?, 'target', ?, ?, 1, ?, ?)", (base_symbol.upper(), direction, float(price), note, db.now_iso()))


def list_rules() -> pd.DataFrame:
    return pd.DataFrame(db.query("SELECT * FROM alert_rules ORDER BY rule_id DESC"))


def set_rule_active(rule_id: int, active: bool) -> None:
    db.execute("UPDATE alert_rules SET active=? WHERE rule_id=?", (1 if active else 0, int(rule_id)))


def delete_rule(rule_id: int) -> None:
    db.execute("DELETE FROM alert_rules WHERE rule_id=?", (int(rule_id),))


# ================================================================ evaluation
def _record(key: str, atype: str, symbol: str, severity: str, message: str) -> dict | None:
    """Insert the alert if its key is new. Returns the alert dict only when new."""
    with db.get_conn() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO alerts(alert_key, alert_type, base_symbol, severity, message, created_at) "
            "VALUES(?,?,?,?,?,?)", (key, atype, symbol, severity, message, db.now_iso()))
        if cur.rowcount == 1:
            return {"alert_key": key, "alert_type": atype, "base_symbol": symbol,
                    "severity": severity, "message": message}
    return None


def _level(value: float, step: float) -> int:
    """Bucket a percentage into integer steps toward zero: 5.2→1, -11→-2 (step 5)."""
    if step <= 0:
        return 0
    return int(math.copysign(math.floor(abs(value) / step), value))


def evaluate_alerts(snapshot: dict, history: dict | None = None) -> list[dict]:
    """Check all alert conditions against the latest snapshot. Returns only NEW alerts."""
    new: list[dict] = []
    positions: pd.DataFrame = snapshot["positions"]
    quotes = snapshot["quotes"]
    today = snapshot["generated_at"].strftime("%Y-%m-%d")
    live_market = snapshot["market"]["is_open"]
    history = history or {}
    cur = config.CURRENCY_SYMBOL

    if not positions.empty:
        for r in positions.itertuples():
            sym = r.yahoo_symbol
            q = quotes.get(sym) if sym else None
            is_live = bool(q and q.get("status") == "Live")
            price = q.get("price") if q else None

            # 1) Price movement vs previous close (only on live data)
            if is_live and pd.notna(r.day_change_pct):
                lvl = _level(r.day_change_pct, config.ALERT_DAY_MOVE_PCT)
                if lvl != 0:
                    a = _record(f"daymove:{r.base_symbol}:{today}:{lvl}", "Price movement", r.base_symbol,
                                "high" if abs(lvl) >= 2 else "medium",
                                f"{r.base_symbol} is {r.day_change_pct:+.2f}% today at {cur}{price:,.2f} "
                                f"(prev close {cur}{r.previous_close:,.2f}).")
                    if a:
                        new.append(a)

            # 2) P&L movement: position return crosses a ±ALERT_PNL_MOVE_PCT step
            if is_live and pd.notna(r.return_pct):
                lvl = _level(r.return_pct, config.ALERT_PNL_MOVE_PCT)
                meta_key = f"pnl_level:{r.base_symbol}"
                prev = db.get_meta(meta_key)
                if prev is None:
                    db.set_meta(meta_key, str(lvl))       # first observation = baseline, no alert
                elif int(prev) != lvl:
                    db.set_meta(meta_key, str(lvl))
                    a = _record(f"pnl:{r.base_symbol}:{today}:{lvl}", "P&L movement", r.base_symbol, "medium",
                                f"{r.base_symbol} position return moved to {r.return_pct:+.2f}% "
                                f"(P&L {cur}{r.pnl:,.0f}).")
                    if a:
                        new.append(a)

            # 3) Unusual volume vs 20-day average
            h = history.get(sym) if sym else None
            if is_live and h is not None and len(h) > 21 and q.get("volume"):
                prior = h[h.index.date < q["data_timestamp"].date()]["Volume"].iloc[-20:]
                avg = float(prior.mean()) if len(prior) >= 20 else None
                if avg and q["volume"] / avg >= config.ALERT_VOLUME_RATIO:
                    a = _record(f"volume:{r.base_symbol}:{today}", "Unusual volume", r.base_symbol, "medium",
                                f"{r.base_symbol} volume {q['volume']:,.0f} is {q['volume'] / avg:.1f}× "
                                f"its 20-day average ({avg:,.0f}).")
                    if a:
                        new.append(a)

        # 4) Target price rules
        rules = db.query("SELECT * FROM alert_rules WHERE active=1 AND rule_type='target'")
        for rule in rules:
            hit = positions[positions["base_symbol"] == rule["base_symbol"]]
            if hit.empty:
                continue
            row = hit.iloc[0]
            q = quotes.get(row["yahoo_symbol"]) if row["yahoo_symbol"] else None
            if not q or q.get("price") is None or q.get("status") not in {"Live", "Market closed", "Closed", "Stored close"}:
                continue
            p = q["price"]
            reached = p >= rule["threshold"] if rule["direction"] == "above" else p <= rule["threshold"]
            if reached:
                a = _record(f"target:{rule['rule_id']}:{today}", "Target reached", rule["base_symbol"], "high",
                            f"{rule['base_symbol']} at {cur}{p:,.2f} is {rule['direction']} your target "
                            f"{cur}{rule['threshold']:,.2f}." + (f" Note: {rule['note']}" if rule["note"] else ""))
                if a:
                    new.append(a)

    # 5) Portfolio movement (today's %)
    s = snapshot["summary"]
    if live_market and s.get("todays_pct") is not None:
        lvl = _level(s["todays_pct"], config.ALERT_PORTFOLIO_MOVE_PCT)
        if lvl != 0:
            a = _record(f"portfolio:{today}:{lvl}", "Portfolio movement", "PORTFOLIO", "high",
                        f"Portfolio is {s['todays_pct']:+.2f}% today (today's P&L {cur}{s['todays_pnl']:,.0f}).")
            if a:
                new.append(a)

    # 6) Important news (published in the last 24 h, each article alerted once)
    if config.ALERT_IMPORTANT_NEWS and not positions.empty:
        syms = positions["yahoo_symbol"].dropna().tolist()
        if syms:
            ph = ",".join("?" * len(syms))
            since = (snapshot["generated_at"] - timedelta(hours=24)).isoformat()
            rows = db.query(f"SELECT * FROM news WHERE category='Important Event' AND yahoo_symbol IN ({ph}) "
                            f"AND published_at >= ?", syms + [since])
            sym_to_base = positions.set_index("yahoo_symbol")["base_symbol"].to_dict()
            for n in rows:
                base = sym_to_base.get(n["yahoo_symbol"], n["yahoo_symbol"])
                a = _record(f"news:{n['news_id']}", "Important news", base, "medium",
                            f"{base}: {n['headline']} ({n['publisher']})")
                if a:
                    a["url"] = n["url"]
                    new.append(a)
    return new


def format_alerts_message(alerts: list[dict]) -> str:
    now = md.now_ist()
    icons = {"Price movement": "📈", "P&L movement": "💰", "Unusual volume": "📊", "Target reached": "🎯",
             "Portfolio movement": "🧭", "Important news": "📰"}
    lines = [f"🔔 <b>{esc(config.PROJECT_NAME.upper())} ALERTS</b>", f"{now.strftime('%d %b %Y, %I:%M %p IST')}", ""]
    shown = alerts[: max(1, config.TELEGRAM_MAX_ALERTS_PER_CYCLE)]
    for a in shown:
        line = f"{icons.get(a['alert_type'], '•')} <b>{esc(a['alert_type'])}</b> – {esc(a['message'])}"
        if a.get("url"):
            line += f'\n<a href="{esc(a["url"])}">Read</a>'
        lines.append(line)
    if len(alerts) > len(shown):
        lines.append(f"…and {len(alerts) - len(shown)} more (see the dashboard Alerts page).")
    lines += ["", "<i>Monitoring only – FinPilot never places trades.</i>"]
    return "\n".join(lines)


def dispatch_alerts(alerts: list[dict]) -> tuple[bool, str]:
    if not alerts:
        return True, "no alerts"
    ok, detail = send_telegram(format_alerts_message(alerts), "alerts")
    if ok:
        keys = [a["alert_key"] for a in alerts]
        with db.get_conn() as conn:
            conn.executemany("UPDATE alerts SET telegram_sent=1, telegram_sent_at=? WHERE alert_key=?",
                             [(db.now_iso(), k) for k in keys])
    return ok, detail


def recent_alerts(limit: int = 100) -> pd.DataFrame:
    return pd.DataFrame(db.query("SELECT * FROM alerts ORDER BY created_at DESC LIMIT ?", (limit,)))


# ================================================================ background monitor
def _hhmm(text: str):
    h, m = text.split(":")
    return int(h), int(m)


def _in_window(now: datetime, start: str, end: str) -> bool:
    s, e = now.replace(hour=_hhmm(start)[0], minute=_hhmm(start)[1], second=0, microsecond=0), \
           now.replace(hour=_hhmm(end)[0], minute=_hhmm(end)[1], second=0, microsecond=0)
    return s <= now <= e


class Monitor:
    """Runs every MARKET_REFRESH_SECONDS in a background thread (never inside the
    Streamlit script body). Refreshes the shared quote cache, evaluates alerts,
    sends scheduled Telegram reports and slowly prefetches fundamentals/news."""

    def __init__(self, source: str = "dashboard"):
        self.source = source
        self.started_at = md.now_ist()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.last_tick: datetime | None = None
        self.last_error: str | None = None
        self.events: list[str] = []
        self._startup_done = False
        self.token_scan: dict | None = None

    # -- lifecycle
    def start(self) -> "Monitor":
        if self.thread and self.thread.is_alive():
            return self
        self.thread = threading.Thread(target=self._run, name="finpilot-monitor", daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()

    def _run(self):
        self.tick()
        while not self.stop_event.wait(config.MARKET_REFRESH_SECONDS):
            self.tick()

    def _event(self, text: str):
        self.events = ([f"{md.fmt_time(md.now_ist())} {text}"] + self.events)[:50]
        log.info(text)

    # -- one cycle
    def tick(self):
        from modules import fundamentals as fnd
        from modules import news as nws
        from modules import portfolio as pf
        from modules import reports as rp
        try:
            snap = pf.build_snapshot()
            self.last_tick = snap["generated_at"]
            syms = snap["positions"]["yahoo_symbol"].dropna().unique().tolist() if not snap["positions"].empty else []
            history = md.get_history(syms + [config.BENCHMARK_SYMBOL]) if syms else {}
            now = snap["generated_at"]
            date = now.strftime("%Y-%m-%d")

            # Startup message (once per process)
            if not self._startup_done:
                self._startup_done = True
                self.token_scan = token_exposure_scan()      # logs a warning if the token leaked
                md.holiday_calendar_status(now)               # logs a warning if this year has no holidays
                if config.TELEGRAM_SEND_STARTUP and config.TELEGRAM_ENABLED:
                    ok, d = send_telegram(rp.startup_message(snap), "startup")
                    self._event(f"Startup message: {d}")
                    db.execute("INSERT OR REPLACE INTO telegram_log(log_key, message_type, status, detail, sent_at) "
                               "VALUES(?,?,?,?,?)", (f"startup:{self.source}:{self.started_at.isoformat()}",
                                                     "startup", "sent" if ok else "failed", d, db.now_iso()))

            trading_day = snap["market"]["trading_day"]
            # Market-open report (once per trading day, after opening prices exist)
            if trading_day and config.TELEGRAM_ENABLED and \
                    _in_window(now, config.TELEGRAM_OPEN_REPORT_TIME, config.TELEGRAM_OPEN_REPORT_UNTIL) and \
                    snap["summary"]["priced_holdings"] > 0 and db.claim_once(f"open:{date}", "market_open"):
                ok, d = send_telegram(rp.market_open_message(snap), "market_open")
                db.update_log(f"open:{date}", "sent" if ok else "failed", d)
                self._event(f"Market-open report: {d}")

            # Closing report (+ four-stock report, optional PDF) once per trading day
            if trading_day and config.TELEGRAM_ENABLED and \
                    _in_window(now, config.TELEGRAM_CLOSE_REPORT_TIME, config.TELEGRAM_CLOSE_REPORT_UNTIL) and \
                    snap["summary"]["priced_holdings"] > 0 and db.claim_once(f"close:{date}", "market_close"):
                ok, d = send_telegram(rp.closing_message(snap, history), "market_close")
                db.update_log(f"close:{date}", "sent" if ok else "failed", d)
                self._event(f"Closing report: {d}")
                if config.REPORT_STOCKS:
                    ok2, d2 = send_telegram(rp.four_stock_message(snap), "four_stock")
                    self._event(f"Four-stock report: {d2}")
                if config.SCHEDULED_PDF_AT_CLOSE:
                    path = rp.generate_pdf(snap, trigger="scheduled-close")
                    send_telegram_document(path, f"{config.PROJECT_NAME} daily report {date}")

            # Alerts
            new = evaluate_alerts(snap, history)
            if new:
                ok, d = dispatch_alerts(new)
                self._event(f"{len(new)} new alert(s); Telegram: {d}")

            # Gentle background prefetch – one symbol per cycle to respect Yahoo limits
            missing = fnd.missing_or_stale(syms)
            if missing:
                fnd.get_fundamentals(missing[0])
            stale_news = nws.stale_symbols(syms)
            if stale_news:
                nws.refresh_news(stale_news[0])
            self.last_error = None
        except Exception as exc:  # the monitor must never die
            self.last_error = redact(exc)
            log.exception("Monitor tick failed")

    def status(self) -> dict:
        return {"running": bool(self.thread and self.thread.is_alive()), "source": self.source,
                "started_at": self.started_at, "last_tick": self.last_tick, "last_error": self.last_error,
                "events": list(self.events)}
