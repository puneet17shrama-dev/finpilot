from __future__ import annotations

import html
import logging
from typing import Any

import requests

import config
from modules import market_data as md

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


def telegram_configured() -> bool:
    return bool(config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID)


def _fmt_money(value: Any) -> str:
    """Missing values are shown as N/A – never as ₹0.00."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if v != v else f"₹{v:,.2f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):,.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def send_telegram(text: str, message_type: str = "message") -> tuple[bool, str]:
    """Send a text message to the configured Telegram chat."""
    if not config.TELEGRAM_ENABLED:
        return False, "Telegram is not configured."
    if not telegram_configured():
        return False, "Telegram report failed. Check Telegram configuration."
    try:
        url = TELEGRAM_API.format(token=config.TELEGRAM_BOT_TOKEN, method="sendMessage")
        payload = {
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }
        r = requests.post(url, data=payload, timeout=20)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if not data.get("ok"):
            from modules.alerts import redact
            return False, redact(data.get("description") or "Telegram report failed. Check Telegram configuration.")
        return True, "Sell analysis report sent to Telegram."
    except Exception as exc:  # never include the URL (it contains the bot token) in logs
        logging.getLogger("finpilot.telegram").warning("Telegram send failed: %s", type(exc).__name__)
        return False, "Telegram report failed. Check Telegram configuration."


def send_sell_analysis_report(
    account,
    stock,
    yahoo_symbol,
    quantity,
    avg_buy_price,
    current_price,
    estimated_sale_value,
    invested_value,
    estimated_pnl,
    return_pct,
    market_status,
    data_status,
    estimated_sell_price=None,
    price_label="Estimated Sell Price",
) -> tuple[bool, str]:
    """Send a sell-analysis summary to Telegram with the exact account and lot context."""
    if not config.TELEGRAM_ENABLED:
        return False, "Telegram is not configured."
    if not telegram_configured():
        return False, "Telegram report failed. Check Telegram configuration."
    if current_price is None:
        return False, "Sell analysis cannot be sent because current market data is unavailable."

    if estimated_sell_price is None or estimated_sell_price == "Current Yahoo LTP":
        sell_price_line = f"Estimated Sell Price: Current Yahoo LTP ({_fmt_money(current_price)})"
    else:
        sell_price_line = f"Estimated Sell Price: {_fmt_money(estimated_sell_price)}"

    report_time = md.fmt_datetime(md.now_ist())
    lines = [
        "<b>FINPILOT — SELL ANALYSIS</b>",
        f"Account: {html.escape(str(account))}",
        f"Stock: {html.escape(str(stock))}",
        f"Yahoo Symbol: {html.escape(str(yahoo_symbol))}",
        "",
        f"Quantity: {quantity}",
        "",
        f"Average Buy Price: {_fmt_money(avg_buy_price)}",
        f"Current Price/LTP: {_fmt_money(current_price)}",
        sell_price_line,
        "",
        f"Invested Value: {_fmt_money(invested_value)}",
        f"Estimated Sale Value: {_fmt_money(estimated_sale_value)}",
        "",
        f"Estimated P&L: {_fmt_money(estimated_pnl)}",
        f"Return: {_fmt_pct(return_pct)}",
        "",
        f"Current Market Status: {html.escape(str(market_status))}",
        f"Data Status: {html.escape(str(data_status))}",
        f"Report Time: {report_time}",
        "",
        "<b>ESTIMATE ONLY</b>",
        "This is a calculation/analysis report.",
        "No order is placed.",
        "FinPilot has no broker connection.",
    ]
    final_text = "\n".join(lines)
    return send_telegram(final_text)
