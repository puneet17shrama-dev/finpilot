# FinPilot – Personal Portfolio Intelligence

Local stock-portfolio **monitoring and analysis** agent. It is **not** a trading bot:
it never places, modifies or executes orders and has no broker connection.
Market data comes from **Yahoo Finance (yfinance) only**. When Yahoo fails, values show
as *Stale* / *Unavailable* – nothing is invented, no other provider is used.

## Run (Windows)

```bat
cd "C:\Users\punee\OneDrive\Desktop\finance agent\FinPilot"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. Live sections refresh every ~10 s (`MARKET_REFRESH_SECONDS`).

Other commands:

| Command | What it does |
|---|---|
| `python agent.py` | Ask FinPilot questions in the terminal |
| `python agent.py "what is my COFFEEDAY position?"` | One question |
| `python agent.py --monitor` | Headless monitor: alerts + scheduled Telegram reports, no dashboard |
| `python agent.py --telegram-chat-id` | Find your Telegram group's chat id |
| `python agent.py --send test\|startup\|open\|close\|four\|pdf` | Send a report to Telegram now |

## Tests

```bat
python -m tests.test_portfolio
```
Offline checks (no Yahoo, no Telegram): totals, account reconciliation, missing prices,
today's P&L, duplicates, Excel refresh, CSV guard, PDF, news, token redaction.

## Portfolio file

`data/holdings.xlsx` (config `HOLDINGS_FILE`) is the ONE authoritative source. It is re-parsed
whenever its modification time/size changes. A CSV is never picked up unless you set
`HOLDINGS_FILE` to it **and** `ALLOW_CSV_HOLDINGS=true`.

All screens, Telegram messages, PDFs, alerts, scanner, risk and sell analysis use the same
snapshot from `portfolio.build_snapshot()`. Holdings without a Yahoo price are marked
UNAVAILABLE and excluded from current value/P&L (never valued at ₹0); the valuation is
then labelled PARTIAL with the coverage and the missing symbols.

Note: the `LTP` / `Current Value` columns in a broker export are the prices at export time.
FinPilot ignores them and uses live Yahoo prices, so its value differs from
Σ Qty × LTP by the price movement since the export (see Data & Settings → Data validation).
Broker exports with several accounts, repeated header rows and summary rows are supported.
Needed columns (any common name): symbol (`Instrument`/`Symbol`), quantity (`Qty`), average price
(`Avg Price`). Optional: account (`UserId`), `Yahoo Symbol` to force a mapping.
Broker LTP/Current Value columns are ignored – live prices come from Yahoo.

## Telegram

The invite link is not an API token. Create a bot with **@BotFather**, add it to your group,
post a message there, run `python agent.py --telegram-chat-id`, then put
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`. Messages are sent only on startup,
the scheduled market-open/closing reports, deduplicated alerts, or when you request one –
never every 10 seconds.

## Layout

```
app.py            Streamlit dashboard           agent.py   Q&A agent + CLI
config.py         all settings (.env)           data/      holdings.xlsx, portfolio.db (auto-created)
modules/          portfolio, market_data, technical, fundamentals, news, risk, scanner,
                  sell_analysis, alerts (+Telegram, monitor), reports (Telegram text + PDF),
                  database (SQLite schema & safe migrations)
reports/          generated PDFs
```
