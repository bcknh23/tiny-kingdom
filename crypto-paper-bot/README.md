# Crypto Paper Trading Bot

A crypto trading bot for research and **paper trading only**. It backtests rule-based strategies on real historical candles, then paper-trades them live with a fictional **$10,000** balance. A Streamlit dashboard shows the results.

> **No real money is involved.** The bot only reads public market data. It has no exchange credentials, can't place real orders and can't withdraw anything. A test (`tests/test_safety.py`) fails if order, credential or withdrawal code is ever added.

Backtest and paper results are hypothetical. They are meant to measure a strategy honestly, not to promise profits. Most simple strategies lose to buy & hold once fees are included, and the bot is built to show you that clearly.

## What it does

| Area | Details |
|---|---|
| Market data | Public, read-only OHLCV from **Coinbase Exchange** (deep history, paginated) or **Kraken** (last 720 candles). A labelled **synthetic** source works offline for demos and tests. Retries with backoff on network errors, 429 and 5xx. |
| Data validation | Every candle set is checked: UTC timestamps, sorted and de-duplicated, aligned to the timeframe, finite positive prices, consistent high/low, non-negative volume. Bad rows are dropped and logged. If more than 5% are bad, the data is rejected. Gaps are logged but never filled with invented prices. Candles that haven't closed yet are discarded. |
| Strategies | Moving-average crossover (SMA/EMA), RSI mean reversion (Wilder RSI), and a trend filter that can wrap either one or run alone. Strategies are long or flat only (spot, no leverage, no shorting). |
| Backtesting | Event-driven. A signal from candle *t*'s close fills at candle *t+1*'s **open**, plus slippage and fees. The last candle's signal stays pending; it is never filled at a price that doesn't exist yet. |
| Benchmark | Buy & hold through the same broker with the same fees and slippage, entering at the earliest fill the strategy could get. |
| Metrics | Total return, CAGR, max drawdown, Sharpe and Sortino (annualised, risk-free rate 0), volatility, win rate, profit factor, average win/loss, fees paid, time in market, and excess return over buy & hold. |
| Paper trading | Polls new closed candles and runs **exactly the same broker code** as the backtester. Each candle is committed to SQLite in one transaction, so a crash never half-updates an account. After downtime it catches up on missed candles in order. A new account starts trading from the latest closed candle, with no back-filled trades. |
| Storage | SQLite (`data/paperbot.sqlite`): candles, accounts, fills, round-trip trades and the equity curve. |
| Logging | Console plus a rotating file at `logs/paperbot.log`. |

### Execution model (defaults)

- **Fee:** 0.40% of notional per fill, a typical entry-tier taker fee on major US exchanges.
- **Slippage:** 5 bps against you on every fill (buys fill higher, sells fill lower).
- **Sizing:** 100% of cash per entry (adjustable), with the fee reserved so cash never goes negative.
- **Valuation:** equity is marked at each candle close.

You can change all of these in the dashboard or with CLI flags. Lower fees make results look better, so set them to what you would actually pay.

### How look-ahead bias is prevented and tested

1. Indicators are causal and leave the warm-up period as NaN. Tests check that every indicator and strategy gives identical values on a truncated history and on the full history.
2. Orders fill at the **next** candle's open, never the close that triggered them.
3. A test multiplies all prices after candle 200 by 3 and checks that the equity curve and fills before candle 200 are unchanged.
4. A test runs the live paper trader candle by candle (with restarts and a day of downtime) and checks that its fills, trades and balances match the backtester on the same data.
5. Candles still in progress are dropped before any signal is computed.

## Setup on macOS

You need Python 3.10 or newer. Check with `python3 --version`. If you don't have it, install it from [python.org](https://www.python.org/downloads/macos/) or with Homebrew (`brew install python`).

```bash
cd crypto-paper-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest            # should print "... passed"
```

Run `source .venv/bin/activate` again in each new Terminal window before using the bot.

## Usage

### Dashboard

```bash
streamlit run dashboard/app.py
```

Your browser opens at http://localhost:8501.

- **Backtest tab:** pick a market, strategy, fees and slippage in the sidebar. You'll see the candlestick chart with indicators and buy/sell fills, the equity curve against buy & hold, drawdown, trade history and every metric. You can also save a backtest to SQLite.
- **Paper trading tab:** create a paper account from the sidebar settings, process new candles on demand, and view any stored account.

### Command line

```bash
# Backtest the default strategy (SMA 20/50 crossover + 200 SMA trend filter) on 180 days of hourly BTC
python -m paperbot backtest --symbol BTC-USD --timeframe 1h --days 180

# Other presets: ma, rsi, trend, ma_trend, rsi_trend. Or pass any config as JSON:
python -m paperbot backtest --strategy rsi_trend --timeframe 1d --days 1500
python -m paperbot backtest --strategy-json '{"type":"ma_crossover","fast":10,"slow":30,"ma_type":"ema"}' --save my-test

# Create a paper account with $10,000 of fictional cash, then run it
python -m paperbot paper-init btc-hourly --symbol BTC-USD --timeframe 1h --strategy ma_trend
python -m paperbot paper-run btc-hourly          # runs until Ctrl+C, waking after each candle closes
python -m paperbot paper-run btc-hourly --once   # process new candles once and exit (e.g. from cron)

# Inspect stored accounts
python -m paperbot status
python -m paperbot status btc-hourly
```

The paper trader only processes candles while it's running. To keep it going, leave a Terminal window open with `paper-run`, or schedule `paper-run NAME --once` with `cron` or `launchd`. Missed candles are caught up on the next run.

Add `--source synthetic` to any command to try it without network access. Synthetic data is a random walk, not real prices.

Settings via environment variables: `PAPERBOT_DB` (database path), `PAPERBOT_LOG` (log file path), and `PAPERBOT_SOURCE` (the dashboard's default data source).

## Project layout

```
paperbot/
  config.py       execution settings (fees, slippage, sizing), timeframes, paths
  data.py         read-only public data fetchers + synthetic source
  validation.py   OHLCV validation
  indicators.py   SMA, EMA, Wilder RSI
  strategies.py   MA crossover, RSI reversion, trend filter, build_strategy()
  broker.py       simulated fills, fees, slippage, P&L (shared by backtest and paper)
  backtest.py     next-candle-open backtester + buy & hold benchmark
  metrics.py      performance metrics
  storage.py      SQLite persistence
  paper.py        paper trading loop
  cli.py          command line
dashboard/app.py  Streamlit dashboard
tests/            pytest suite (runs offline)
```

## Limitations

- Fills use the next candle's open plus fixed slippage. Real slippage grows with order size and thin markets, and there's no order-book simulation.
- Long or flat only. No shorting, leverage, stop-losses or partial position sizing beyond the allocation fraction.
- Kraken only returns the latest 720 candles, so use Coinbase for long backtests.
- Coinbase Exchange may be unavailable in some regions. If a source fails, try the other one.
- A strategy tuned on the same history it is tested on will look better than it really is. Use a fresh period, or paper trade it, before trusting a result.
