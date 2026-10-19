"""Streamlit dashboard for backtests and paper accounts.

Run from the project folder:  streamlit run dashboard/app.py

PAPER TRADING ONLY: nothing here can place a real order.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperbot.backtest import BacktestResult, run_backtest  # noqa: E402
from paperbot.config import DEFAULT_DB_PATH, STARTING_CASH, TIMEFRAME_SECONDS, ExecutionConfig  # noqa: E402
from paperbot.data import DataSourceError, fetch_history  # noqa: E402
from paperbot.logging_setup import setup_logging  # noqa: E402
from paperbot.metrics import drawdown_series, format_metric, metrics_table  # noqa: E402
from paperbot.paper import PaperTrader, account_summary  # noqa: E402
from paperbot.storage import AccountExistsError, Database  # noqa: E402
from paperbot.strategies import MACrossover, RSIReversion, Strategy, TrendFilter, build_strategy  # noqa: E402
from paperbot.validation import DataValidationError  # noqa: E402

# Colors by role (validated categorical slots 1-2; up/down candles use aqua/red).
STRATEGY = "#2a78d6"
BENCHMARK = "#eb6834"
UP = "#1baf7a"
DOWN = "#e34948"
INDICATOR_COLORS = ["#6250d6", "#eda100", "#e87ba4"]
MUTED = "#8a8984"

st.set_page_config(page_title="Crypto Paper Bot", page_icon="📈", layout="wide")
setup_logging("INFO")


@st.cache_resource
def get_db() -> Database:
    return Database(DEFAULT_DB_PATH)


@st.cache_data(ttl=300, show_spinner="Fetching candles...")
def load_candles(source: str, symbol: str, timeframe: str, days: float) -> pd.DataFrame:
    return fetch_history(source, symbol, timeframe, days)


# ---------------------------------------------------------------- sidebar
def sidebar() -> tuple[dict, Strategy, ExecutionConfig]:
    sb = st.sidebar
    sb.header("Market")
    sources = ["coinbase", "kraken", "synthetic"]
    default_source = os.environ.get("PAPERBOT_SOURCE", "coinbase")
    source = sb.selectbox("Data source", sources, index=sources.index(default_source) if default_source in sources else 0,
                          help="Coinbase has deep history. Kraken returns only the last 720 candles. "
                               "Synthetic is a random walk for offline testing, not real data.")
    symbol = sb.text_input("Symbol", "BTC-USD").strip().upper()
    timeframe = sb.selectbox("Timeframe", list(TIMEFRAME_SECONDS), index=3)
    days = sb.number_input("History (days)", 1, 3000, 365 if timeframe in ("1h", "1d") else 30)

    sb.header("Strategy")
    kind = sb.selectbox("Signal", ["MA crossover", "RSI mean reversion", "Trend only"])
    base: Strategy | None
    if kind == "MA crossover":
        c1, c2 = sb.columns(2)
        fast = c1.number_input("Fast MA", 2, 400, 20)
        slow = c2.number_input("Slow MA", 3, 500, 50)
        ma_type = sb.radio("MA type", ["sma", "ema"], horizontal=True)
        if fast >= slow:
            sb.error("Fast MA must be shorter than slow MA.")
            st.stop()
        base = MACrossover(int(fast), int(slow), ma_type)
    elif kind == "RSI mean reversion":
        period = sb.number_input("RSI period", 2, 100, 14)
        c1, c2 = sb.columns(2)
        lo = c1.number_input("Buy below", 1, 98, 30)
        hi = c2.number_input("Sell above", 2, 99, 70)
        if lo >= hi:
            sb.error("Buy threshold must be below sell threshold.")
            st.stop()
        base = RSIReversion(int(period), float(lo), float(hi))
    else:
        base = None
    use_filter = kind == "Trend only" or sb.checkbox("Trend filter (only long above long MA)", True)
    strategy: Strategy = base  # type: ignore[assignment]
    if use_filter:
        period = sb.number_input("Trend MA period", 10, 500, 200)
        strategy = TrendFilter(int(period), "sma", base)

    sb.header("Costs & sizing")
    fee = sb.number_input("Fee per fill (%)", 0.0, 5.0, 0.40, 0.05) / 100
    slip = sb.number_input("Slippage per fill (bps)", 0.0, 200.0, 5.0, 1.0)
    alloc = sb.slider("Cash used per entry", 0.1, 1.0, 1.0, 0.05)
    execution = ExecutionConfig(starting_cash=STARTING_CASH, fee_rate=fee, slippage_bps=slip, allocation=alloc)
    sb.caption("Fictional $10,000 balance. No exchange account, keys or real orders.")
    market = {"source": source, "symbol": symbol, "timeframe": timeframe, "days": float(days)}
    return market, strategy, execution


# ---------------------------------------------------------------- charts
def price_chart(candles: pd.DataFrame, indicators: pd.DataFrame, fills: pd.DataFrame,
                max_bars: int = 1500) -> go.Figure:
    candles = candles.iloc[-max_bars:]
    indicators = indicators.reindex(candles.index)
    rsi_cols = [c for c in indicators.columns if c.startswith("rsi")]
    overlay_cols = [c for c in indicators.columns if c not in rsi_cols]
    rows = 2 if rsi_cols else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.75, 0.25] if rows == 2 else [1.0])
    fig.add_trace(go.Candlestick(
        x=candles.index, open=candles["open"], high=candles["high"], low=candles["low"],
        close=candles["close"], name="Price",
        increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN),
    ), row=1, col=1)
    for color, col in zip(INDICATOR_COLORS, overlay_cols):
        fig.add_trace(go.Scatter(x=indicators.index, y=indicators[col], name=col.upper(),
                                 mode="lines", line=dict(color=color, width=2)), row=1, col=1)
    if len(fills):
        fills = fills[fills["ts"] >= candles.index[0]]
        for side, symbol, color, label in (("buy", "triangle-up", UP, "Buy fill"),
                                           ("sell", "triangle-down", DOWN, "Sell fill")):
            f = fills[fills["side"] == side]
            fig.add_trace(go.Scatter(
                x=f["ts"], y=f["price"], mode="markers", name=label,
                marker=dict(symbol=symbol, size=12, color=color, line=dict(width=2, color="white")),
                hovertemplate=f"{label}<br>%{{x}}<br>$%{{y:,.2f}}<extra></extra>",
            ), row=1, col=1)
    for col in rsi_cols:
        fig.add_trace(go.Scatter(x=indicators.index, y=indicators[col], name=col.upper(),
                                 mode="lines", line=dict(color=INDICATOR_COLORS[0], width=2)), row=2, col=1)
        for level in (30, 70):
            fig.add_hline(y=level, line=dict(color=MUTED, width=1, dash="dot"), row=2, col=1)
        fig.update_yaxes(range=[0, 100], title_text="RSI", row=2, col=1)
    fig.update_layout(height=560 if rows == 2 else 480, xaxis_rangeslider_visible=False,
                      hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.06, x=0))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    return fig


def equity_chart(equity: pd.Series, benchmark: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Strategy", mode="lines",
                             line=dict(color=STRATEGY, width=2),
                             hovertemplate="Strategy $%{y:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark, name="Buy & hold", mode="lines",
                             line=dict(color=BENCHMARK, width=2),
                             hovertemplate="Buy & hold $%{y:,.2f}<extra></extra>"))
    fig.add_hline(y=float(equity.iloc[0]), line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(height=340, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="Equity ($)", legend=dict(orientation="h", y=1.1, x=0))
    return fig


def drawdown_chart(equity: pd.Series, benchmark: pd.Series) -> go.Figure:
    fig = go.Figure()
    for series, name, color in ((equity, "Strategy", STRATEGY), (benchmark, "Buy & hold", BENCHMARK)):
        dd = drawdown_series(series) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=name, mode="lines", line=dict(color=color, width=2),
                                 hovertemplate=f"{name} %{{y:.1f}}%<extra></extra>"))
    fig.update_layout(height=240, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      yaxis_title="Drawdown (%)", legend=dict(orientation="h", y=1.15, x=0))
    return fig


# ---------------------------------------------------------------- views
def kpi_row(metrics: dict) -> None:
    items = [
        ("Total return", "total_return", "pct"),
        ("Buy & hold", "benchmark_return", "pct"),
        ("Max drawdown", "max_drawdown", "pct"),
        ("Win rate", "win_rate", "pct"),
        ("Closed trades", "num_trades", "int"),
        ("Sharpe", "sharpe", "num"),
    ]
    for col, (label, key, kind) in zip(st.columns(len(items)), items):
        col.metric(label, format_metric(metrics.get(key), kind))


def trades_table(trades: pd.DataFrame) -> None:
    if trades.empty:
        st.info("No closed trades yet.")
        return
    t = trades.copy()
    t["return_pct"] = t["return_pct"] * 100
    st.dataframe(t.iloc[::-1], hide_index=True, width="stretch", column_config={
        "entry_ts": st.column_config.DatetimeColumn("Entry", format="YYYY-MM-DD HH:mm"),
        "exit_ts": st.column_config.DatetimeColumn("Exit", format="YYYY-MM-DD HH:mm"),
        "qty": st.column_config.NumberColumn("Qty", format="%.6f"),
        "entry_price": st.column_config.NumberColumn("Entry price", format="$%.2f"),
        "exit_price": st.column_config.NumberColumn("Exit price", format="$%.2f"),
        "fees": st.column_config.NumberColumn("Fees", format="$%.2f"),
        "pnl": st.column_config.NumberColumn("P&L (net)", format="$%.2f"),
        "return_pct": st.column_config.NumberColumn("Return", format="%.2f%%"),
    })


def fills_table(fills: pd.DataFrame) -> None:
    if fills.empty:
        st.info("No fills yet.")
        return
    st.dataframe(fills.iloc[::-1], hide_index=True, width="stretch", column_config={
        "ts": st.column_config.DatetimeColumn("Time (candle open)", format="YYYY-MM-DD HH:mm"),
        "price": st.column_config.NumberColumn("Fill price", format="$%.2f"),
        "reference_price": st.column_config.NumberColumn("Open (pre-slippage)", format="$%.2f"),
        "fee": st.column_config.NumberColumn("Fee", format="$%.2f"),
        "cash_after": st.column_config.NumberColumn("Cash after", format="$%.2f"),
    })


def all_metrics(metrics: dict) -> None:
    st.dataframe(pd.DataFrame(metrics_table(metrics), columns=["Metric", "Value"]),
                 hide_index=True, width="stretch")


def backtest_view(market: dict, strategy: Strategy, execution: ExecutionConfig) -> None:
    try:
        candles = load_candles(market["source"], market["symbol"], market["timeframe"], market["days"])
        result: BacktestResult = run_backtest(candles, strategy, execution, timeframe=market["timeframe"])
    except (DataSourceError, DataValidationError, ValueError) as exc:
        st.error(f"Could not run backtest: {exc}")
        return

    if market["source"] == "synthetic":
        st.warning("Synthetic data: a random walk for testing the software, not real market prices.")
    st.caption(f"{strategy.label()} · {len(result.candles):,} closed candles "
               f"{result.candles.index[0]:%Y-%m-%d %H:%M} → {result.candles.index[-1]:%Y-%m-%d %H:%M} UTC · "
               f"fee {execution.fee_rate:.2%}, slippage {execution.slippage_bps:g} bps per fill")
    kpi_row(result.metrics)

    st.subheader("Price, indicators and fills")
    if len(result.candles) > 1500:
        st.caption("Chart shows the latest 1,500 candles; metrics cover the full period.")
    st.plotly_chart(price_chart(result.candles, result.indicators, result.fills_frame()),
                    width="stretch", key="bt_price")
    if result.pending_target is not None:
        side = "BUY" if result.pending_target else "SELL"
        st.info(f"Signal on the last closed candle: {side} at the next candle's open (not yet filled).")

    st.subheader("Equity vs buy & hold")
    st.plotly_chart(equity_chart(result.equity["equity"], result.benchmark), width="stretch", key="bt_equity")
    st.plotly_chart(drawdown_chart(result.equity["equity"], result.benchmark), width="stretch", key="bt_dd")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Trade history")
        trades_table(result.trades_frame())
    with c2:
        st.subheader("All metrics")
        all_metrics(result.metrics)
    with st.expander("Fills"):
        fills_table(result.fills_frame())

    with st.form("save_bt"):
        name = st.text_input("Save this backtest to SQLite as", f"bt-{market['symbol']}-{market['timeframe']}")
        if st.form_submit_button("Save backtest"):
            get_db().save_backtest(name, result, market["source"], market["symbol"])
            st.success(f"Saved {name!r}.")

    st.caption("Results are hypothetical, computed on past data with the parameters above. Parameters tuned "
               "on the same history they are tested on will overstate future performance.")


def paper_view(market: dict, strategy: Strategy, execution: ExecutionConfig) -> None:
    db = get_db()
    accounts = db.list_accounts()
    with st.expander("Create a paper account from the sidebar settings", expanded=accounts.empty):
        name = st.text_input("Account name", "paper-1")
        st.caption(f"{market['source']} · {market['symbol']} · {market['timeframe']} · {strategy.label()} · "
                   f"${execution.starting_cash:,.0f} fictional cash")
        if st.button("Create account"):
            try:
                PaperTrader.create_account(db, name, source=market["source"], symbol=market["symbol"],
                                           timeframe=market["timeframe"], strategy=strategy,
                                           execution=execution)
                st.success(f"Created {name!r}. Trading starts from the latest closed candle.")
                accounts = db.list_accounts()
            except (AccountExistsError, ValueError) as exc:
                st.error(str(exc))

    if accounts.empty:
        st.info("No accounts yet.")
        return
    labels = {f"{r['name']} ({r['kind']})": r["name"] for _, r in accounts.iterrows()}
    choice = labels[st.selectbox("Account", list(labels))]
    acct = db.get_account(choice)
    if acct is None:
        st.error("Account not found.")
        return

    c1, c2 = st.columns([3, 1])
    c1.caption(f"{acct['kind']} · {acct['source']} · {acct['symbol']} · {acct['timeframe']} · "
               f"strategy {acct['strategy']}")
    if acct["kind"] == "paper" and c2.button("Process new candles now"):
        try:
            report = PaperTrader(db, choice).step()
            st.success(f"Processed {report.processed} new candle(s).")
        except Exception as exc:  # surface any failure in the UI rather than crashing
            st.error(f"Paper step failed: {exc}")
    if acct["kind"] == "paper":
        st.caption(f"To trade automatically, run `python -m paperbot paper-run {choice}` in a terminal.")

    summary = account_summary(db, choice)
    state = summary["account"]["state"]
    if summary["metrics"] is None:
        st.info("No candles processed yet.")
        return
    kpi_row(summary["metrics"])
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Equity", f"${summary['equity']['equity'].iloc[-1]:,.2f}")
    s2.metric("Cash", f"${state.cash:,.2f}")
    s3.metric("Position", f"{state.qty:.6f}")
    s4.metric("Pending order", {None: "none", 1.0: "BUY next open", 0.0: "SELL next open"}
              .get(state.pending_target, str(state.pending_target)))

    st.subheader("Equity vs buy & hold")
    st.plotly_chart(equity_chart(summary["equity"]["equity"], summary["benchmark"]), width="stretch", key="pp_equity")
    st.plotly_chart(drawdown_chart(summary["equity"]["equity"], summary["benchmark"]), width="stretch", key="pp_dd")

    candles = db.load_candles(acct["source"], acct["symbol"], acct["timeframe"], limit=1500)
    if len(candles):
        strat = build_strategy(acct["strategy"])
        st.subheader("Price, indicators and fills")
        st.plotly_chart(price_chart(candles, strat.indicators(candles), summary["fills"]),
                        width="stretch", key="pp_price")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Trade history")
        trades_table(summary["trades"])
    with c2:
        st.subheader("All metrics")
        all_metrics(summary["metrics"])
    with st.expander("Fills"):
        fills_table(summary["fills"])


def main() -> None:
    st.title("Crypto Paper Trading Bot")
    st.caption("Simulation only: public read-only market data, a fictional balance, and no real orders.")
    market, strategy, execution = sidebar()
    tab_bt, tab_paper = st.tabs(["Backtest", "Paper trading"])
    with tab_bt:
        backtest_view(market, strategy, execution)
    with tab_paper:
        paper_view(market, strategy, execution)


main()
