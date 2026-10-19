"""Automated PAPER trading.

The paper trader polls public candles, and for every newly *closed* candle
runs exactly the same broker steps as the backtester:

    fill the pending order at this candle's open  ->  mark at close  ->  queue new target

Nothing here talks to an exchange's trading API. There are no credentials,
no order endpoints and no withdrawals; the balance is a fictional $10,000.

A new account starts at the most recent closed candle: it never back-fills
trades from before it existed. If the bot was offline, missed candles are
processed in order on the next run, using the same next-open fill rule.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from .broker import BrokerState, Fill, PaperBroker, Trade
from .config import ExecutionConfig, timeframe_seconds
from .data import BaseFetcher, drop_incomplete, get_fetcher
from .metrics import compute_metrics
from .storage import Database
from .strategies import Strategy, build_strategy
from .validation import validate_ohlcv

log = logging.getLogger(__name__)


@dataclass
class StepReport:
    processed: int = 0
    fills: list[Fill] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    equity: float | None = None
    last_candle: pd.Timestamp | None = None
    pending_target: float | None = None


class PaperTrader:
    SETTLE_SECONDS = 15  # wait after a candle closes so the exchange has finalised it

    def __init__(
        self,
        db: Database,
        account: str,
        fetcher: BaseFetcher | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        max_history: int = 5000,
    ) -> None:
        acct = db.get_account(account)
        if acct is None:
            raise KeyError(f"no paper account named {account!r}; create one first")
        if acct["kind"] != "paper":
            raise ValueError(f"account {account!r} is a {acct['kind']} record, not a paper account")
        self.db = db
        self.name = account
        self.source: str = acct["source"]
        self.symbol: str = acct["symbol"]
        self.timeframe: str = acct["timeframe"]
        self.execution: ExecutionConfig = acct["execution"]
        self.strategy: Strategy = build_strategy(acct["strategy"])
        self.clock = clock
        self.sleep = sleep
        self.fetcher = fetcher or get_fetcher(self.source, clock=clock)
        self.max_history = max_history
        self.step_seconds = timeframe_seconds(self.timeframe)

    @staticmethod
    def create_account(
        db: Database,
        name: str,
        *,
        source: str,
        symbol: str,
        timeframe: str,
        strategy: Strategy,
        execution: ExecutionConfig | None = None,
    ) -> None:
        timeframe_seconds(timeframe)
        execution = execution or ExecutionConfig()
        db.create_account(
            name, kind="paper", source=source, symbol=symbol.upper(), timeframe=timeframe,
            strategy=strategy.to_config(), execution=execution,
        )
        log.info("created paper account %r: %s %s %s, $%.2f fictional cash, %s",
                 name, source, symbol, timeframe, execution.starting_cash, strategy.label())

    def _state(self) -> BrokerState:
        acct = self.db.get_account(self.name)
        if acct is None:
            raise KeyError(f"paper account {self.name!r} disappeared")
        return acct["state"]

    def refresh_candles(self) -> None:
        now = self.clock()
        last = self.db.last_candle_ts(self.source, self.symbol, self.timeframe)
        history_bars = max(3 * self.strategy.warmup, 300)
        if last is None:
            start = now - history_bars * self.step_seconds
        else:
            # Re-fetch a little overlap in case the exchange revised recent candles.
            start = max(last - 3 * self.step_seconds, now - history_bars * self.step_seconds)
        df = self.fetcher.fetch_ohlcv(self.symbol, self.timeframe, start=start, end=now)
        self.db.upsert_candles(self.source, self.symbol, self.timeframe, df)

    def step(self) -> StepReport:
        """Fetch new candles and process every closed candle not yet seen."""
        self.refresh_candles()
        candles = self.db.load_candles(self.source, self.symbol, self.timeframe, limit=self.max_history)
        candles = drop_incomplete(validate_ohlcv(candles, self.timeframe), self.timeframe, self.clock())
        if len(candles) < self.strategy.warmup + 1:
            log.warning("only %d candles cached; %s needs %d to warm up (staying flat until then)",
                        len(candles), self.strategy.label(), self.strategy.warmup)
        targets = self.strategy.target_position(candles).reindex(candles.index).fillna(0.0)

        state = self._state()
        broker = PaperBroker(self.execution, state)
        report = StepReport()
        if state.last_ts is None:
            todo = candles.iloc[-1:]
            starting = True
        else:
            todo = candles[candles.index > state.last_ts]
            starting = False
            if len(todo) > 1:
                log.info("catching up on %d missed candles", len(todo))

        newest_close = candles.index[-1].timestamp() + self.step_seconds
        if self.clock() - newest_close > 3 * self.step_seconds:
            log.warning("latest closed candle %s looks stale; data source may be lagging", candles.index[-1])

        for ts, row in todo.iterrows():
            with self.db.transaction() as conn:
                fills: list[Fill] = []
                trades: list[Trade] = []
                if not starting:
                    ev = broker.on_open(ts, float(row["open"]))
                    fills, trades = ev.fills, ev.trades
                point = broker.on_close(ts, float(row["close"]), float(targets.loc[ts]))
                self.db.record_fills(conn, self.name, fills)
                self.db.record_trades(conn, self.name, trades)
                self.db.record_equity(conn, self.name, [point])
                self.db.save_state(conn, self.name, broker.state)
            starting = False
            for f in fills:
                log.info("PAPER %s %.8f %s @ %.2f (fee %.2f) at %s",
                         f.side.upper(), f.qty, self.symbol, f.price, f.fee, ts)
            for t in trades:
                log.info("PAPER trade closed: pnl %.2f (%.2f%%)", t.pnl, t.return_pct * 100)
            report.processed += 1
            report.fills += fills
            report.trades += trades
            report.equity = point.equity
            report.last_candle = ts
        report.pending_target = broker.state.pending_target
        if report.processed == 0:
            log.debug("no new closed candle yet")
        return report

    def seconds_until_next_candle(self) -> float:
        now = self.clock()
        next_close = (now // self.step_seconds + 1) * self.step_seconds
        return max(next_close + self.SETTLE_SECONDS - now, 1.0)

    def run(self, max_steps: int | None = None) -> None:
        """Run until interrupted, waking shortly after each candle closes."""
        log.info("paper trading %r started (Ctrl+C to stop)", self.name)
        failures = 0
        steps = 0
        try:
            while max_steps is None or steps < max_steps:
                try:
                    report = self.step()
                    failures = 0
                    if report.processed:
                        log.info("processed %d candle(s); equity $%.2f; pending=%s",
                                 report.processed, report.equity, report.pending_target)
                    wait = self.seconds_until_next_candle()
                except Exception:  # keep the loop alive through transient failures
                    failures += 1
                    wait = min(30 * 2 ** (failures - 1), max(self.step_seconds, 60))
                    log.exception("paper step failed (%d in a row); retrying in %ds", failures, wait)
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break
                self.sleep(wait)
        except KeyboardInterrupt:
            log.info("paper trading stopped by user")


def account_summary(db: Database, name: str) -> dict[str, Any]:
    """Equity, trades and metrics for a stored account (paper or backtest)."""
    acct = db.get_account(name)
    if acct is None:
        raise KeyError(name)
    equity = db.load_equity(name)
    fills_df = db.load_fills(name)
    trades_df = db.load_trades(name)
    summary: dict[str, Any] = {"account": acct, "equity": equity, "fills": fills_df, "trades": trades_df}
    if equity.empty:
        summary["metrics"] = None
        summary["benchmark"] = pd.Series(dtype=float)
        return summary
    cfg: ExecutionConfig = acct["execution"]
    # Buy & hold from the account's first mark, paying the same entry costs.
    first_price = float(equity["price"].iloc[0])
    qty = cfg.starting_cash / (first_price * (1 + cfg.slippage) * (1 + cfg.fee_rate))
    benchmark = (equity["price"] * qty).rename("benchmark")
    benchmark.iloc[0] = cfg.starting_cash
    trades = [Trade(**{k: r[k] for k in trades_df.columns}) for _, r in trades_df.iterrows()]
    fills = [Fill(ts=r["ts"], side=r["side"], qty=r["qty"], price=r["price"],
                  reference_price=r["reference_price"], fee=r["fee"], cash_after=r["cash_after"],
                  qty_after=r["qty_after"]) for _, r in fills_df.iterrows()]
    summary["metrics"] = compute_metrics(
        equity["equity"], trades, fills, acct["timeframe"],
        position=(equity["qty"] > 0).astype(float), benchmark=benchmark,
    )
    summary["benchmark"] = benchmark
    return summary
