"""Event-driven backtester with next-candle execution.

For each candle t (in time order):
  1. any order decided at close t-1 fills at open t (+slippage, +fees)
  2. equity is marked at close t
  3. the strategy's target for t (computed from data <= t) is queued

The order decided on the final candle is left pending and reported, never
filled at a price that does not exist yet.

The buy-and-hold benchmark goes through the same broker with the same costs,
entering at the same first possible fill (the open after the first bar).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .broker import BrokerState, EquityPoint, Fill, PaperBroker, Trade
from .config import ExecutionConfig
from .metrics import compute_metrics, drawdown_series
from .strategies import Strategy
from .validation import validate_ohlcv

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    candles: pd.DataFrame
    signals: pd.Series  # strategy target per bar (decided at close)
    indicators: pd.DataFrame
    equity: pd.DataFrame  # cash, qty, price, equity, drawdown, position
    benchmark: pd.Series
    fills: list[Fill]
    trades: list[Trade]
    metrics: dict[str, Any]
    final_state: BrokerState
    strategy: Strategy
    config: ExecutionConfig
    timeframe: str
    pending_target: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def fills_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self.fills])

    def trades_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_dict() for t in self.trades])


def simulate(
    candles: pd.DataFrame, targets: pd.Series, config: ExecutionConfig
) -> tuple[list[EquityPoint], list[Fill], list[Trade], PaperBroker]:
    """Drive a fresh broker through `candles` with precomputed `targets`."""
    if not targets.index.equals(candles.index):
        raise ValueError("targets must be aligned with candles")
    broker = PaperBroker(config)
    points: list[EquityPoint] = []
    fills: list[Fill] = []
    trades: list[Trade] = []
    opens = candles["open"].to_numpy()
    closes = candles["close"].to_numpy()
    tgt = targets.to_numpy(dtype=float)
    for i, ts in enumerate(candles.index):
        if i > 0:
            ev = broker.on_open(ts, float(opens[i]))
            fills.extend(ev.fills)
            trades.extend(ev.trades)
        points.append(broker.on_close(ts, float(closes[i]), float(tgt[i])))
    return points, fills, trades, broker


def _points_frame(points: list[EquityPoint]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "cash": [p.cash for p in points],
            "qty": [p.qty for p in points],
            "price": [p.price for p in points],
            "equity": [p.equity for p in points],
        },
        index=pd.DatetimeIndex([p.ts for p in points], name="time"),
    )
    return df


def run_backtest(
    candles: pd.DataFrame,
    strategy: Strategy,
    config: ExecutionConfig | None = None,
    timeframe: str = "1h",
) -> BacktestResult:
    config = config or ExecutionConfig()
    candles = validate_ohlcv(candles, timeframe)
    if len(candles) < strategy.warmup + 2:
        raise ValueError(
            f"need at least {strategy.warmup + 2} candles for {strategy.label()}, got {len(candles)}"
        )
    targets = strategy.target_position(candles).reindex(candles.index).fillna(0.0)

    points, fills, trades, broker = simulate(candles, targets, config)
    equity = _points_frame(points)
    equity["position"] = (equity["qty"] > 0).astype(float)
    equity["drawdown"] = drawdown_series(equity["equity"])

    bench_points, _, _, _ = simulate(candles, pd.Series(1.0, index=candles.index), config)
    benchmark = _points_frame(bench_points)["equity"].rename("benchmark")

    metrics = compute_metrics(
        equity["equity"], trades, fills, timeframe, position=equity["position"], benchmark=benchmark
    )
    log.info(
        "backtest %s: %d bars, %d trades, return %.2f%% (buy&hold %.2f%%), max DD %.2f%%",
        strategy.label(), len(candles), len(trades), metrics["total_return"] * 100,
        metrics["benchmark_return"] * 100, metrics["max_drawdown"] * 100,
    )
    return BacktestResult(
        candles=candles,
        signals=targets,
        indicators=strategy.indicators(candles),
        equity=equity,
        benchmark=benchmark,
        fills=fills,
        trades=trades,
        metrics=metrics,
        final_state=broker.state,
        strategy=strategy,
        config=config,
        timeframe=timeframe,
        pending_target=broker.state.pending_target,
    )
