"""Performance measurement.

All figures are computed from the simulated equity curve (marked to market at
each candle close) and closed round-trip trades, after fees and slippage.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .broker import Fill, Trade
from .config import SECONDS_PER_YEAR, timeframe_seconds


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak (0 at new highs, negative below)."""
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float(drawdown_series(equity).min())


def compute_metrics(
    equity: pd.Series,
    trades: Sequence[Trade],
    fills: Sequence[Fill],
    timeframe: str,
    position: pd.Series | None = None,
    benchmark: pd.Series | None = None,
) -> dict[str, Any]:
    if equity.empty:
        raise ValueError("empty equity curve")
    start_value = float(equity.iloc[0])
    end_value = float(equity.iloc[-1])
    total_return = end_value / start_value - 1.0

    span_seconds = (equity.index[-1] - equity.index[0]).total_seconds()
    years = span_seconds / SECONDS_PER_YEAR
    cagr = (end_value / start_value) ** (1 / years) - 1 if years >= 30 / 365 and end_value > 0 else None

    rets = equity.pct_change().dropna()
    periods_per_year = SECONDS_PER_YEAR / timeframe_seconds(timeframe)
    std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / std * math.sqrt(periods_per_year)) if std > 0 else None
    downside = rets[rets < 0]
    dstd = float(np.sqrt((downside**2).sum() / len(rets))) if len(rets) else 0.0
    sortino = float(rets.mean() / dstd * math.sqrt(periods_per_year)) if dstd > 0 else None

    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_loss = float(-losses.sum())
    metrics: dict[str, Any] = {
        "start_equity": start_value,
        "end_equity": end_value,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown(equity),
        "volatility_ann": std * math.sqrt(periods_per_year) if std > 0 else 0.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "num_trades": int(len(trades)),
        "win_rate": float(len(wins) / len(pnls)) if len(pnls) else None,
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else None,
        "total_fees": float(sum(f.fee for f in fills)),
        "exposure": float((position > 0).mean()) if position is not None and len(position) else None,
        "bars": int(len(equity)),
    }
    if benchmark is not None and not benchmark.empty:
        b_ret = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0)
        metrics["benchmark_return"] = b_ret
        metrics["benchmark_max_drawdown"] = max_drawdown(benchmark)
        metrics["excess_return"] = total_return - b_ret
    return metrics


METRIC_LABELS = {
    "total_return": ("Total return", "pct"),
    "benchmark_return": ("Buy & hold return", "pct"),
    "excess_return": ("Excess vs buy & hold", "pct"),
    "cagr": ("CAGR", "pct"),
    "max_drawdown": ("Max drawdown", "pct"),
    "benchmark_max_drawdown": ("Buy & hold max drawdown", "pct"),
    "sharpe": ("Sharpe (ann., rf=0)", "num"),
    "sortino": ("Sortino (ann.)", "num"),
    "volatility_ann": ("Volatility (ann.)", "pct"),
    "num_trades": ("Closed trades", "int"),
    "win_rate": ("Win rate", "pct"),
    "profit_factor": ("Profit factor", "num"),
    "avg_win": ("Avg win ($)", "money"),
    "avg_loss": ("Avg loss ($)", "money"),
    "total_fees": ("Fees paid ($)", "money"),
    "exposure": ("Time in market", "pct"),
    "end_equity": ("Ending equity ($)", "money"),
}


def format_metric(value: Any, kind: str) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "money":
        return f"${value:,.2f}"
    if kind == "int":
        return f"{int(value)}"
    return f"{value:.2f}"


def metrics_table(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (label, format_metric(metrics.get(key), kind))
        for key, (label, kind) in METRIC_LABELS.items()
        if key in metrics
    ]
