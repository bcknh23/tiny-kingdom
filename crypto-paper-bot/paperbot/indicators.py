"""Technical indicators.

Every indicator here is causal: the value at bar t uses only bars <= t.
Warm-up bars are NaN rather than partial-window guesses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    _check_period(period)
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    _check_period(period)
    out = series.ewm(span=period, adjust=False, min_periods=period).mean()
    return out


def moving_average(series: pd.Series, period: int, kind: str = "sma") -> pd.Series:
    if kind == "sma":
        return sma(series, period)
    if kind == "ema":
        return ema(series, period)
    raise ValueError(f"unknown moving average type {kind!r}; use 'sma' or 'ema'")


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, seeded with a simple average of the first `period` changes."""
    _check_period(period)
    values = close.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    if len(values) <= period:
        return pd.Series(out, index=close.index, name="rsi")
    delta = np.diff(values)
    gains = np.clip(delta, 0, None)
    losses = np.clip(-delta, 0, None)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        out[i + 1] = _rsi_value(avg_gain, avg_loss)
    return pd.Series(out, index=close.index, name="rsi")


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 50.0 if avg_gain == 0 else 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _check_period(period: int) -> None:
    if not isinstance(period, (int, np.integer)) or period < 1:
        raise ValueError(f"period must be a positive integer, got {period!r}")
