"""OHLCV data validation.

Every candle DataFrame used for signals or simulated fills goes through
`validate_ohlcv`. Bad rows are never repaired with invented prices: they are
dropped and logged, and if too many are bad the data is rejected outright.
Gaps are reported but not filled, because forward-filling would fabricate
prices that could have been traded on.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import timeframe_seconds

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataValidationError(ValueError):
    """Raised when market data is unusable."""


def validate_ohlcv(
    df: pd.DataFrame,
    timeframe: str | None = None,
    *,
    max_bad_fraction: float = 0.05,
) -> pd.DataFrame:
    """Return a cleaned copy of `df`, or raise DataValidationError.

    Expects a UTC DatetimeIndex (candle open time) and OHLCV columns.
    """
    if df is None or len(df) == 0:
        raise DataValidationError("no candles")
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("index must be a DatetimeIndex of candle open times")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise DataValidationError("index must be timezone-aware UTC")

    out = df[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce").astype(float)
    out.index.name = "time"
    n_input = len(out)

    if not out.index.is_monotonic_increasing:
        log.warning("candles were not sorted by time; sorting")
        out = out.sort_index()
    dupes = out.index.duplicated(keep="last")
    if dupes.any():
        log.warning("dropping %d duplicate candle timestamps", int(dupes.sum()))
        out = out[~dupes]

    values = out.to_numpy()
    bad = ~np.isfinite(values).all(axis=1)
    bad |= (out[["open", "high", "low", "close"]] <= 0).any(axis=1).to_numpy()
    bad |= (out["volume"] < 0).to_numpy()
    body_high = out[["open", "close"]].max(axis=1)
    body_low = out[["open", "close"]].min(axis=1)
    tol = 1e-9 * out["close"].abs()
    bad |= (out["high"] + tol < body_high).to_numpy()
    bad |= (out["low"] - tol > body_low).to_numpy()
    bad |= (out["high"] + tol < out["low"]).to_numpy()

    if timeframe is not None:
        step = timeframe_seconds(timeframe)
        bad |= (_epoch_seconds(out.index) % step) != 0

    n_bad = int(bad.sum())
    if n_bad:
        frac = n_bad / max(n_input, 1)
        if frac > max_bad_fraction:
            raise DataValidationError(
                f"{n_bad} of {n_input} candles are invalid ({frac:.1%}); refusing to use this data"
            )
        log.warning("dropping %d invalid candles", n_bad)
        out = out[~bad]

    if len(out) == 0:
        raise DataValidationError("no valid candles after cleaning")

    if timeframe is not None and len(out) > 1:
        gaps = np.diff(_epoch_seconds(out.index)) != timeframe_seconds(timeframe)
        if gaps.any():
            log.info("candle series has %d gap(s); gaps are left unfilled", int(gaps.sum()))

    return out


def _epoch_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    return ((index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)
