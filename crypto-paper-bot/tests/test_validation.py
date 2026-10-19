import numpy as np
import pandas as pd
import pytest

from paperbot.validation import DataValidationError, validate_ohlcv
from tests.conftest import make_candles


def test_clean_data_passes_unchanged():
    df = make_candles(np.linspace(100, 110, 50))
    out = validate_ohlcv(df, "1h")
    pd.testing.assert_frame_equal(out, df.astype(float))


def test_sorts_and_drops_duplicates():
    df = make_candles(np.linspace(100, 110, 50))
    shuffled = pd.concat([df.iloc[25:], df.iloc[:25], df.iloc[[3]]])
    out = validate_ohlcv(shuffled, "1h")
    assert out.index.is_monotonic_increasing and out.index.is_unique and len(out) == 50


def test_drops_invalid_rows():
    df = make_candles(np.linspace(100, 110, 100))
    df.iloc[5, df.columns.get_loc("close")] = np.nan
    df.iloc[6, df.columns.get_loc("high")] = df["low"].iloc[6] * 0.5  # high below low
    df.iloc[7, df.columns.get_loc("volume")] = -1
    df.iloc[8, df.columns.get_loc("open")] = 0
    out = validate_ohlcv(df, "1h")
    assert len(out) == 96
    assert not out.index.isin(df.index[5:9]).any()


def test_rejects_mostly_bad_data():
    df = make_candles(np.linspace(100, 110, 20))
    df.loc[df.index[:5], "close"] = np.nan
    with pytest.raises(DataValidationError):
        validate_ohlcv(df, "1h")


def test_rejects_naive_index_and_missing_columns():
    df = make_candles(np.linspace(100, 110, 20))
    with pytest.raises(DataValidationError):
        validate_ohlcv(df.tz_localize(None), "1h")
    with pytest.raises(DataValidationError):
        validate_ohlcv(df.drop(columns="volume"), "1h")
    with pytest.raises(DataValidationError):
        validate_ohlcv(df.iloc[:0], "1h")


def test_drops_misaligned_timestamps():
    df = make_candles(np.linspace(100, 110, 100))
    idx = df.index.tolist()
    idx[10] = idx[10] + pd.Timedelta(minutes=7)
    df.index = pd.DatetimeIndex(idx, name="time")
    assert len(validate_ohlcv(df, "1h")) == 99


def test_gaps_are_not_filled():
    df = make_candles(np.linspace(100, 110, 100)).drop(index=make_candles(np.ones(100)).index[40:45])
    assert len(validate_ohlcv(df, "1h")) == 95
