import numpy as np
import pandas as pd
import pytest

from paperbot.indicators import ema, rsi, sma


def test_sma_matches_manual_mean():
    s = pd.Series([1.0, 2, 3, 4, 5])
    out = sma(s, 3)
    assert out.isna().sum() == 2
    assert out.iloc[2:].tolist() == [2.0, 3.0, 4.0]


def test_ema_warmup_is_nan():
    out = ema(pd.Series(np.arange(10.0)), 5)
    assert out.iloc[:4].isna().all() and out.iloc[4:].notna().all()


def test_rsi_extremes_and_bounds():
    up = rsi(pd.Series(np.arange(1.0, 40.0)), 14)
    assert up.iloc[:14].isna().all()
    assert (up.dropna() == 100).all()
    flat = rsi(pd.Series(np.ones(30)), 14)
    assert (flat.dropna() == 50).all()
    rng = np.random.default_rng(0)
    noisy = rsi(pd.Series(100 + rng.normal(0, 1, 500).cumsum()), 14).dropna()
    assert ((noisy >= 0) & (noisy <= 100)).all()


def test_rsi_known_value():
    # Seed window of 14 changes: 7 gains of +1 and 7 losses of -1 -> RS = 1 -> RSI = 50.
    closes = [10.0]
    for i in range(14):
        closes.append(closes[-1] + (1 if i % 2 == 0 else -1))
    assert rsi(pd.Series(closes), 14).iloc[14] == pytest.approx(50.0)


@pytest.mark.parametrize("fn", [sma, ema, rsi])
def test_indicators_are_causal(fn):
    rng = np.random.default_rng(1)
    s = pd.Series(100 + rng.normal(0, 1, 400).cumsum())
    full = fn(s, 14)
    part = fn(s.iloc[:250], 14)
    pd.testing.assert_series_equal(full.iloc[:250], part, check_names=False)


def test_bad_period_rejected():
    with pytest.raises(ValueError):
        sma(pd.Series([1.0, 2.0]), 0)
