import numpy as np
import pandas as pd
import pytest

from paperbot.strategies import MACrossover, RSIReversion, TrendFilter, build_strategy
from tests.conftest import make_candles

STRATEGIES = [
    MACrossover(5, 20),
    MACrossover(10, 30, "ema"),
    RSIReversion(14, 30, 70),
    TrendFilter(50),
    TrendFilter(100, base=MACrossover(10, 40)),
    TrendFilter(100, "ema", base=RSIReversion(14, 35, 65)),
]


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.label())
def test_no_lookahead(strategy, synthetic_hourly):
    """target[t] must not change when bars after t are added."""
    full = strategy.target_position(synthetic_hourly)
    for cut in (300, 1000, 2200):
        part = strategy.target_position(synthetic_hourly.iloc[:cut])
        pd.testing.assert_series_equal(full.iloc[:cut], part)


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: s.label())
def test_targets_are_binary_and_flat_during_warmup(strategy, synthetic_hourly):
    t = strategy.target_position(synthetic_hourly)
    assert set(t.unique()) <= {0.0, 1.0}
    assert (t.iloc[: strategy.warmup - 1] == 0).all()


def test_ma_crossover_direction():
    df = make_candles(np.concatenate([np.linspace(100, 50, 60), np.linspace(50, 150, 60)]))
    t = MACrossover(5, 20).target_position(df)
    assert t.iloc[50] == 0 and t.iloc[-1] == 1


def test_rsi_enters_oversold_and_holds_until_overbought():
    closes = np.concatenate([np.linspace(100, 70, 30), np.full(10, 70.0), np.linspace(70, 110, 30)])
    t = RSIReversion(14, 30, 70).target_position(make_candles(closes))
    first_long = int(np.argmax(t.to_numpy() == 1))
    assert 14 <= first_long < 30  # entered on the sell-off
    assert (t.iloc[first_long:40] == 1).all()  # held through the flat stretch
    assert t.iloc[-1] == 0  # exited after the rally pushed RSI above 70


def test_trend_filter_blocks_longs_in_downtrend():
    df = make_candles(np.linspace(200, 100, 300) + np.sin(np.arange(300)) * 3)
    base = MACrossover(3, 8)
    assert base.target_position(df).sum() > 0
    assert TrendFilter(100, base=base).target_position(df).sum() == 0


def test_build_strategy_roundtrip_and_errors():
    for s in STRATEGIES:
        assert build_strategy(s.to_config()).to_config() == s.to_config()
    with pytest.raises(ValueError):
        build_strategy({"type": "nope"})
    with pytest.raises(ValueError):
        build_strategy({"type": "ma_crossover", "fast": 50, "slow": 20})
    with pytest.raises(ValueError):
        build_strategy({"type": "rsi", "bogus": 1})
