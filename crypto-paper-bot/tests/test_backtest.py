import numpy as np
import pandas as pd
import pytest

from paperbot.backtest import run_backtest, simulate
from paperbot.broker import PaperBroker
from paperbot.config import ExecutionConfig
from paperbot.metrics import compute_metrics, max_drawdown
from paperbot.strategies import MACrossover, Strategy, TrendFilter
from tests.conftest import make_candles

FREE = ExecutionConfig(fee_rate=0.0, slippage_bps=0.0)


class FixedTargets(Strategy):
    """Test helper: replays a precomputed target series."""

    kind = "fixed"

    def __init__(self, targets):
        self.targets = list(map(float, targets))

    @property
    def warmup(self):
        return 0

    def indicators(self, df):
        return pd.DataFrame(index=df.index)

    def target_position(self, df):
        return pd.Series(self.targets[: len(df)], index=df.index)

    def to_config(self):
        return {"type": self.kind}


def test_signal_fills_at_next_open_with_fees_and_slippage():
    closes = [100, 101, 102, 103, 104, 105]
    opens = [100, 100.5, 101.5, 102.5, 103.5, 104.5]
    df = make_candles(closes, opens)
    cfg = ExecutionConfig(fee_rate=0.001, slippage_bps=10)
    # Decide long at close of bar 1, flat at close of bar 3.
    res = run_backtest(df, FixedTargets([0, 1, 1, 0, 0, 0]), cfg, timeframe="1h")
    buy, sell = res.fills
    assert buy.ts == df.index[2] and buy.reference_price == 101.5
    assert buy.price == pytest.approx(101.5 * 1.001)
    assert buy.qty == pytest.approx(10_000 / (buy.price * 1.001))
    assert buy.fee == pytest.approx(buy.qty * buy.price * 0.001)
    assert buy.cash_after == pytest.approx(0, abs=1e-6)
    assert sell.ts == df.index[4] and sell.price == pytest.approx(103.5 * 0.999)
    expected_cash = buy.qty * sell.price * (1 - 0.001)
    assert res.equity["equity"].iloc[-1] == pytest.approx(expected_cash)
    trade = res.trades[0]
    assert trade.pnl == pytest.approx(expected_cash - 10_000)
    assert trade.fees == pytest.approx(buy.fee + sell.fee)


def test_last_bar_signal_stays_pending():
    df = make_candles([100, 101, 102, 103])
    res = run_backtest(df, FixedTargets([0, 0, 0, 1]), FREE, timeframe="1h")
    assert res.fills == [] and res.pending_target == 1.0


def test_no_lookahead_in_results(synthetic_hourly):
    """Equity and fills up to bar t must not depend on bars after t."""
    strat = TrendFilter(100, base=MACrossover(10, 40))
    full = run_backtest(synthetic_hourly, strat, timeframe="1h")
    cut = 1800
    part = run_backtest(synthetic_hourly.iloc[:cut], strat, timeframe="1h")
    pd.testing.assert_frame_equal(full.equity.iloc[:cut], part.equity)
    assert [f.to_dict() for f in full.fills if f.ts < synthetic_hourly.index[cut]] == \
        [f.to_dict() for f in part.fills]


def test_future_shock_does_not_change_past():
    """Corrupting future prices must not alter any earlier decision or fill."""
    df = make_candles(100 + np.sin(np.arange(300) / 7) * 10)
    strat = MACrossover(5, 20)
    base = run_backtest(df, strat, timeframe="1h")
    shocked = df.copy()
    shocked.iloc[200:, :4] *= 3
    after = run_backtest(shocked, strat, timeframe="1h")
    pd.testing.assert_frame_equal(base.equity.iloc[:200], after.equity.iloc[:200])


def test_accounting_identities_hold(synthetic_hourly):
    res = run_backtest(synthetic_hourly, MACrossover(10, 30), timeframe="1h")
    eq = res.equity
    assert np.allclose(eq["equity"], eq["cash"] + eq["qty"] * eq["price"])
    assert (eq["cash"] >= 0).all() and (eq["qty"] >= 0).all()
    realised = sum(t.pnl for t in res.trades)
    unrealised = eq["equity"].iloc[-1] - eq["cash"].iloc[-1]
    open_cost = res.final_state.entry_cost
    assert eq["equity"].iloc[-1] == pytest.approx(10_000 + realised + unrealised - open_cost)
    sides = [f.side for f in res.fills]
    assert all(a != b for a, b in zip(sides, sides[1:]))  # strictly alternating buy/sell


def test_costs_reduce_performance(synthetic_hourly):
    strat = MACrossover(10, 30)
    free = run_backtest(synthetic_hourly, strat, FREE, timeframe="1h")
    costly = run_backtest(synthetic_hourly, strat, ExecutionConfig(fee_rate=0.006, slippage_bps=20), timeframe="1h")
    assert costly.metrics["end_equity"] < free.metrics["end_equity"]
    assert costly.metrics["total_fees"] > 0 == free.metrics["total_fees"]


def test_zero_cost_round_trip_at_flat_price_is_breakeven():
    df = make_candles(np.full(10, 50.0))
    res = run_backtest(df, FixedTargets([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), FREE, timeframe="1h")
    assert res.trades[0].pnl == pytest.approx(0.0, abs=1e-9)
    assert res.metrics["end_equity"] == pytest.approx(10_000)


def test_benchmark_is_buy_and_hold_with_same_costs():
    closes = np.linspace(100, 200, 50)
    df = make_candles(closes)
    cfg = ExecutionConfig(fee_rate=0.002, slippage_bps=5)
    res = run_backtest(df, FixedTargets([0] * 50), cfg, timeframe="1h")
    entry = df["open"].iloc[1] * (1 + cfg.slippage)
    qty = 10_000 / (entry * (1 + cfg.fee_rate))
    assert res.benchmark.iloc[0] == 10_000
    assert res.benchmark.iloc[-1] == pytest.approx(qty * closes[-1])
    assert res.metrics["total_return"] == 0 and res.metrics["num_trades"] == 0
    assert res.metrics["win_rate"] is None


def test_allocation_keeps_cash():
    df = make_candles([100, 100, 100, 100])
    cfg = ExecutionConfig(fee_rate=0.0, slippage_bps=0.0, allocation=0.25)
    res = run_backtest(df, FixedTargets([1, 1, 1, 1]), cfg, timeframe="1h")
    assert res.fills[0].cash_after == pytest.approx(7_500)


def test_requires_enough_history():
    with pytest.raises(ValueError):
        run_backtest(make_candles(np.ones(30) * 100), MACrossover(20, 50), timeframe="1h")


def test_broker_rejects_out_of_order_candles():
    b = PaperBroker(FREE)
    t = pd.Timestamp("2024-01-01", tz="UTC")
    b.on_close(t, 100, 0.0)
    with pytest.raises(ValueError):
        b.on_open(t, 100)
    with pytest.raises(ValueError):
        b.on_close(t + pd.Timedelta(hours=1), 100, 0.7)


def test_simulate_requires_aligned_targets():
    df = make_candles([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        simulate(df, pd.Series([1.0, 1.0]), FREE)


def test_metrics_drawdown_and_win_rate():
    eq = pd.Series([100, 120, 90, 130, 65, 70.0],
                   index=pd.date_range("2024-01-01", periods=6, freq="1D", tz="UTC"))
    assert max_drawdown(eq) == pytest.approx(65 / 130 - 1)
    df = make_candles([10, 10, 12, 12, 9, 9, 9], freq="1D")
    res = run_backtest(df, FixedTargets([1, 0, 1, 0, 0, 0, 0]), FREE, timeframe="1d")
    # trade 1: buy open bar1=10, sell open bar2=10 -> 0 pnl (counts as non-win)
    # trade 2: buy open bar3=12, sell open bar4=12 -> 0 pnl
    assert res.metrics["num_trades"] == 2 and res.metrics["win_rate"] == 0.0
    m = compute_metrics(res.equity["equity"], res.trades, res.fills, "1d")
    assert m["max_drawdown"] <= 0
