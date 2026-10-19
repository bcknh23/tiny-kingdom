import pandas as pd
import pytest

from paperbot.backtest import simulate
from paperbot.config import ExecutionConfig
from paperbot.data import SyntheticFetcher
from paperbot.paper import PaperTrader, account_summary
from paperbot.storage import AccountExistsError, Database
from paperbot.strategies import MACrossover, TrendFilter, build_strategy
from tests.conftest import NOW

H = 3600


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.sqlite")


def make_trader(db, clock, name="acct"):
    return PaperTrader(db, name, fetcher=SyntheticFetcher(seed=11, clock=clock), clock=clock)


def test_paper_trading_matches_backtest(db):
    strategy = TrendFilter(60, base=MACrossover(5, 20))
    cfg = ExecutionConfig()
    PaperTrader.create_account(db, "acct", source="synthetic", symbol="BTC-USD", timeframe="1h",
                               strategy=strategy, execution=cfg)
    clock = Clock(NOW + 5)
    trader = make_trader(db, clock)
    first = trader.step()
    assert first.processed == 1 and first.fills == []  # starts now; no back-filled trades

    for _ in range(400):  # one candle at a time, restarting the trader now and then
        clock.t += H
        if clock.t % (37 * H) < H:
            trader = make_trader(db, clock)
        trader.step()
    clock.t += 30 * H  # bot offline for a day; it must catch up
    assert make_trader(db, clock).step().processed == 30

    # Re-run the same candles through the backtest machinery from the account's start.
    candles = db.load_candles("synthetic", "BTC-USD", "1h")
    targets = build_strategy(strategy.to_config()).target_position(candles)
    start = db.load_equity("acct").index[0]
    _, fills, trades, broker = simulate(candles[candles.index >= start], targets[targets.index >= start], cfg)

    stored = db.load_fills("acct")
    assert len(stored) == len(fills) > 4
    for (_, row), f in zip(stored.iterrows(), fills):
        assert row["ts"] == f.ts and row["side"] == f.side
        assert row["price"] == pytest.approx(f.price) and row["qty"] == pytest.approx(f.qty)
    assert len(db.load_trades("acct")) == len(trades)
    state = db.get_account("acct")["state"]
    assert state.cash == pytest.approx(broker.state.cash) and state.qty == pytest.approx(broker.state.qty)
    assert len(db.load_equity("acct")) == 1 + 400 + 30


def test_step_without_new_candle_is_a_noop(db):
    PaperTrader.create_account(db, "acct", source="synthetic", symbol="BTC-USD", timeframe="1h",
                               strategy=MACrossover(5, 20))
    clock = Clock(NOW + 5)
    trader = make_trader(db, clock)
    trader.step()
    clock.t += 1800  # half a candle later: the forming candle must be ignored
    assert trader.step().processed == 0
    assert len(db.load_equity("acct")) == 1


def test_account_errors(db):
    PaperTrader.create_account(db, "acct", source="synthetic", symbol="BTC-USD", timeframe="1h",
                               strategy=MACrossover(5, 20))
    with pytest.raises(AccountExistsError):
        PaperTrader.create_account(db, "acct", source="synthetic", symbol="BTC-USD", timeframe="1h",
                                   strategy=MACrossover(5, 20))
    with pytest.raises(KeyError):
        PaperTrader(db, "missing")


def test_failed_step_leaves_state_untouched(db):
    PaperTrader.create_account(db, "acct", source="synthetic", symbol="BTC-USD", timeframe="1h",
                               strategy=MACrossover(5, 20))
    clock = Clock(NOW + 5)
    trader = make_trader(db, clock)
    trader.step()
    before = db.get_account("acct")["state"]

    def broken(*a, **k):
        raise RuntimeError("disk full")

    db.record_equity = broken
    clock.t += 3 * H
    with pytest.raises(RuntimeError):
        trader.step()
    assert db.get_account("acct")["state"] == before


def test_run_loop_survives_errors(db):
    PaperTrader.create_account(db, "acct", source="synthetic", symbol="BTC-USD", timeframe="1h",
                               strategy=MACrossover(5, 20))
    clock = Clock(NOW + 5)
    sleeps = []
    trader = PaperTrader(db, "acct", fetcher=SyntheticFetcher(clock=clock), clock=clock, sleep=sleeps.append)
    calls = {"n": 0}
    real_step = trader.step

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("network down")
        return real_step()

    trader.step = flaky
    trader.run(max_steps=3)
    assert calls["n"] == 3 and len(sleeps) == 2 and sleeps[0] == 30


def test_account_summary(db):
    PaperTrader.create_account(db, "acct", source="synthetic", symbol="BTC-USD", timeframe="1h",
                               strategy=MACrossover(3, 8))
    clock = Clock(NOW + 5)
    trader = make_trader(db, clock)
    for _ in range(200):
        trader.step()
        clock.t += H
    s = account_summary(db, "acct")
    m = s["metrics"]
    assert m["num_trades"] == len(s["trades"]) > 0
    assert m["start_equity"] == 10_000
    assert "benchmark_return" in m and len(s["benchmark"]) == len(s["equity"])
