import pandas as pd
import pytest

from paperbot.backtest import run_backtest
from paperbot.broker import BrokerState
from paperbot.config import ExecutionConfig
from paperbot.storage import Database
from paperbot.strategies import MACrossover


def test_candle_roundtrip_and_upsert(tmp_path, synthetic_hourly):
    db = Database(tmp_path / "db.sqlite")
    db.upsert_candles("synthetic", "BTC-USD", "1h", synthetic_hourly)
    db.upsert_candles("synthetic", "BTC-USD", "1h", synthetic_hourly.iloc[-100:])  # overlap is idempotent
    out = db.load_candles("synthetic", "BTC-USD", "1h")
    pd.testing.assert_frame_equal(out, synthetic_hourly, check_freq=False, check_index_type=False)
    assert len(db.load_candles("synthetic", "BTC-USD", "1h", limit=50)) == 50
    assert db.last_candle_ts("synthetic", "BTC-USD", "1h") == int(synthetic_hourly.index[-1].timestamp())


def test_account_state_roundtrip(tmp_path):
    db = Database(tmp_path / "db.sqlite")
    state = BrokerState(cash=1234.5, qty=0.1, pending_target=0.0,
                        entry_ts=pd.Timestamp("2024-01-01", tz="UTC"), entry_price=50_000.0,
                        entry_cost=5_020.0, entry_fee=20.0, last_ts=pd.Timestamp("2024-02-01", tz="UTC"))
    cfg = ExecutionConfig(fee_rate=0.001)
    db.create_account("a", kind="paper", source="coinbase", symbol="BTC-USD", timeframe="1h",
                      strategy={"type": "rsi"}, execution=cfg, state=state)
    acct = db.get_account("a")
    assert acct["state"] == state and acct["execution"] == cfg and acct["strategy"] == {"type": "rsi"}
    assert db.get_account("missing") is None


def test_save_backtest(tmp_path, synthetic_hourly):
    db = Database(tmp_path / "db.sqlite")
    res = run_backtest(synthetic_hourly, MACrossover(10, 30), timeframe="1h")
    db.save_backtest("bt", res, "synthetic", "BTC-USD")
    db.save_backtest("bt", res, "synthetic", "BTC-USD")  # replacing must not duplicate rows
    assert len(db.load_fills("bt")) == len(res.fills)
    assert len(db.load_trades("bt")) == len(res.trades)
    eq = db.load_equity("bt")
    assert len(eq) == len(res.equity)
    assert eq["equity"].iloc[-1] == pytest.approx(res.equity["equity"].iloc[-1])
    assert list(db.list_accounts()["name"]) == ["bt"]
