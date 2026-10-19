import pandas as pd
import pytest
import requests

from paperbot.data import (
    CoinbaseFetcher,
    DataSourceError,
    KrakenFetcher,
    SyntheticFetcher,
    drop_incomplete,
    get_fetcher,
)
from tests.conftest import NOW


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    """Serves canned responses and records every request made."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        out = self.handler(url, params or {})
        if isinstance(out, Exception):
            raise out
        return out


def coinbase_handler(url, params):
    """Mimic Coinbase: newest-first [time, low, high, open, close, volume] rows."""
    gran = params["granularity"]
    start = int(pd.Timestamp(params["start"]).timestamp())
    end = int(pd.Timestamp(params["end"]).timestamp())
    rows = []
    for t in range(start, end + 1, gran):
        if t > NOW:  # exchange includes the still-forming candle
            break
        price = 100 + (t // gran) % 10
        rows.append([t, price - 1, price + 1, price, price + 0.5, 3.0])
    return FakeResponse(200, rows[::-1])


def test_coinbase_paginates_and_orders():
    session = FakeSession(coinbase_handler)
    f = CoinbaseFetcher(session=session, sleep=lambda s: None, clock=lambda: NOW)
    df = f.fetch_ohlcv("BTC-USD", "1h", start=NOW - 1000 * 3600, end=NOW)
    assert len(session.calls) == 4  # 300 candles per request
    assert all(c[0].endswith("/products/BTC-USD/candles") for c in session.calls)
    assert df.index.is_monotonic_increasing and df.index.is_unique
    # The candle opening exactly at NOW has not closed and must be dropped.
    assert df.index[-1] == pd.Timestamp(NOW - 3600, unit="s", tz="UTC")
    assert len(df) == 1000
    row = df.iloc[0]
    assert row.low < row.open < row.close < row.high


def test_retries_then_succeeds_on_429_and_network_errors():
    responses = [FakeResponse(429, {}), requests.ConnectionError("boom")]

    def handler(url, params):
        return responses.pop(0) if responses else coinbase_handler(url, params)

    sleeps = []
    f = CoinbaseFetcher(session=FakeSession(handler), sleep=sleeps.append, clock=lambda: NOW, backoff=0.5)
    df = f.fetch_ohlcv("BTC-USD", "1h", start=NOW - 10 * 3600, end=NOW)
    assert len(df) == 10
    assert sleeps == [0.5, 1.0]


def test_gives_up_after_retries_and_on_client_errors():
    f = CoinbaseFetcher(session=FakeSession(lambda u, p: FakeResponse(503, {})),
                        sleep=lambda s: None, clock=lambda: NOW, max_retries=2)
    with pytest.raises(DataSourceError):
        f.fetch_ohlcv("BTC-USD", "1h", start=NOW - 10 * 3600, end=NOW)
    session = FakeSession(lambda u, p: FakeResponse(404, {"message": "NotFound"}))
    f = CoinbaseFetcher(session=session, sleep=lambda s: None, clock=lambda: NOW)
    with pytest.raises(DataSourceError):
        f.fetch_ohlcv("NOPE-USD", "1h", start=NOW - 10 * 3600, end=NOW)
    assert len(session.calls) == 1  # 4xx is not retried


def test_coinbase_rejects_malformed_rows():
    f = CoinbaseFetcher(session=FakeSession(lambda u, p: FakeResponse(200, [[1, 2]])),
                        sleep=lambda s: None, clock=lambda: NOW)
    with pytest.raises(DataSourceError):
        f.fetch_ohlcv("BTC-USD", "1h", start=NOW - 10 * 3600, end=NOW)


def test_kraken_parsing_and_errors():
    times = [NOW - 3600 * k for k in range(5, -1, -1)]  # includes forming candle at NOW
    rows = [[t, "100.0", "102.0", "99.0", "101.0", "100.5", "7.5", 42] for t in times]
    session = FakeSession(lambda u, p: FakeResponse(200, {"error": [], "result": {"XXBTZUSD": rows, "last": NOW}}))
    f = KrakenFetcher(session=session, sleep=lambda s: None, clock=lambda: NOW)
    df = f.fetch_ohlcv("BTC-USD", "1h", start=NOW - 5 * 3600, end=NOW)
    assert session.calls[0][1]["pair"] == "XBTUSD"
    assert len(df) == 5 and df["volume"].iloc[0] == 7.5 and df["close"].iloc[0] == 101.0

    f = KrakenFetcher(session=FakeSession(lambda u, p: FakeResponse(200, {"error": ["EQuery:Unknown asset pair"]})),
                      sleep=lambda s: None, clock=lambda: NOW)
    with pytest.raises(DataSourceError):
        f.fetch_ohlcv("FOO-USD", "1h", start=NOW - 5 * 3600, end=NOW)


def test_synthetic_is_deterministic_across_overlapping_fetches():
    a = SyntheticFetcher(clock=lambda: NOW).fetch_ohlcv("BTC-USD", "1h", start=NOW - 500 * 3600, end=NOW)
    later = NOW + 50 * 3600
    b = SyntheticFetcher(clock=lambda: later).fetch_ohlcv("BTC-USD", "1h", start=NOW - 300 * 3600, end=later)
    common = a.index.intersection(b.index)
    assert len(common) == 300
    pd.testing.assert_frame_equal(a.loc[common], b.loc[common])


def test_drop_incomplete():
    idx = pd.date_range(pd.Timestamp(NOW - 2 * 3600, unit="s", tz="UTC"), periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
    assert len(drop_incomplete(df, "1h", NOW)) == 2
    assert len(drop_incomplete(df, "1h", NOW + 3600)) == 3


def test_unknown_source_and_timeframe():
    with pytest.raises(ValueError):
        get_fetcher("nasdaq")
    with pytest.raises(ValueError):
        SyntheticFetcher(clock=lambda: NOW).fetch_ohlcv("BTC-USD", "7m")
