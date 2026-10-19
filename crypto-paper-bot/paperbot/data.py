"""Read-only market data from public exchange endpoints.

Only unauthenticated HTTP GET requests to public market-data endpoints are
made here. No API keys are read or sent, and nothing in this module (or this
project) can place orders or move funds.

Sources:
  * coinbase  - Coinbase Exchange public candles (deep history, paginated)
  * kraken    - Kraken public OHLC (only the most recent 720 candles)
  * synthetic - deterministic random walk for offline demos and tests;
                clearly NOT real market data
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import pandas as pd
import requests

from .config import timeframe_seconds
from .validation import OHLCV_COLUMNS, validate_ohlcv

log = logging.getLogger(__name__)

USER_AGENT = "paperbot/0.1 (read-only paper trading research)"


class DataSourceError(RuntimeError):
    """Raised when a data source cannot provide usable candles."""


def utc_now() -> float:
    return time.time()


def _to_frame(rows: list[list[Any]], columns: list[str]) -> pd.DataFrame:
    """Build an OHLCV frame from rows whose `time` column is epoch seconds."""
    if not rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="time"))
    df = pd.DataFrame(rows, columns=columns)
    df.index = pd.to_datetime(df["time"].astype("int64"), unit="s", utc=True)
    df.index.name = "time"
    return df[OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")


def drop_incomplete(df: pd.DataFrame, timeframe: str, now: float) -> pd.DataFrame:
    """Remove candles that have not closed yet as of `now` (epoch seconds).

    A still-forming candle's close/high/low will change, so using it would
    leak information that was not final at decision time.
    """
    if df.empty:
        return df
    step = timeframe_seconds(timeframe)
    close_times = (df.index - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1) + step
    return df[np.asarray(close_times) <= now]


class BaseFetcher(ABC):
    name: str = "base"

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = 15.0,
        max_retries: int = 4,
        backoff: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = utc_now,
    ) -> None:
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.sleep = sleep
        self.clock = clock

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """GET with retries on network errors, 429 and 5xx."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                log.warning("%s: network error (%s), attempt %d", self.name, exc, attempt + 1)
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise DataSourceError(f"{self.name}: response was not JSON") from exc
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = DataSourceError(f"{self.name}: HTTP {resp.status_code}")
                    log.warning("%s: HTTP %d, attempt %d", self.name, resp.status_code, attempt + 1)
                else:
                    raise DataSourceError(
                        f"{self.name}: HTTP {resp.status_code}: {resp.text[:200]}"
                    )
            if attempt < self.max_retries:
                self.sleep(self.backoff * 2**attempt)
        raise DataSourceError(f"{self.name}: request failed after retries: {last_error}")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: float | None = None,
        end: float | None = None,
    ) -> pd.DataFrame:
        """Return validated, closed candles in [start, end] (epoch seconds)."""
        timeframe_seconds(timeframe)  # validate early
        now = self.clock()
        end = now if end is None else min(end, now)
        if start is None:
            start = end - 300 * timeframe_seconds(timeframe)
        if start >= end:
            raise ValueError("start must be before end")
        raw = self._fetch(symbol.upper(), timeframe, float(start), float(end))
        raw = drop_incomplete(raw, timeframe, now)
        if raw.empty:
            raise DataSourceError(f"{self.name}: no closed candles returned for {symbol} {timeframe}")
        df = validate_ohlcv(raw, timeframe)
        log.info("%s: %d %s candles for %s (%s .. %s)", self.name, len(df), timeframe, symbol,
                 df.index[0], df.index[-1])
        return df

    @abstractmethod
    def _fetch(self, symbol: str, timeframe: str, start: float, end: float) -> pd.DataFrame: ...


class CoinbaseFetcher(BaseFetcher):
    """Coinbase Exchange public candles: GET /products/{id}/candles.

    Rows come back newest-first as [time, low, high, open, close, volume],
    at most 300 per request, so longer ranges are paginated.
    """

    name = "coinbase"
    BASE_URL = "https://api.exchange.coinbase.com"
    MAX_CANDLES = 300
    GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}
    REQUEST_SPACING = 0.2  # seconds between pages, well under public rate limits

    def _fetch(self, symbol: str, timeframe: str, start: float, end: float) -> pd.DataFrame:
        gran = self.GRANULARITY[timeframe]
        rows: list[list[Any]] = []
        chunk_start = int(start) - int(start) % gran
        first = True
        while chunk_start <= end:
            chunk_end = min(chunk_start + (self.MAX_CANDLES - 1) * gran, int(end))
            if not first:
                self.sleep(self.REQUEST_SPACING)
            first = False
            data = self._get_json(
                f"{self.BASE_URL}/products/{symbol}/candles",
                {
                    "granularity": gran,
                    "start": pd.Timestamp(chunk_start, unit="s", tz="UTC").isoformat(),
                    "end": pd.Timestamp(chunk_end, unit="s", tz="UTC").isoformat(),
                },
            )
            if isinstance(data, dict):
                raise DataSourceError(f"coinbase: {data.get('message', data)}")
            if not isinstance(data, list):
                raise DataSourceError("coinbase: unexpected response shape")
            for row in data:
                if not isinstance(row, list) or len(row) < 6:
                    raise DataSourceError("coinbase: malformed candle row")
                rows.append(row[:6])
            chunk_start = chunk_end + gran
        df = _to_frame(rows, ["time", "low", "high", "open", "close", "volume"])
        return df[(df.index >= pd.Timestamp(start, unit="s", tz="UTC"))]


class KrakenFetcher(BaseFetcher):
    """Kraken public OHLC: GET /0/public/OHLC.

    Kraken returns at most the 720 most recent candles regardless of `since`,
    so it is suited to paper trading rather than long backtests.
    """

    name = "kraken"
    BASE_URL = "https://api.kraken.com/0/public/OHLC"
    INTERVAL = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": 1440}
    ASSET_ALIASES = {"BTC": "XBT"}

    @classmethod
    def pair(cls, symbol: str) -> str:
        try:
            base, quote = symbol.split("-")
        except ValueError:
            raise ValueError(f"symbol must look like BTC-USD, got {symbol!r}") from None
        return cls.ASSET_ALIASES.get(base, base) + quote

    def _fetch(self, symbol: str, timeframe: str, start: float, end: float) -> pd.DataFrame:
        data = self._get_json(
            self.BASE_URL,
            {"pair": self.pair(symbol), "interval": self.INTERVAL[timeframe], "since": int(start) - 1},
        )
        if not isinstance(data, dict):
            raise DataSourceError("kraken: unexpected response shape")
        if data.get("error"):
            raise DataSourceError(f"kraken: {data['error']}")
        result = data.get("result") or {}
        series = [v for k, v in result.items() if k != "last"]
        if len(series) != 1 or not isinstance(series[0], list):
            raise DataSourceError("kraken: unexpected result payload")
        rows = []
        for row in series[0]:
            if not isinstance(row, list) or len(row) < 7:
                raise DataSourceError("kraken: malformed candle row")
            # [time, open, high, low, close, vwap, volume, count]
            rows.append([row[0], row[1], row[2], row[3], row[4], row[6]])
        df = _to_frame(rows, ["time", "open", "high", "low", "close", "volume"])
        lo = pd.Timestamp(start, unit="s", tz="UTC")
        hi = pd.Timestamp(end, unit="s", tz="UTC")
        if not df.empty and df.index[0] > lo + pd.Timedelta(seconds=timeframe_seconds(timeframe)):
            log.warning("kraken only returns the latest 720 candles; history starts at %s", df.index[0])
        return df[(df.index >= lo) & (df.index <= hi)]


class SyntheticFetcher(BaseFetcher):
    """Deterministic geometric random walk. NOT real market data.

    Used for offline demos and tests. Candles are a pure function of
    (seed, symbol, timeframe, timestamp), so repeated or overlapping fetches
    always agree, like a real exchange history would.
    """

    name = "synthetic"

    def __init__(self, seed: int = 7, start_price: float = 30_000.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.seed = seed
        self.start_price = start_price

    def _fetch(self, symbol: str, timeframe: str, start: float, end: float) -> pd.DataFrame:
        step = timeframe_seconds(timeframe)
        # The walk starts at a fixed origin; intraday frames start later to stay small.
        origin = 1_577_836_800 if step >= 3600 else 1_735_689_600  # 2020-01-01 / 2025-01-01
        first = max(int(start) - int(start) % step, origin)
        times = np.arange(first, int(end) + 1, step, dtype=np.int64)
        if len(times) == 0:
            return _to_frame([], ["time", *OHLCV_COLUMNS])
        idx = (times - origin) // step
        n_total = int(idx[-1]) + 1

        def stream(k: int) -> np.random.Generator:
            # One generator per component, so values at a given index never
            # depend on how far into the future the walk was generated.
            return np.random.default_rng([self.seed, sum(symbol.encode()), step, k])

        vol = 0.6 * np.sqrt(step / (365 * 86400))  # ~60% annualised volatility
        # Slowly varying trend regimes so strategies have something to react to.
        regime = np.repeat(stream(0).normal(0, vol * 0.02, n_total // 200 + 1), 200)[:n_total]
        log_ret = stream(1).normal(0, vol, n_total) + regime
        closes = self.start_price * np.exp(np.cumsum(log_ret))
        opens = np.concatenate([[self.start_price], closes[:-1]])
        highs = np.maximum(opens, closes) * np.exp(np.abs(stream(2).normal(0, vol * 0.5, n_total)))
        lows = np.minimum(opens, closes) * np.exp(-np.abs(stream(3).normal(0, vol * 0.5, n_total)))
        volume = stream(4).lognormal(3, 0.5, n_total)
        rows = np.column_stack([times, opens[idx], highs[idx], lows[idx], closes[idx], volume[idx]])
        return _to_frame(rows.tolist(), ["time", *OHLCV_COLUMNS])


FETCHERS: dict[str, type[BaseFetcher]] = {
    "coinbase": CoinbaseFetcher,
    "kraken": KrakenFetcher,
    "synthetic": SyntheticFetcher,
}


def get_fetcher(source: str, **kwargs: Any) -> BaseFetcher:
    try:
        return FETCHERS[source](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown data source {source!r}; choose one of {sorted(FETCHERS)}") from None


def fetch_history(source: str, symbol: str, timeframe: str, days: float, **kwargs: Any) -> pd.DataFrame:
    fetcher = get_fetcher(source, **kwargs)
    end = fetcher.clock()
    return fetcher.fetch_ohlcv(symbol, timeframe, start=end - days * 86400, end=end)
