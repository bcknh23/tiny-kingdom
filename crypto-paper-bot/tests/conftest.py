from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paperbot.data import SyntheticFetcher  # noqa: E402

NOW = 1_780_000_000 - 1_780_000_000 % 86400  # a fixed "now", aligned to midnight UTC


def make_candles(closes, opens=None, start="2024-01-01", freq="1h", spread=0.01) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    opens = np.asarray(opens, dtype=float) if opens is not None else np.concatenate([[closes[0]], closes[:-1]])
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC", name="time")
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) * (1 + spread),
            "low": np.minimum(opens, closes) * (1 - spread),
            "close": closes,
            "volume": 1.0,
        },
        index=idx,
    )


@pytest.fixture
def synthetic_hourly() -> pd.DataFrame:
    f = SyntheticFetcher(seed=3, clock=lambda: NOW)
    return f.fetch_ohlcv("BTC-USD", "1h", start=NOW - 3000 * 3600, end=NOW)
