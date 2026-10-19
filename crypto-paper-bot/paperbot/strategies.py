"""Strategy engine.

A strategy maps candles to a *target position* per bar: 1.0 (long) or 0.0
(flat). Spot, long-only: no leverage or shorting.

The target at bar t is decided from data up to and including bar t's close.
The execution layer acts on it at the *next* bar's open, so strategies never
see the price they trade at. Strategies must be causal (target[t] may not
depend on bars after t); `tests/test_strategies.py` checks this for every
strategy by comparing results on truncated and full histories.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .indicators import moving_average, rsi


class Strategy(ABC):
    kind: str = "base"

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Bars needed before the strategy can produce a meaningful signal."""

    @abstractmethod
    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Indicator columns used by the strategy (for charts and debugging)."""

    @abstractmethod
    def target_position(self, df: pd.DataFrame) -> pd.Series:
        """Target position per bar in {0.0, 1.0}; 0 during warm-up."""

    def to_config(self) -> dict[str, Any]:
        return {"type": self.kind, **asdict(self)}  # type: ignore[call-overload]

    def label(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.to_config().items() if k != "type")
        return f"{self.kind}({params})"


@dataclass
class MACrossover(Strategy):
    """Long while the fast moving average is above the slow one."""

    fast: int = 20
    slow: int = 50
    ma_type: str = "sma"
    kind = "ma_crossover"

    def __post_init__(self) -> None:
        if not 1 <= self.fast < self.slow:
            raise ValueError("ma_crossover needs 1 <= fast < slow")
        if self.ma_type not in ("sma", "ema"):
            raise ValueError("ma_type must be 'sma' or 'ema'")

    @property
    def warmup(self) -> int:
        return self.slow

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                f"{self.ma_type}_{self.fast}": moving_average(df["close"], self.fast, self.ma_type),
                f"{self.ma_type}_{self.slow}": moving_average(df["close"], self.slow, self.ma_type),
            },
            index=df.index,
        )

    def label(self) -> str:
        return f"{self.ma_type.upper()} {self.fast}/{self.slow} crossover"

    def target_position(self, df: pd.DataFrame) -> pd.Series:
        ind = self.indicators(df)
        fast, slow = ind.iloc[:, 0], ind.iloc[:, 1]
        return (fast > slow).astype(float).rename("target")  # NaN comparisons are False


@dataclass
class RSIReversion(Strategy):
    """Mean reversion: buy when RSI drops below `oversold`, sell above `overbought`."""

    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0
    kind = "rsi"

    def __post_init__(self) -> None:
        if self.period < 2:
            raise ValueError("rsi period must be >= 2")
        if not 0 < self.oversold < self.overbought < 100:
            raise ValueError("rsi needs 0 < oversold < overbought < 100")

    @property
    def warmup(self) -> int:
        return self.period + 1

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f"rsi_{self.period}": rsi(df["close"], self.period)}, index=df.index)

    def label(self) -> str:
        return f"RSI({self.period}) buy<{self.oversold:g} sell>{self.overbought:g}"

    def target_position(self, df: pd.DataFrame) -> pd.Series:
        r = self.indicators(df).iloc[:, 0]
        events = pd.Series(np.nan, index=df.index)
        events[r < self.oversold] = 1.0
        events[r > self.overbought] = 0.0
        # Hold the last entry/exit decision in between (a causal forward fill).
        return events.ffill().fillna(0.0).rename("target")


@dataclass
class TrendFilter(Strategy):
    """Only allow longs while close is above a long moving average.

    With no `base` strategy it is a plain trend-following rule.
    """

    period: int = 200
    ma_type: str = "sma"
    base: Strategy | None = field(default=None)
    kind = "trend_filter"

    def __post_init__(self) -> None:
        if self.period < 2:
            raise ValueError("trend filter period must be >= 2")
        if self.ma_type not in ("sma", "ema"):
            raise ValueError("ma_type must be 'sma' or 'ema'")

    @property
    def warmup(self) -> int:
        return max(self.period, self.base.warmup if self.base else 0)

    def indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        trend = pd.DataFrame(
            {f"trend_{self.ma_type}_{self.period}": moving_average(df["close"], self.period, self.ma_type)},
            index=df.index,
        )
        if self.base is None:
            return trend
        return pd.concat([self.base.indicators(df), trend], axis=1)

    def target_position(self, df: pd.DataFrame) -> pd.Series:
        trend = moving_average(df["close"], self.period, self.ma_type)
        allowed = (df["close"] > trend).astype(float)
        if self.base is None:
            return allowed.rename("target")
        return (self.base.target_position(df) * allowed).rename("target")

    def label(self) -> str:
        trend = f"close > {self.ma_type.upper()}{self.period}"
        return f"{self.base.label()} + trend filter ({trend})" if self.base else f"Trend: long while {trend}"

    def to_config(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "period": self.period,
            "ma_type": self.ma_type,
            "base": self.base.to_config() if self.base else None,
        }


STRATEGIES: dict[str, type[Strategy]] = {
    "ma_crossover": MACrossover,
    "rsi": RSIReversion,
    "trend_filter": TrendFilter,
}


def build_strategy(config: dict[str, Any]) -> Strategy:
    """Build a strategy from a plain dict, e.g.

    {"type": "trend_filter", "period": 200, "base": {"type": "ma_crossover", "fast": 20, "slow": 50}}
    """
    cfg = dict(config)
    kind = cfg.pop("type", None)
    if kind not in STRATEGIES:
        raise ValueError(f"unknown strategy type {kind!r}; choose one of {sorted(STRATEGIES)}")
    if kind == "trend_filter" and cfg.get("base") is not None:
        cfg["base"] = build_strategy(cfg["base"])
    try:
        return STRATEGIES[kind](**cfg)
    except TypeError as exc:
        raise ValueError(f"bad parameters for {kind}: {exc}") from None
