"""Shared configuration and constants."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = Path(os.environ.get("PAPERBOT_DB", PROJECT_ROOT / "data" / "paperbot.sqlite"))
DEFAULT_LOG_PATH = Path(os.environ.get("PAPERBOT_LOG", PROJECT_ROOT / "logs" / "paperbot.log"))

STARTING_CASH = 10_000.0

# Candle sizes supported across the data sources, in seconds.
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "1d": 86400,
}

SECONDS_PER_YEAR = 365 * 86400  # crypto trades every day of the year


def timeframe_seconds(timeframe: str) -> int:
    try:
        return TIMEFRAME_SECONDS[timeframe]
    except KeyError:
        raise ValueError(
            f"Unsupported timeframe {timeframe!r}; choose one of {sorted(TIMEFRAME_SECONDS)}"
        ) from None


@dataclass(frozen=True)
class ExecutionConfig:
    """How simulated orders are filled.

    Defaults are deliberately conservative: a 0.40% taker fee (a typical
    entry-tier spot fee on major US exchanges) and 5 bps of slippage per side.
    """

    starting_cash: float = STARTING_CASH
    fee_rate: float = 0.004  # fraction of notional charged per fill
    slippage_bps: float = 5.0  # adverse price move per fill, in basis points
    allocation: float = 1.0  # fraction of available cash committed when entering
    min_notional: float = 10.0  # orders smaller than this (in quote currency) are skipped

    def __post_init__(self) -> None:
        if self.starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if not 0 <= self.fee_rate < 0.1:
            raise ValueError("fee_rate must be in [0, 0.1)")
        if not 0 <= self.slippage_bps < 1000:
            raise ValueError("slippage_bps must be in [0, 1000)")
        if not 0 < self.allocation <= 1:
            raise ValueError("allocation must be in (0, 1]")
        if self.min_notional < 0:
            raise ValueError("min_notional must be >= 0")

    @property
    def slippage(self) -> float:
        return self.slippage_bps / 10_000.0

    def to_dict(self) -> dict:
        return asdict(self)
