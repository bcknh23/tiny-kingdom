"""Simulated execution: the single source of truth for fills and P&L.

Both the backtester and the live paper trader drive the same `PaperBroker`
through the same two events per candle:

  on_open(bar)   fill any order decided at the previous candle's close, at
                 this candle's open price adjusted for slippage, plus fees
  on_close(bar)  mark the portfolio to this candle's close and record the
                 strategy's new target, to be executed at the next open

Because the decision at close t can only be filled at open t+1, a strategy
can never trade at a price it used to make the decision.

There is no exchange connection here: fills are arithmetic on a fictional
cash balance.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from .config import ExecutionConfig

log = logging.getLogger(__name__)

EPS = 1e-12


@dataclass
class Fill:
    ts: pd.Timestamp
    side: str  # "buy" | "sell"
    qty: float
    price: float  # execution price after slippage
    reference_price: float  # candle open before slippage
    fee: float
    cash_after: float
    qty_after: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trade:
    """A closed round trip (entry and exit), net of all fees and slippage."""

    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    qty: float
    entry_price: float
    exit_price: float
    fees: float
    pnl: float
    return_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EquityPoint:
    ts: pd.Timestamp
    cash: float
    qty: float
    price: float
    equity: float


@dataclass
class BrokerState:
    """Everything needed to resume a paper account exactly where it stopped."""

    cash: float
    qty: float = 0.0
    pending_target: float | None = None
    entry_ts: pd.Timestamp | None = None
    entry_price: float = 0.0
    entry_cost: float = 0.0  # cash spent on the open position, fees included
    entry_fee: float = 0.0
    last_ts: pd.Timestamp | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["entry_ts"] = self.entry_ts.isoformat() if self.entry_ts is not None else None
        d["last_ts"] = self.last_ts.isoformat() if self.last_ts is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BrokerState":
        d = dict(d)
        for key in ("entry_ts", "last_ts"):
            d[key] = pd.Timestamp(d[key]) if d.get(key) else None
        return cls(**d)


@dataclass
class BarEvents:
    fills: list[Fill] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)


class PaperBroker:
    def __init__(self, config: ExecutionConfig, state: BrokerState | None = None) -> None:
        self.config = config
        self.state = state or BrokerState(cash=config.starting_cash)

    @property
    def in_position(self) -> bool:
        return self.state.qty > EPS

    def on_open(self, ts: pd.Timestamp, open_price: float) -> BarEvents:
        """Execute the pending order (if any) at this candle's open."""
        self._check_time(ts)
        events = BarEvents()
        target = self.state.pending_target
        self.state.pending_target = None
        if target is None or open_price <= 0:
            return events
        if target >= 0.5 and not self.in_position:
            fill = self._buy(ts, open_price)
            if fill:
                events.fills.append(fill)
        elif target < 0.5 and self.in_position:
            fill, trade = self._sell(ts, open_price)
            events.fills.append(fill)
            events.trades.append(trade)
        return events

    def on_close(self, ts: pd.Timestamp, close_price: float, target: float) -> EquityPoint:
        """Mark to market at the close and queue the new target for next open."""
        self._check_time(ts)
        self.state.last_ts = ts
        if target not in (0.0, 1.0):
            raise ValueError(f"target position must be 0 or 1, got {target!r}")
        want_long = target >= 0.5
        if want_long != self.in_position:
            self.state.pending_target = float(target)
        equity = self.state.cash + self.state.qty * close_price
        return EquityPoint(ts, self.state.cash, self.state.qty, close_price, equity)

    def _buy(self, ts: pd.Timestamp, open_price: float) -> Fill | None:
        cfg = self.config
        price = open_price * (1 + cfg.slippage)
        budget = self.state.cash * cfg.allocation
        if budget < cfg.min_notional:
            log.info("skipping buy at %s: budget %.2f below min notional", ts, budget)
            return None
        qty = budget / (price * (1 + cfg.fee_rate))
        notional = qty * price
        fee = notional * cfg.fee_rate
        self.state.cash -= notional + fee
        if self.state.cash < 0:
            if self.state.cash < -1e-6:
                raise AssertionError("simulated cash went negative")
            self.state.cash = 0.0
        self.state.qty += qty
        self.state.entry_ts = ts
        self.state.entry_price = price
        self.state.entry_cost = notional + fee
        self.state.entry_fee = fee
        return Fill(ts, "buy", qty, price, open_price, fee, self.state.cash, self.state.qty)

    def _sell(self, ts: pd.Timestamp, open_price: float) -> tuple[Fill, Trade]:
        cfg = self.config
        price = open_price * (1 - cfg.slippage)
        qty = self.state.qty
        proceeds = qty * price
        fee = proceeds * cfg.fee_rate
        self.state.cash += proceeds - fee
        self.state.qty = 0.0
        pnl = proceeds - fee - self.state.entry_cost
        trade = Trade(
            entry_ts=self.state.entry_ts,  # type: ignore[arg-type]
            exit_ts=ts,
            qty=qty,
            entry_price=self.state.entry_price,
            exit_price=price,
            fees=self.state.entry_fee + fee,
            pnl=pnl,
            return_pct=pnl / self.state.entry_cost if self.state.entry_cost > 0 else 0.0,
        )
        self.state.entry_ts = None
        self.state.entry_price = self.state.entry_cost = self.state.entry_fee = 0.0
        return Fill(ts, "sell", qty, price, open_price, fee, self.state.cash, 0.0), trade

    def _check_time(self, ts: pd.Timestamp) -> None:
        last = self.state.last_ts
        if last is not None and ts <= last:
            raise ValueError(f"candle {ts} is not newer than the last processed candle {last}")
