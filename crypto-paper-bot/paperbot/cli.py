"""Command line interface: `python -m paperbot <command>`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from .backtest import run_backtest
from .config import DEFAULT_DB_PATH, STARTING_CASH, TIMEFRAME_SECONDS, ExecutionConfig
from .data import FETCHERS, DataSourceError, fetch_history
from .logging_setup import setup_logging
from .metrics import metrics_table
from .paper import PaperTrader, account_summary
from .storage import AccountExistsError, Database
from .strategies import Strategy, build_strategy
from .validation import DataValidationError

log = logging.getLogger("paperbot")

PRESETS: dict[str, dict[str, Any]] = {
    "ma": {"type": "ma_crossover", "fast": 20, "slow": 50, "ma_type": "sma"},
    "rsi": {"type": "rsi", "period": 14, "oversold": 30, "overbought": 70},
    "trend": {"type": "trend_filter", "period": 200, "ma_type": "sma"},
    "ma_trend": {"type": "trend_filter", "period": 200, "ma_type": "sma",
                 "base": {"type": "ma_crossover", "fast": 20, "slow": 50, "ma_type": "sma"}},
    "rsi_trend": {"type": "trend_filter", "period": 200, "ma_type": "sma",
                  "base": {"type": "rsi", "period": 14, "oversold": 30, "overbought": 70}},
}


def _strategy(args: argparse.Namespace) -> Strategy:
    if args.strategy_json:
        return build_strategy(json.loads(args.strategy_json))
    return build_strategy(PRESETS[args.strategy])


def _execution(args: argparse.Namespace) -> ExecutionConfig:
    return ExecutionConfig(
        starting_cash=args.cash, fee_rate=args.fee, slippage_bps=args.slippage_bps,
        allocation=args.allocation,
    )


def _add_market_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--source", default="coinbase", choices=sorted(FETCHERS))
    p.add_argument("--symbol", default="BTC-USD", help="e.g. BTC-USD, ETH-USD")
    p.add_argument("--timeframe", default="1h", choices=list(TIMEFRAME_SECONDS))


def _add_strategy_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--strategy", default="ma_trend", choices=sorted(PRESETS))
    p.add_argument("--strategy-json", help="full strategy config as JSON (overrides --strategy)")
    p.add_argument("--cash", type=float, default=STARTING_CASH, help="fictional starting balance")
    p.add_argument("--fee", type=float, default=0.004, help="fee rate per fill (0.004 = 0.40%%)")
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--allocation", type=float, default=1.0, help="fraction of cash used per entry")


def cmd_backtest(args: argparse.Namespace) -> int:
    strategy = _strategy(args)
    candles = fetch_history(args.source, args.symbol, args.timeframe, args.days)
    result = run_backtest(candles, strategy, _execution(args), timeframe=args.timeframe)
    print(f"\nBacktest  {args.symbol} {args.timeframe} [{args.source}]  {strategy.label()}")
    print(f"Period    {result.candles.index[0]}  ->  {result.candles.index[-1]}  ({len(result.candles)} candles)")
    for label, value in metrics_table(result.metrics):
        print(f"  {label:<26} {value}")
    if result.pending_target is not None:
        print(f"  Pending order for next open: target={result.pending_target}")
    if args.save:
        db = Database(args.db)
        db.save_backtest(args.save, result, args.source, args.symbol)
        print(f"Saved as backtest account {args.save!r} in {args.db}")
    print("\nHypothetical results on historical data. Past performance does not predict future returns.")
    return 0


def cmd_paper_init(args: argparse.Namespace) -> int:
    db = Database(args.db)
    PaperTrader.create_account(
        db, args.account, source=args.source, symbol=args.symbol, timeframe=args.timeframe,
        strategy=_strategy(args), execution=_execution(args),
    )
    print(f"Created paper account {args.account!r} with ${args.cash:,.2f} of fictional cash.")
    return 0


def cmd_paper_run(args: argparse.Namespace) -> int:
    trader = PaperTrader(Database(args.db), args.account)
    if args.once:
        report = trader.step()
        print(f"Processed {report.processed} new candle(s); equity "
              f"{'n/a' if report.equity is None else f'${report.equity:,.2f}'}; "
              f"pending target: {report.pending_target}")
    else:
        trader.run()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    db = Database(args.db)
    if not args.account:
        accounts = db.list_accounts()
        print(accounts.to_string(index=False) if len(accounts) else "No accounts yet.")
        return 0
    s = account_summary(db, args.account)
    state = s["account"]["state"]
    print(f"Account {args.account!r}: cash ${state.cash:,.2f}, position {state.qty:.8f} "
          f"{s['account']['symbol']}, last candle {state.last_ts}, pending {state.pending_target}")
    if s["metrics"]:
        for label, value in metrics_table(s["metrics"]):
            print(f"  {label:<26} {value}")
    if len(s["trades"]):
        print("\nRecent trades:")
        print(s["trades"].tail(10).to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paperbot", description="Crypto PAPER trading bot (no real orders).")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backtest", help="backtest a strategy on historical candles")
    _add_market_args(p)
    _add_strategy_args(p)
    p.add_argument("--days", type=float, default=180)
    p.add_argument("--save", metavar="NAME", help="store the result in SQLite under this name")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("paper-init", help="create a paper trading account")
    p.add_argument("account")
    _add_market_args(p)
    _add_strategy_args(p)
    p.set_defaults(func=cmd_paper_init)

    p = sub.add_parser("paper-run", help="run the paper trader")
    p.add_argument("account")
    p.add_argument("--once", action="store_true", help="process new candles once and exit")
    p.set_defaults(func=cmd_paper_run)

    p = sub.add_parser("status", help="list accounts, or show one account's performance")
    p.add_argument("account", nargs="?")
    p.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return args.func(args)
    except (DataSourceError, DataValidationError, AccountExistsError, KeyError, ValueError) as exc:
        log.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
