"""Local SQLite storage for candles, paper accounts, fills, trades and equity.

Every candle the paper trader processes is written in a single transaction
(fills + trades + equity point + account state), so a crash can never leave
an account half-updated or process the same candle twice.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .broker import BrokerState, EquityPoint, Fill, Trade
from .config import DEFAULT_DB_PATH, ExecutionConfig

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    source TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
    ts INTEGER NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (source, symbol, timeframe, ts)
);
CREATE TABLE IF NOT EXISTS accounts (
    name TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('paper', 'backtest')),
    source TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
    strategy TEXT NOT NULL,          -- JSON strategy config
    execution TEXT NOT NULL,         -- JSON ExecutionConfig
    state TEXT NOT NULL,             -- JSON BrokerState
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT NOT NULL REFERENCES accounts(name) ON DELETE CASCADE,
    ts INTEGER NOT NULL, side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
    qty REAL NOT NULL, price REAL NOT NULL, reference_price REAL NOT NULL,
    fee REAL NOT NULL, cash_after REAL NOT NULL, qty_after REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT NOT NULL REFERENCES accounts(name) ON DELETE CASCADE,
    entry_ts INTEGER NOT NULL, exit_ts INTEGER NOT NULL,
    qty REAL NOT NULL, entry_price REAL NOT NULL, exit_price REAL NOT NULL,
    fees REAL NOT NULL, pnl REAL NOT NULL, return_pct REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS equity (
    account TEXT NOT NULL REFERENCES accounts(name) ON DELETE CASCADE,
    ts INTEGER NOT NULL,
    cash REAL NOT NULL, qty REAL NOT NULL, price REAL NOT NULL, equity REAL NOT NULL,
    PRIMARY KEY (account, ts)
);
CREATE INDEX IF NOT EXISTS fills_account_ts ON fills(account, ts);
CREATE INDEX IF NOT EXISTS trades_account_ts ON trades(account, exit_ts);
"""


def _ts(t: pd.Timestamp) -> int:
    return int(pd.Timestamp(t).timestamp())


def _from_ts(col: pd.Series) -> pd.Series:
    return pd.to_datetime(col.astype("int64"), unit="s", utc=True)


def _now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


class AccountExistsError(ValueError):
    pass


class Database:
    def __init__(self, path: Path | str = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connect() as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    # ---- candles -------------------------------------------------------
    def upsert_candles(self, source: str, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        rows = [
            (source, symbol, timeframe, _ts(ts), r.open, r.high, r.low, r.close, r.volume)
            for ts, r in zip(df.index, df.itertuples(index=False))
        ]
        with self.transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO candles VALUES (?,?,?,?,?,?,?,?,?)", rows
            )
        return len(rows)

    def load_candles(
        self, source: str, symbol: str, timeframe: str, limit: int | None = None
    ) -> pd.DataFrame:
        sql = ("SELECT ts, open, high, low, close, volume FROM candles "
               "WHERE source=? AND symbol=? AND timeframe=? ORDER BY ts DESC")
        params: list[Any] = [source, symbol, timeframe]
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self.connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        df = df.iloc[::-1]
        df.index = pd.DatetimeIndex(_from_ts(df.pop("ts")), name="time")
        return df.astype(float)

    def last_candle_ts(self, source: str, symbol: str, timeframe: str) -> int | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT MAX(ts) FROM candles WHERE source=? AND symbol=? AND timeframe=?",
                (source, symbol, timeframe),
            ).fetchone()
        return row[0] if row and row[0] is not None else None

    # ---- accounts ------------------------------------------------------
    def create_account(
        self,
        name: str,
        *,
        kind: str,
        source: str,
        symbol: str,
        timeframe: str,
        strategy: dict[str, Any],
        execution: ExecutionConfig,
        state: BrokerState | None = None,
        replace: bool = False,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        state = state or BrokerState(cash=execution.starting_cash)
        values = (name, kind, source, symbol, timeframe, json.dumps(strategy),
                  json.dumps(execution.to_dict()), json.dumps(state.to_dict()),
                  _now_iso(), _now_iso())

        def _write(c: sqlite3.Connection) -> None:
            exists = c.execute("SELECT 1 FROM accounts WHERE name=?", (name,)).fetchone()
            if exists:
                if not replace:
                    raise AccountExistsError(f"account {name!r} already exists")
                c.execute("DELETE FROM accounts WHERE name=?", (name,))
            c.execute("INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?,?,?)", values)

        if conn is not None:
            _write(conn)
        else:
            with self.transaction() as c:
                _write(c)

    def get_account(self, name: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM accounts WHERE name=?", (name,)).fetchone()
        if row is None:
            return None
        acct = dict(row)
        acct["strategy"] = json.loads(acct["strategy"])
        acct["execution"] = ExecutionConfig(**json.loads(acct["execution"]))
        acct["state"] = BrokerState.from_dict(json.loads(acct["state"]))
        return acct

    def list_accounts(self, kind: str | None = None) -> pd.DataFrame:
        sql = "SELECT name, kind, source, symbol, timeframe, strategy, created_at, updated_at FROM accounts"
        params: tuple[Any, ...] = ()
        if kind:
            sql += " WHERE kind=?"
            params = (kind,)
        with self.connect() as conn:
            return pd.read_sql_query(sql + " ORDER BY updated_at DESC", conn, params=params)

    def delete_account(self, name: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM accounts WHERE name=?", (name,))

    def save_state(self, conn: sqlite3.Connection, name: str, state: BrokerState) -> None:
        conn.execute(
            "UPDATE accounts SET state=?, updated_at=? WHERE name=?",
            (json.dumps(state.to_dict()), _now_iso(), name),
        )

    # ---- history -------------------------------------------------------
    def record_fills(self, conn: sqlite3.Connection, name: str, fills: list[Fill]) -> None:
        conn.executemany(
            "INSERT INTO fills (account, ts, side, qty, price, reference_price, fee, cash_after, qty_after) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [(name, _ts(f.ts), f.side, f.qty, f.price, f.reference_price, f.fee, f.cash_after, f.qty_after)
             for f in fills],
        )

    def record_trades(self, conn: sqlite3.Connection, name: str, trades: list[Trade]) -> None:
        conn.executemany(
            "INSERT INTO trades (account, entry_ts, exit_ts, qty, entry_price, exit_price, fees, pnl, return_pct) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [(name, _ts(t.entry_ts), _ts(t.exit_ts), t.qty, t.entry_price, t.exit_price, t.fees, t.pnl,
              t.return_pct) for t in trades],
        )

    def record_equity(self, conn: sqlite3.Connection, name: str, points: list[EquityPoint]) -> None:
        conn.executemany(
            "INSERT OR REPLACE INTO equity (account, ts, cash, qty, price, equity) VALUES (?,?,?,?,?,?)",
            [(name, _ts(p.ts), p.cash, p.qty, p.price, p.equity) for p in points],
        )

    def _load(self, sql: str, name: str, ts_cols: tuple[str, ...]) -> pd.DataFrame:
        with self.connect() as conn:
            df = pd.read_sql_query(sql, conn, params=(name,))
        for col in ts_cols:
            df[col] = _from_ts(df[col]) if len(df) else pd.Series(dtype="datetime64[ns, UTC]")
        return df

    def load_fills(self, name: str) -> pd.DataFrame:
        return self._load("SELECT ts, side, qty, price, reference_price, fee, cash_after, qty_after "
                          "FROM fills WHERE account=? ORDER BY ts, id", name, ("ts",))

    def load_trades(self, name: str) -> pd.DataFrame:
        return self._load("SELECT entry_ts, exit_ts, qty, entry_price, exit_price, fees, pnl, return_pct "
                          "FROM trades WHERE account=? ORDER BY exit_ts, id", name, ("entry_ts", "exit_ts"))

    def load_equity(self, name: str) -> pd.DataFrame:
        df = self._load("SELECT ts, cash, qty, price, equity FROM equity WHERE account=? ORDER BY ts",
                        name, ("ts",))
        return df.set_index("ts")

    def save_backtest(self, name: str, result: Any, source: str, symbol: str) -> None:
        """Persist a BacktestResult under a 'backtest' account (replacing any previous one)."""
        with self.transaction() as conn:
            self.create_account(
                name, kind="backtest", source=source, symbol=symbol, timeframe=result.timeframe,
                strategy=result.strategy.to_config(), execution=result.config,
                state=result.final_state, replace=True, conn=conn,
            )
            self.record_fills(conn, name, result.fills)
            self.record_trades(conn, name, result.trades)
            eq = result.equity
            self.record_equity(conn, name, [
                EquityPoint(ts, r.cash, r.qty, r.price, r.equity)
                for ts, r in zip(eq.index, eq.itertuples(index=False))
            ])
