"""Persistence layer.

SQLite is used initially. The schema uses portable SQL types (TEXT, REAL, INTEGER) and
avoids SQLite-only features so it can be migrated to PostgreSQL by swapping the
connection factory (``Database.connect``) - placeholders are converted from ``?`` to
``%s`` when a PostgreSQL URL is configured.

Large structured payloads (decision traces, agent outputs, model metadata) are stored
as JSON text so the decision state of every signal can be reconstructed exactly.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..core.runtime import utcnow_iso

SCHEMA = [
    # ---------------------------------------------------------------- market data
    """CREATE TABLE IF NOT EXISTS prices (
        symbol TEXT NOT NULL, date TEXT NOT NULL, open REAL, high REAL, low REAL, close REAL,
        adj_close REAL, volume REAL, source TEXT, retrieved_at TEXT, available_at TEXT,
        PRIMARY KEY (symbol, date, source))""",
    """CREATE TABLE IF NOT EXISTS adjustments (
        symbol TEXT NOT NULL, date TEXT NOT NULL, kind TEXT NOT NULL, value REAL, source TEXT,
        retrieved_at TEXT, PRIMARY KEY (symbol, date, kind))""",
    # ---------------------------------------------------------------- macro (point-in-time)
    """CREATE TABLE IF NOT EXISTS macro_series (
        series_id TEXT PRIMARY KEY, title TEXT, frequency TEXT, units TEXT, source TEXT,
        publication_lag_days INTEGER, notes TEXT, updated_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS macro_releases (
        series_id TEXT NOT NULL, effective_date TEXT NOT NULL, value REAL,
        published_at TEXT NOT NULL, vintage TEXT, pit_method TEXT, source TEXT, retrieved_at TEXT,
        PRIMARY KEY (series_id, effective_date, published_at))""",
    # ---------------------------------------------------------------- news
    """CREATE TABLE IF NOT EXISTS news_articles (
        article_id TEXT PRIMARY KEY, title TEXT, summary TEXT, url TEXT, source TEXT,
        published_at TEXT, retrieved_at TEXT, relevance REAL, source_quality REAL,
        sentiment REAL, label TEXT, topics TEXT, duplicate_of TEXT)""",
    """CREATE TABLE IF NOT EXISTS news_events (
        event_id TEXT PRIMARY KEY, article_id TEXT, event_date TEXT, category TEXT, description TEXT,
        direction TEXT, magnitude REAL, confidence REAL, horizon TEXT, available_at TEXT, payload TEXT)""",
    """CREATE TABLE IF NOT EXISTS sentiment (
        date TEXT NOT NULL, scope TEXT NOT NULL, score REAL, dispersion REAL, n_articles INTEGER,
        available_at TEXT, PRIMARY KEY (date, scope))""",
    # ---------------------------------------------------------------- features / regimes
    """CREATE TABLE IF NOT EXISTS features (
        run_id TEXT NOT NULL, date TEXT NOT NULL, feature TEXT NOT NULL, value REAL,
        PRIMARY KEY (run_id, date, feature))""",
    """CREATE TABLE IF NOT EXISTS feature_lineage (
        run_id TEXT NOT NULL, feature TEXT NOT NULL, source TEXT, calculated_at TEXT,
        transformation TEXT, original_value REAL, normalized_value REAL, missing_status TEXT,
        PRIMARY KEY (run_id, feature))""",
    """CREATE TABLE IF NOT EXISTS regimes (
        date TEXT NOT NULL, method TEXT NOT NULL, regime TEXT, probability REAL, payload TEXT,
        PRIMARY KEY (date, method))""",
    # ---------------------------------------------------------------- models
    """CREATE TABLE IF NOT EXISTS models (
        model_name TEXT PRIMARY KEY, family TEXT, description TEXT, status TEXT, created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS model_versions (
        model_id TEXT PRIMARY KEY, model_name TEXT, horizon INTEGER, train_start TEXT, train_end TEXT,
        feature_version TEXT, dataset_version TEXT, hyperparameters TEXT, validation TEXT,
        artifact_path TEXT, role TEXT, status TEXT, created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS predictions (
        prediction_id TEXT PRIMARY KEY, run_id TEXT, as_of TEXT, horizon INTEGER, model_id TEXT,
        price REAL, expected_return REAL, p_up REAL, p_down REAL, p_gt5 REAL, p_lt5 REAL,
        p_dd10 REAL, confidence REAL, signal TEXT, target_date TEXT, actual_return REAL,
        error REAL, resolved_at TEXT, regime TEXT)""",
    """CREATE TABLE IF NOT EXISTS prediction_intervals (
        prediction_id TEXT NOT NULL, level REAL NOT NULL, lower REAL, upper REAL, method TEXT,
        PRIMARY KEY (prediction_id, level))""",
    # ---------------------------------------------------------------- signals
    """CREATE TABLE IF NOT EXISTS signals (
        run_id TEXT PRIMARY KEY, as_of TEXT, signal TEXT, strength TEXT, opportunity REAL, risk REAL,
        confidence REAL, agreement REAL, final_score REAL, previous_signal TEXT, duration_days INTEGER,
        change_reason TEXT, data_current INTEGER, created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS signal_components (
        run_id TEXT NOT NULL, component TEXT NOT NULL, score REAL, weight REAL, contribution REAL,
        evidence TEXT, PRIMARY KEY (run_id, component))""",
    # ---------------------------------------------------------------- backtests
    """CREATE TABLE IF NOT EXISTS backtests (
        backtest_id TEXT PRIMARY KEY, created_at TEXT, config TEXT, metrics TEXT, leakage_audit TEXT,
        equity_curve TEXT, status TEXT)""",
    """CREATE TABLE IF NOT EXISTS trades (
        backtest_id TEXT NOT NULL, trade_no INTEGER NOT NULL, signal_date TEXT, exec_date TEXT, side TEXT,
        price REAL, units REAL, cost REAL, PRIMARY KEY (backtest_id, trade_no))""",
    """CREATE TABLE IF NOT EXISTS performance_metrics (
        scope TEXT NOT NULL, horizon INTEGER NOT NULL, regime TEXT NOT NULL, metric TEXT NOT NULL,
        value REAL, n INTEGER, updated_at TEXT, PRIMARY KEY (scope, horizon, regime, metric))""",
    # ---------------------------------------------------------------- monitoring
    """CREATE TABLE IF NOT EXISTS model_drift (
        checked_at TEXT NOT NULL, model_id TEXT NOT NULL, metric TEXT, historical REAL, recent REAL,
        drift INTEGER, payload TEXT, PRIMARY KEY (checked_at, model_id))""",
    """CREATE TABLE IF NOT EXISTS feature_drift (
        checked_at TEXT NOT NULL, feature TEXT NOT NULL, psi REAL, ks_stat REAL, ks_p REAL,
        out_of_range INTEGER, PRIMARY KEY (checked_at, feature))""",
    """CREATE TABLE IF NOT EXISTS data_quality (
        checked_at TEXT NOT NULL, dataset TEXT NOT NULL, source TEXT, status TEXT, latency_ms REAL,
        rows INTEGER, last_date TEXT, issues TEXT, PRIMARY KEY (checked_at, dataset, source))""",
    """CREATE TABLE IF NOT EXISTS alerts (
        alert_id TEXT PRIMARY KEY, created_at TEXT, kind TEXT, severity TEXT, message TEXT,
        payload TEXT, acknowledged INTEGER DEFAULT 0)""",
    # ---------------------------------------------------------------- research / reporting
    """CREATE TABLE IF NOT EXISTS daily_reports (
        run_id TEXT PRIMARY KEY, as_of TEXT, markdown TEXT, html TEXT, created_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS research_runs (
        run_id TEXT PRIMARY KEY, kind TEXT, started_at TEXT, finished_at TEXT, status TEXT,
        mode TEXT, steps TEXT, environment TEXT, summary TEXT)""",
    """CREATE TABLE IF NOT EXISTS agent_outputs (
        run_id TEXT NOT NULL, agent TEXT NOT NULL, stance TEXT, score REAL, payload TEXT,
        PRIMARY KEY (run_id, agent))""",
    """CREATE TABLE IF NOT EXISTS decision_trace (
        run_id TEXT PRIMARY KEY, as_of TEXT, signal TEXT, trace TEXT, created_at TEXT)""",
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS ix_prices_symbol_date ON prices(symbol, date)",
    "CREATE INDEX IF NOT EXISTS ix_macro_pub ON macro_releases(series_id, published_at)",
    "CREATE INDEX IF NOT EXISTS ix_pred_asof ON predictions(as_of, horizon)",
    "CREATE INDEX IF NOT EXISTS ix_news_pub ON news_articles(published_at)",
]


class Database:
    """Thin, thread-safe database wrapper."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path not in (None, ":memory:") else path
        if isinstance(self.path, Path):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path or ":memory:"), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL" if isinstance(self.path, Path) else "PRAGMA journal_mode=MEMORY")
        self.init_schema()

    def init_schema(self) -> None:
        with self._lock:
            for stmt in SCHEMA + INDEXES:
                self._conn.execute(stmt)
            self._conn.commit()

    # ------------------------------------------------------------------ primitives
    def execute(self, sql: str, params: Iterable[Any] = ()) -> None:
        with self._lock:
            self._conn.execute(sql, tuple(params))
            self._conn.commit()

    def executemany(self, sql: str, rows: list[tuple]) -> None:
        if not rows:
            return
        with self._lock:
            self._conn.executemany(sql, rows)
            self._conn.commit()

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return [dict(r) for r in cur.fetchall()]

    def query_df(self, sql: str, params: Iterable[Any] = ()) -> pd.DataFrame:
        rows = self.query(sql, params)
        return pd.DataFrame(rows)

    def upsert(self, table: str, row: dict[str, Any]) -> None:
        cols = list(row.keys())
        vals = [json.dumps(v, default=str) if isinstance(v, (dict, list)) else v for v in row.values()]
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        self.execute(sql, vals)

    def upsert_many(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        cols = list(rows[0].keys())
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})"
        data = [tuple(json.dumps(r[c], default=str) if isinstance(r[c], (dict, list)) else r[c] for c in cols)
                for r in rows]
        self.executemany(sql, data)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ domain helpers
    def save_prices(self, symbol: str, df: pd.DataFrame, source: str) -> None:
        if df is None or df.empty:
            return
        now = utcnow_iso()
        rows = []
        for idx, r in df.iterrows():
            rows.append((symbol, pd.Timestamp(idx).strftime("%Y-%m-%d"), _f(r.get("open")), _f(r.get("high")),
                         _f(r.get("low")), _f(r.get("close")), _f(r.get("adj_close", r.get("close"))),
                         _f(r.get("volume")), source, now, _s(r.get("available_at"))))
        self.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)

    def load_prices(self, symbol: str, source: str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM prices WHERE symbol=?"
        params: list[Any] = [symbol]
        if source:
            sql += " AND source=?"
            params.append(source)
        df = self.query_df(sql + " ORDER BY date", params)
        if df.empty:
            return df
        # if multiple sources exist keep the most recently retrieved row per date
        df = df.sort_values(["date", "retrieved_at"]).drop_duplicates("date", keep="last")
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def save_trace(self, run_id: str, as_of: str, signal: str, trace: dict) -> None:
        self.upsert("decision_trace", {"run_id": run_id, "as_of": as_of, "signal": signal,
                                       "trace": json.dumps(trace, default=str), "created_at": utcnow_iso()})

    def load_trace(self, run_id: str) -> dict | None:
        rows = self.query("SELECT trace FROM decision_trace WHERE run_id=?", [run_id])
        return json.loads(rows[0]["trace"]) if rows else None

    def trace_for_date(self, date: str) -> dict | None:
        rows = self.query("SELECT trace FROM decision_trace WHERE as_of LIKE ? ORDER BY created_at DESC",
                          [f"{date}%"])
        return json.loads(rows[0]["trace"]) if rows else None


def _f(v):
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _s(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)
