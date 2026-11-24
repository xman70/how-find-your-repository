"""Incremental price downloads: only recent bars when a fresh full download is cached."""
from copy import deepcopy

import numpy as np
import pandas as pd

from vusa_quant_ai.data.market.providers import _finalize
from vusa_quant_ai.data.service import DataService
from vusa_quant_ai.database.db import Database


def _series(end, scale=1.0, n=600):
    idx = pd.bdate_range(end=end, periods=n)
    c = scale * 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, n)))
    return _finalize(pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e5}, index=idx),
                     "XETRA")


def _service(settings, tmp_path, provider):
    ds = DataService(deepcopy(settings), Database(tmp_path / "p.sqlite"))
    calls = []

    def attempts(symbol, start, exchange):
        def fetch():
            calls.append(start)
            full = provider()
            return full[full.index >= pd.Timestamp(start)]
        return [("yfinance", fetch)]

    ds._market_attempts = attempts
    return ds, calls


def test_second_run_only_fetches_recent_bars(settings, tmp_path):
    state = {"df": _series("2026-11-20")}
    ds, calls = _service(settings, tmp_path, lambda: state["df"])
    df1, r1 = ds.fetch_symbol("VUSA", "VUSA.DE", "2020-01-01", "XETRA")
    assert calls[-1] == "2020-01-01" and len(df1) == 600
    # next day: provider has one more bar
    nxt = _series("2026-11-23", n=601)
    state["df"] = nxt
    df2, r2 = ds.fetch_symbol("VUSA", "VUSA.DE", "2020-01-01", "XETRA")
    assert pd.Timestamp(calls[-1]) > pd.Timestamp("2026-10-01")  # incremental request
    assert len(df2) == 601 and df2.index[-1] == pd.Timestamp("2026-11-23")
    assert np.allclose(df2["close"].values, nxt["close"].values)


def test_readjusted_history_forces_full_refresh(settings, tmp_path):
    state = {"df": _series("2026-11-20")}
    ds, calls = _service(settings, tmp_path, lambda: state["df"])
    ds.fetch_symbol("VUSA", "VUSA.DE", "2020-01-01", "XETRA")
    state["df"] = _series("2026-11-23", scale=0.97, n=601)  # provider re-adjusted all prices by 3%
    df, _ = ds.fetch_symbol("VUSA", "VUSA.DE", "2020-01-01", "XETRA")
    assert calls[-1] == "2020-01-01"  # fell back to a full download
    assert np.allclose(df["close"].values, state["df"]["close"].values)


def test_old_cache_gets_weekly_full_refresh(settings, tmp_path):
    ds, calls = _service(settings, tmp_path, lambda: _series("2026-11-20"))
    ds.fetch_symbol("VUSA", "VUSA.DE", "2020-01-01", "XETRA")
    ds.db.execute("UPDATE fetch_meta SET last_full_refresh=? WHERE symbol=?",
                  [(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=8)).isoformat(), "VUSA.DE"])
    ds.fetch_symbol("VUSA", "VUSA.DE", "2020-01-01", "XETRA")
    assert calls[-1] == "2020-01-01"
