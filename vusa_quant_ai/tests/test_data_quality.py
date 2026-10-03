"""Data-quality, calendar and point-in-time tests."""
from datetime import date

import numpy as np
import pandas as pd

from vusa_quant_ai.data.calendar import (add_trading_days, easter_sunday, is_trading_day, market_close_utc,
                                         next_trading_day, trading_days)
from vusa_quant_ai.data.point_in_time.engine import PointInTimeStore, align_market_to_decision
from vusa_quant_ai.data.validation import validate_ohlcv


def test_easter_and_holidays():
    assert easter_sunday(2024) == date(2024, 3, 31)
    assert easter_sunday(2026) == date(2026, 4, 5)
    assert not is_trading_day("2026-04-03", "XETRA")  # Good Friday
    assert not is_trading_day("2026-04-06", "XETRA")  # Easter Monday
    assert not is_trading_day("2026-12-25", "XETRA")
    assert not is_trading_day("2026-10-03", "XETRA")  # Saturday
    assert is_trading_day("2026-10-02", "XETRA")
    assert not is_trading_day("2026-11-26", "NYSE")  # Thanksgiving
    assert not is_trading_day("2026-07-03", "NYSE")  # July 4th observed (Saturday)
    assert next_trading_day("2026-10-02", "XETRA") == pd.Timestamp("2026-10-05")
    assert add_trading_days("2026-12-23", 1, "XETRA") == pd.Timestamp("2026-12-28")


def test_market_close_timezone():
    # 17:30 Berlin in summer = 15:30 UTC; in winter = 16:30 UTC
    assert market_close_utc("2026-07-01", "XETRA").hour == 15
    assert market_close_utc("2026-01-15", "XETRA").hour == 16
    assert market_close_utc("2026-07-01", "NYSE").hour == 20


def _frame(n=400):
    idx = trading_days("2020-01-01", "2022-12-31", "XETRA")[:n]
    c = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, len(idx))))
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e5}, index=idx)


def test_validation_detects_problems():
    df = _frame()
    df.iloc[10, df.columns.get_loc("high")] = df["low"].iloc[10] * 0.5  # inconsistent OHLC
    df.iloc[50, df.columns.get_loc("close")] = -1  # invalid price
    spike = df.index[100]
    df.loc[spike, "close"] *= 2.0  # bad tick reversed next day
    out, rep = validate_ohlcv(df, "T", "XETRA")
    checks = {i["check"] for i in rep.issues}
    assert {"ohlc_consistency", "non_positive_close", "bad_tick"} <= checks
    assert spike not in out.index
    assert (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-9).all()
    assert rep.ok  # no critical issues


def test_future_rows_are_critical():
    df = _frame()
    out, rep = validate_ohlcv(df, "T", "XETRA", as_of=df.index[200])
    assert not rep.ok
    assert out.index.max() <= df.index[200]


def test_point_in_time_respects_publication_and_revisions():
    st = PointInTimeStore()
    st.add_records("CPI", pd.DataFrame({
        "effective_date": ["2024-01-01", "2024-01-01", "2024-02-01"],
        "value": [100.0, 101.0, 102.0],  # January later revised 100 -> 101
        "published_at": ["2024-02-13 13:30", "2024-03-12 13:30", "2024-03-12 13:30"]}), "vintage", "test")
    v, eff, _ = st.value_as_of("CPI", "2024-02-20")
    assert v == 100.0 and eff == pd.Timestamp("2024-01-01")
    v, eff, _ = st.value_as_of("CPI", "2024-02-13 13:00")  # before first release
    assert np.isnan(v)
    h = st.history_as_of("CPI", "2024-03-13")
    assert h.loc["2024-01-01"] == 101.0 and h.loc["2024-02-01"] == 102.0
    dv = st.daily_view("CPI", pd.DatetimeIndex(["2024-02-12", "2024-02-14", "2024-03-13"], tz="UTC"))
    assert np.isnan(dv.iloc[0]) and dv.iloc[1] == 100.0 and dv.iloc[2] == 102.0


def test_release_lag_approximation():
    st = PointInTimeStore()
    s = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3, freq="MS"))
    st.add_latest_with_lag("X", s, 15, "test")
    r = st.records("X")
    assert (r["published_at"].dt.tz_convert(None) > (r["effective_date"] + pd.offsets.MonthEnd(0))).all()
    assert st.method("X") == "release_lag"


def test_us_close_not_visible_to_same_day_european_decision():
    us = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2026-07-01", "2026-07-02"]))
    us["available_at"] = [market_close_utc(d, "NYSE").isoformat() for d in us.index]
    dec = pd.DatetimeIndex([market_close_utc("2026-07-02", "XETRA") + pd.Timedelta(minutes=60)])
    m = align_market_to_decision(us, dec)
    assert m["close"].iloc[0] == 1.0  # 2 July US close (20:00 UTC) is not yet known at 16:30 UTC


def test_synthetic_bundle_is_labelled(bundle):
    assert bundle.is_synthetic
    assert "SYNTHETIC" in bundle.status_label
    assert not bundle.data_current


def test_pre_inception_proxy_is_labelled_and_point_in_time(settings, bundle):
    from copy import deepcopy

    from vusa_quant_ai.data.service import DataService, MarketDataBundle
    from vusa_quant_ai.features.builder import build_features

    b = MarketDataBundle(pd.DataFrame(), "VUSA.DE", assets={k: bundle.assets[k] for k in ("SPX", "EURUSD")})
    tgt = bundle.target.iloc[-900:].drop(columns=["is_proxy"])
    ext = DataService(deepcopy(settings))._extend_with_proxy(tgt, b)
    assert b.proxy_used and "proxy" in b.proxy_note
    assert ext["is_proxy"].iloc[0] == 1 and ext["is_proxy"].iloc[-1] == 0
    first_real = tgt.index[0]
    # level continuity at the splice point (scaled to VUSA)
    pre = ext[ext["is_proxy"] == 1]["close"].iloc[-1]
    assert abs(pre / ext.loc[first_real, "close"] - 1) < 0.1
    b.target = ext
    fs = build_features(b)
    proxy_rows = ext["is_proxy"].values.astype(bool)
    assert (fs.decision_times[proxy_rows].hour >= 20).all()  # NYSE close (+60 min) for S&P proxy rows
    assert (fs.decision_times[~proxy_rows].hour <= 17).all()  # XETRA close for real VUSA rows
