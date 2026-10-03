"""Cross-asset features. All reference series are first aligned point-in-time onto the
VUSA decision timestamps (a US close at 22:00 CET is NOT visible to a 17:30 CET decision
on the same calendar day)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..technical.indicators import realized_vol, rolling_percentile, rolling_zscore


def aligned_close(assets: dict[str, pd.DataFrame], key: str, decision_times: pd.DatetimeIndex,
                  index: pd.DatetimeIndex) -> pd.Series:
    from ...data.point_in_time.engine import align_market_to_decision

    df = assets.get(key)
    if df is None or df.empty:
        return pd.Series(np.nan, index=index)
    m = align_market_to_decision(df, decision_times)
    s = pd.Series(m["close"].values, index=index)
    # staleness guard: drop values older than 7 calendar days at decision time
    src = pd.to_datetime(m["source_date"].values)
    age = (index - src).days
    return s.where(age <= 7)


def cross_asset_features(assets: dict[str, pd.DataFrame], target_close: pd.Series, decision_times: pd.DatetimeIndex
                         ) -> tuple[pd.DataFrame, dict[str, str]]:
    idx = target_close.index
    get = {k: aligned_close(assets, k, decision_times, idx) for k in assets}
    f, src = {}, {}

    def add(name, series, source):
        f[name] = series
        src[name] = source

    tr = np.log(target_close).diff()
    if "VIX" in get:
        vix = get["VIX"]
        add("vix_level", vix, "VIX")
        add("vix_z_252", rolling_zscore(vix, 252, 60), "VIX")
        add("vix_20d_zscore_5y", rolling_zscore(vix.rolling(20).mean(), 1260, 250), "VIX")
        add("vix_pct_5y", rolling_percentile(vix, 1260, 250), "VIX")
        add("vix_chg_5d", vix.pct_change(5), "VIX")
        add("vrp", vix / 100 - realized_vol(tr, 20), "VIX, VUSA")  # implied minus realised
        if "VIX3M" in get:
            add("vix_term_structure", get["VIX3M"] / vix - 1, "VIX, VIX3M")
    for key, nm in (("SPX", "spx"), ("NDX", "ndx"), ("DJI", "dji"), ("RUT", "rut")):
        if key in get:
            s = get[key]
            add(f"{nm}_ret_5d", s.pct_change(5), key)
            add(f"{nm}_ret_20d", s.pct_change(20), key)
            add(f"{nm}_dist_sma200", s / s.rolling(200).mean() - 1, key)
    if "NDX" in get and "SPX" in get:
        ratio = get["NDX"] / get["SPX"]
        add("ndx_spx_ratio_mom", ratio.pct_change(60), "NDX, SPX")
    if "RUT" in get and "SPX" in get:
        ratio = get["RUT"] / get["SPX"]
        add("small_large_ratio_mom", ratio.pct_change(60), "RUT, SPX")
    if "RSP" in get and "SPX" in get:
        ratio = get["RSP"] / get["SPX"]
        add("equal_cap_ratio_mom", ratio.pct_change(60), "RSP, SPX")
    if "TNX" in get:
        y10 = get["TNX"] / 10.0
        add("us10y_level", y10, "TNX")
        add("us10y_chg_20d", y10.diff(20), "TNX")
        add("us10y_chg_60d", y10.diff(60), "TNX")
        if "IRX" in get:
            add("curve_10y_3m_mkt", y10 - get["IRX"] / 10.0, "TNX, IRX")
    if "HYG" in get and "LQD" in get:
        add("credit_risk_appetite", (get["HYG"] / get["LQD"]).pct_change(20), "HYG, LQD")
    if "HYG" in get and "TLT" in get:
        add("hy_vs_treasury", (get["HYG"] / get["TLT"]).pct_change(20), "HYG, TLT")
    for key, nm in (("DXY", "usd"), ("EURUSD", "eurusd"), ("GOLD", "gold"), ("OIL", "oil"), ("TLT", "tlt")):
        if key in get:
            add(f"{nm}_ret_20d", get[key].pct_change(20), key)
            add(f"{nm}_ret_60d", get[key].pct_change(60), key)
    if "EURUSD" in get:
        add("eurusd_vol_20", realized_vol(np.log(get["EURUSD"]).diff(), 20), "EURUSD")
    # rolling correlations (regime-transition information)
    for key, nm in (("TLT", "bonds"), ("GOLD", "gold"), ("DXY", "usd"), ("VIX", "vix")):
        if key in get:
            corr = tr.rolling(60, min_periods=40).corr(np.log(get[key]).diff())
            add(f"corr_{nm}_60", corr, f"VUSA, {key}")
            add(f"corr_{nm}_chg", corr - corr.shift(60), f"VUSA, {key}")
    out = pd.DataFrame(f, index=idx).replace([np.inf, -np.inf], np.nan)
    return out, src
