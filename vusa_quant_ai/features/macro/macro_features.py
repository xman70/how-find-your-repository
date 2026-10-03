"""Macro features computed strictly from the point-in-time store."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ...data.point_in_time.engine import PointInTimeStore

# feature -> (series, transform, periods, description)
MACRO_FEATURES = {
    "cpi_yoy": ("CPIAUCSL", "yoy", 12, "CPI YoY inflation (vintage known at t)"),
    "core_cpi_yoy": ("CPILFESL", "yoy", 12, "Core CPI YoY"),
    "core_pce_yoy": ("PCEPILFE", "yoy", 12, "Core PCE YoY"),
    "unrate": ("UNRATE", "level", 0, "Unemployment rate"),
    "unrate_chg_12m": ("UNRATE", "diff", 12, "Unemployment change 12m"),
    "payrolls_yoy": ("PAYEMS", "yoy", 12, "Payroll growth YoY"),
    "claims_level": ("ICSA", "level", 0, "Initial claims"),
    "claims_chg_13w": ("ICSA", "yoy", 13, "Initial claims change over 13 weeks"),
    "indpro_yoy": ("INDPRO", "yoy", 12, "Industrial production YoY"),
    "gdp_yoy": ("GDPC1", "yoy", 4, "Real GDP YoY"),
    "sentiment_umich": ("UMCSENT", "level", 0, "UMich consumer sentiment"),
    "ust10y": ("DGS10", "level", 0, "10Y Treasury yield"),
    "ust2y": ("DGS2", "level", 0, "2Y Treasury yield"),
    "curve_10y2y": ("T10Y2Y", "level", 0, "10Y-2Y spread"),
    "curve_10y3m": ("T10Y3M", "level", 0, "10Y-3M spread"),
    "real_yield_10y": ("DFII10", "level", 0, "10Y real yield (TIPS)"),
    "breakeven_10y": ("T10YIE", "level", 0, "10Y breakeven inflation"),
    "fed_funds": ("DFF", "level", 0, "Effective fed funds rate"),
    "hy_oas": ("BAMLH0A0HYM2", "level", 0, "High-yield OAS"),
    "ig_oas": ("BAMLC0A0CM", "level", 0, "Investment-grade OAS"),
    "nfci": ("NFCI", "level", 0, "Chicago Fed NFCI"),
    "fed_assets_yoy": ("WALCL", "yoy", 52, "Fed balance sheet YoY (liquidity proxy)"),
    "m2_yoy": ("M2SL", "yoy", 12, "M2 YoY"),
    "sahm_rt": ("SAHMREALTIME", "level", 0, "Sahm rule real-time"),
}


def macro_features(pit: PointInTimeStore, decision_times: pd.DatetimeIndex, index: pd.DatetimeIndex
                   ) -> tuple[pd.DataFrame, dict[str, str]]:
    f, src = {}, {}
    for name, (sid, tr, per, desc) in MACRO_FEATURES.items():
        if sid not in pit.series_ids:
            continue
        s = pit.daily_view(sid, decision_times, tr, max(per, 1))
        s.index = index
        if s.notna().sum() < 50:
            continue
        f[name] = s
        src[name] = f"{sid} [{pit.method(sid)}] - {desc}"
    df = pd.DataFrame(f, index=index)
    # derived: trends and real-rate proxy
    if "cpi_yoy" in df:
        df["inflation_trend_6m"] = df["cpi_yoy"] - df["cpi_yoy"].shift(126)
        src["inflation_trend_6m"] = "CPIAUCSL derived"
    if "ust10y" in df:
        df["ust10y_chg_60d"] = df["ust10y"] - df["ust10y"].shift(60)
        src["ust10y_chg_60d"] = "DGS10 derived"
        if "cpi_yoy" in df:
            df["real_rate_proxy"] = df["ust10y"] - 100 * df["cpi_yoy"]
            src["real_rate_proxy"] = "DGS10 - CPI YoY"
    if "hy_oas" in df:
        df["hy_oas_chg_20d"] = df["hy_oas"] - df["hy_oas"].shift(20)
        src["hy_oas_chg_20d"] = "BAMLH0A0HYM2 derived"
    if "unrate" in df:
        df["unrate_vs_12m_low"] = df["unrate"] - df["unrate"].rolling(252, min_periods=100).min()
        src["unrate_vs_12m_low"] = "UNRATE derived (Sahm-style)"
    if "fed_funds" in df:
        df["fed_funds_chg_6m"] = df["fed_funds"] - df["fed_funds"].shift(126)
        src["fed_funds_chg_6m"] = "DFF derived"
    for sid in ("CPIAUCSL", "UNRATE", "DGS10"):
        if sid in pit.series_ids:
            st = pit.staleness_days(sid, decision_times)
            st.index = index
            df[f"staleness_{sid.lower()}"] = st
            src[f"staleness_{sid.lower()}"] = f"{sid} days since publication"
    return df.replace([np.inf, -np.inf], np.nan), src
