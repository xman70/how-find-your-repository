"""Leading-indicator dashboard.

For each candidate indicator: rank correlation with *subsequent* 20/60-day VUSA returns,
best lag, stability (share of rolling 3-year windows with the same sign), current value,
historical percentile and the direction it currently points to. Correlation is not
causation - indicators are shown as statistical associations only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INDICATORS = ["curve_10y2y", "curve_10y3m", "curve_10y_3m_mkt", "hy_oas", "hy_oas_chg_20d", "credit_risk_appetite",
              "vix_level", "vix_term_structure", "vrp", "sector_pct_above_sma200_proxy", "cyclical_vs_defensive_60d",
              "small_large_ratio_mom", "equal_cap_ratio_mom", "usd_ret_60d", "us10y_chg_60d", "real_rate_proxy",
              "unrate_vs_12m_low", "claims_chg_13w", "cpi_yoy", "inflation_trend_6m", "nfci", "fed_funds_chg_6m",
              "copper_gold", "oil_ret_60d", "mom_12_1", "dist_sma200"]


def leading_indicator_table(features: pd.DataFrame, close: pd.Series, horizons=(20, 60), lags=(0, 20, 60)) -> pd.DataFrame:
    rows = []
    fwd = {h: close.pct_change(h).shift(-h) for h in horizons}
    for ind in INDICATORS:
        if ind not in features or features[ind].notna().sum() < 500:
            continue
        x = features[ind]
        cur = x.dropna().iloc[-1]
        pct = float((x.dropna() < cur).mean())
        best = None
        for h in horizons:
            for lag in lags:
                d = pd.concat([x.shift(lag), fwd[h]], axis=1).dropna().iloc[::max(1, h // 2)]
                if len(d) < 60:
                    continue
                ic = d.iloc[:, 0].corr(d.iloc[:, 1], method="spearman")
                if best is None or abs(ic) > abs(best[0]):
                    best = (ic, h, lag)
        if best is None:
            continue
        ic, h, lag = best
        # stability: sign consistency across rolling 3y windows
        d = pd.concat([x.shift(lag), fwd[h]], axis=1).dropna()
        signs = []
        for s in range(0, len(d) - 756, 126):
            w = d.iloc[s: s + 756]
            signs.append(np.sign(w.iloc[:, 0].corr(w.iloc[:, 1], method="spearman")))
        stab = float(np.mean(np.array(signs) == np.sign(ic))) if signs else np.nan
        z = (cur - x.dropna().tail(1260).mean()) / (x.dropna().tail(1260).std() or np.nan)
        direction = np.sign(ic) * np.sign(z) if np.isfinite(z) and abs(z) > 0.25 else 0
        rows.append({"indicator": ind, "best_horizon": h, "best_lag": lag, "rank_ic": ic, "abs_ic": abs(ic),
                     "stability": stab, "current": float(cur), "percentile": pct, "z_score": float(z) if np.isfinite(z) else np.nan,
                     "signal": "bullish" if direction > 0 else ("bearish" if direction < 0 else "neutral"),
                     "reliable": bool(abs(ic) >= 0.08 and (stab or 0) >= 0.65)})
    return pd.DataFrame(rows).sort_values("abs_ic", ascending=False) if rows else pd.DataFrame()
