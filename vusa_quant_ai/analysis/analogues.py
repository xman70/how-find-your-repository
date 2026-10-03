"""Historical analogue search.

Today's market state is a standardised vector of interpretable state variables. Nearest
historical states are found with standardised Euclidean, Mahalanobis and cosine
distances (consensus rank). Only analogues whose subsequent outcome window had fully
ended before today are used, and analogues are de-clustered (min 20 sessions apart) so
one episode cannot dominate. Outcomes are descriptive - never a guarantee.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATE_VARS = ["ret_20d", "ret_60d", "ret_252d", "dist_sma200", "sma50_sma200", "rv_20", "rv_ratio_20_252",
              "drawdown_252", "rsi14", "vix_level", "vix_term_structure", "us10y_chg_60d", "curve_10y_3m_mkt",
              "credit_risk_appetite", "sector_pct_above_sma200_proxy", "usd_ret_60d"]


def find_analogues(features: pd.DataFrame, close: pd.Series, regimes: pd.Series | None = None, k: int = 10,
                   min_gap: int = 20, max_horizon: int = 60, as_of=None) -> dict:
    as_of = pd.Timestamp(as_of or features.index[-1])
    cols = [c for c in STATE_VARS if c in features.columns and features[c].notna().mean() > 0.7]
    F = features[cols].loc[:as_of].astype(float)
    today = F.iloc[-1]
    cols = [c for c in cols if pd.notna(today[c])]
    F = F[cols].dropna()
    if len(F) < 500 or len(cols) < 4:
        return {"analogues": pd.DataFrame(), "summary": {}, "variables": cols,
                "note": "insufficient history / state variables for analogue search"}
    today = F.iloc[-1]
    # candidate rows: outcome window (max_horizon) must have ended before as-of
    pos_asof = close.index.get_loc(F.index[-1])
    cand = F.iloc[: max(0, len(F) - max_horizon - 1)]
    cand = cand[close.index.get_indexer(cand.index) + max_horizon < pos_asof]
    mu, sd = cand.mean(), cand.std().replace(0, 1)
    Z, z0 = (cand - mu) / sd, (today - mu) / sd
    d_euc = np.sqrt(((Z - z0) ** 2).sum(axis=1))
    try:
        VI = np.linalg.pinv(np.cov(Z.values.T))
        diff = (Z - z0).values
        d_mah = pd.Series(np.sqrt(np.einsum("ij,jk,ik->i", diff, VI, diff).clip(0)), index=Z.index)
    except Exception:
        d_mah = d_euc
    cos = (Z.values @ z0.values) / (np.linalg.norm(Z.values, axis=1) * np.linalg.norm(z0.values) + 1e-12)
    d_cos = pd.Series(1 - cos, index=Z.index)
    rank = d_euc.rank() + d_mah.rank() + d_cos.rank()
    chosen: list[pd.Timestamp] = []
    for dt in rank.sort_values().index:
        p = close.index.get_loc(dt)
        if all(abs(p - close.index.get_loc(c)) >= min_gap for c in chosen):
            chosen.append(dt)
        if len(chosen) >= k:
            break
    rows = []
    for dt in chosen:
        p = close.index.get_loc(dt)
        c0 = close.iloc[p]
        fwd = {h: float(close.iloc[p + h] / c0 - 1) for h in (5, 20, 60) if p + h < len(close)}
        path = close.iloc[p: p + 61]
        mdd = float((path / path.cummax() - 1).min())
        sim = float(1 / (1 + d_euc[dt] / np.sqrt(len(cols))))
        rows.append({"date": dt.date(), "similarity": round(sim, 3), "cosine": round(float(1 - d_cos[dt]), 3),
                     "regime": regimes.get(dt) if regimes is not None else None,
                     "ret_5d": fwd.get(5), "ret_20d": fwd.get(20), "ret_60d": fwd.get(60), "max_dd_60d": mdd})
    A = pd.DataFrame(rows)
    w = A["similarity"] / A["similarity"].sum()
    summ = {}
    for c in ("ret_5d", "ret_20d", "ret_60d", "max_dd_60d"):
        v = A[c].astype(float)
        summ[c] = {"weighted_mean": float((v * w).sum()), "median": float(v.median()),
                   "p_positive": float((v > 0).mean()) if c != "max_dd_60d" else None, "min": float(v.min()),
                   "max": float(v.max())}
    # unconditional base rates for comparison (analogues only matter if they differ)
    base = {f"base_{h}d_mean": float(close.pct_change(h).shift(-h).loc[:cand.index[-1]].mean()) for h in (5, 20, 60)}
    base.update({f"base_{h}d_p_pos": float((close.pct_change(h).shift(-h).loc[:cand.index[-1]] > 0).mean()) for h in (5, 20, 60)})
    return {"analogues": A, "summary": summ, "base_rates": base, "variables": cols, "today_state": today.to_dict()}
