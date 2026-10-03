"""Shock / anomaly detection. Every detector is fitted on history *before* the evaluated
day. Detected shocks reduce signal confidence (see signals.decision)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def robust_z(s: pd.Series, window: int = 252) -> pd.Series:
    med = s.rolling(window, min_periods=60).median().shift(1)
    mad = (s - med).abs().rolling(window, min_periods=60).median().shift(1) * 1.4826
    return (s - med) / mad.replace(0, np.nan)


def isolation_forest_score(F: pd.DataFrame, train_end_pos: int | None = None, seed: int = 0) -> pd.Series:
    from sklearn.ensemble import IsolationForest

    F = F.dropna()
    if len(F) < 400:
        return pd.Series(np.nan, index=F.index)
    cut = train_end_pos or len(F) - 1
    mu, sd = F.iloc[:cut].mean(), F.iloc[:cut].std().replace(0, 1)
    Z = ((F - mu) / sd).clip(-10, 10)
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=seed).fit(Z.iloc[:cut])
    score = -iso.score_samples(Z)  # higher = more anomalous
    ref = -iso.score_samples(Z.iloc[:cut])
    pct = np.searchsorted(np.sort(ref), score) / len(ref)
    return pd.Series(pct, index=F.index)


def autoencoder_score(F: pd.DataFrame, train_end_pos: int | None = None, seed: int = 0) -> pd.Series:
    """Bottleneck MLP autoencoder; reconstruction-error percentile vs training errors."""
    from sklearn.neural_network import MLPRegressor

    F = F.dropna()
    if len(F) < 400:
        return pd.Series(np.nan, index=F.index)
    cut = train_end_pos or len(F) - 1
    mu, sd = F.iloc[:cut].mean(), F.iloc[:cut].std().replace(0, 1)
    Z = ((F - mu) / sd).clip(-10, 10).values
    k = max(2, Z.shape[1] // 3)
    ae = MLPRegressor(hidden_layer_sizes=(16, k, 16), max_iter=300, random_state=seed, early_stopping=True)
    ae.fit(Z[:cut], Z[:cut])
    err = ((ae.predict(Z) - Z) ** 2).mean(axis=1)
    ref = np.sort(err[:cut])
    return pd.Series(np.searchsorted(ref, err) / len(ref), index=F.index)


def detect_shocks(target: pd.DataFrame, features: pd.DataFrame, regime: dict | None = None,
                  change: dict | None = None, news_shock: bool = False, macro_shock: bool = False) -> dict:
    c = target["close"]
    r = np.log(c).diff()
    out: dict = {"flags": [], "details": {}}
    rz = robust_z(r).iloc[-1]
    out["details"]["return_robust_z"] = float(rz) if pd.notna(rz) else None
    if pd.notna(rz) and abs(rz) > 4:
        out["flags"].append(f"Abnormal daily return (robust z={rz:.1f})")
    if "rv_5" in features and "rv_60" in features:
        ratio = (features["rv_5"] / features["rv_60"]).iloc[-1]
        out["details"]["vol_ratio_5_60"] = float(ratio) if pd.notna(ratio) else None
        if pd.notna(ratio) and ratio > 2.0:
            out["flags"].append(f"Volatility explosion (5d/60d realised vol = {ratio:.1f}x)")
    if "volume" in target and target["volume"].notna().sum() > 300 and (target["volume"].fillna(0) > 0).mean() > 0.5:
        vz = robust_z(np.log1p(target["volume"].astype(float))).iloc[-1]
        out["details"]["volume_robust_z"] = float(vz) if pd.notna(vz) else None
        if pd.notna(vz) and vz > 4:
            out["flags"].append(f"Abnormal volume (robust z={vz:.1f})")
    if "gap" in features:
        gz = robust_z(features["gap"]).iloc[-1]
        if pd.notna(gz) and abs(gz) > 4:
            out["flags"].append(f"Large overnight gap (robust z={gz:.1f})")
    for col in ("corr_bonds_chg", "corr_usd_chg"):
        if col in features and pd.notna(features[col].iloc[-1]) and abs(features[col].iloc[-1]) > 0.6:
            out["flags"].append(f"Unusual correlation shift ({col}={features[col].iloc[-1]:+.2f})")
    cols = [c for c in ("logret_1d", "rv_5", "rv_20", "hl_range", "gap", "vix_level", "vix_chg_5d", "us10y_chg_20d",
                        "credit_risk_appetite", "sector_dispersion_20d") if c in features]
    F = features[cols].tail(2520)
    iso = isolation_forest_score(F)
    ae = autoencoder_score(F)
    out["details"]["isolation_forest_pct"] = float(iso.iloc[-1]) if len(iso) and pd.notna(iso.iloc[-1]) else None
    out["details"]["autoencoder_pct"] = float(ae.iloc[-1]) if len(ae) and pd.notna(ae.iloc[-1]) else None
    if out["details"]["isolation_forest_pct"] and out["details"]["isolation_forest_pct"] > 0.99:
        out["flags"].append("Multivariate anomaly (Isolation Forest > 99th pct)")
    if out["details"]["autoencoder_pct"] and out["details"]["autoencoder_pct"] > 0.99:
        out["flags"].append("Multivariate anomaly (autoencoder reconstruction > 99th pct)")
    if change and change.get("structural_change_detected"):
        out["flags"].append("Structural change detected (change-point tests)")
    if regime and regime.get("days_in_regime", 99) <= 3 and regime.get("previous_regime"):
        out["flags"].append(f"Regime transition: {regime['previous_regime']} -> {regime['regime']}")
    if news_shock:
        out["flags"].append("News shock (high-magnitude negative events from credible sources)")
    if macro_shock:
        out["flags"].append("Macro shock (large move in rates / credit)")
    out["shock_level"] = min(1.0, 0.2 * len(out["flags"]))
    out["series"] = {"isolation_forest": iso, "autoencoder": ae}
    return out
