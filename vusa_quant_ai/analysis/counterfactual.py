"""Counterfactual / scenario analysis by feature perturbation and model re-evaluation.

"What if VIX were 30?", "What if rates rose 0.5%?", "What if momentum turned negative?"
The fitted production models are re-evaluated on a perturbed copy of today's feature
vector. Correlated features are moved together via simple linked rules so scenarios
stay internally consistent. Results are SCENARIOS, not predictions; they only show the
model's sensitivity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIOS = {
    "Volatility 20% instead of current": {"set": {"rv_20": 0.20, "rv_10": 0.20, "rv_5": 0.20}},
    "VIX spikes +10 points": {"add": {"vix_level": 10.0, "vix_chg_5d": 0.5}, "mult": {"vix_term_structure": 0.0}},
    "Rates +0.5% (10Y)": {"add": {"us10y_level": 0.5, "us10y_chg_20d": 0.5, "us10y_chg_60d": 0.5, "ust10y": 0.5,
                                  "ust10y_chg_60d": 0.5, "real_rate_proxy": 0.5}},
    "Momentum turns negative": {"set": {"ret_20d": -0.04, "ret_60d": -0.06, "roc_20": -0.04, "roc_60": -0.06,
                                        "macd_hist_norm": -0.004, "rsi14": 38.0}},
    "Price falls below SMA200": {"set": {"dist_sma200": -0.03, "dist_sma100": -0.05, "dist_sma50": -0.06}},
    "Credit spreads widen": {"add": {"hy_oas": 1.5, "hy_oas_chg_20d": 1.5}, "set": {"credit_risk_appetite": -0.03}},
    "Breadth deteriorates": {"set": {"sector_pct_above_sma200_proxy": 0.3, "sector_pct_above_sma50_proxy": 0.2}},
}


def perturb(x: pd.DataFrame, spec: dict) -> pd.DataFrame:
    y = x.copy()
    for k, v in spec.get("set", {}).items():
        if k in y:
            y[k] = v
    for k, v in spec.get("add", {}).items():
        if k in y:
            y[k] = y[k] + v
    for k, v in spec.get("mult", {}).items():
        if k in y:
            y[k] = y[k] * v
    return y


def run_counterfactuals(final_models: dict, x_now: pd.DataFrame, weights: dict, scenarios: dict | None = None) -> pd.DataFrame:
    """Re-evaluate cross-sectional (feature-based) models on perturbed inputs.

    Context-based models (ARIMA/ETS/sequence models) read the price history directly and
    cannot be perturbed this way; they are held at their base forecast and listed."""
    scenarios = scenarios or SCENARIOS
    feat_models = {n: m for n, m in final_models.items() if not getattr(m, "needs_context", False)}
    if not feat_models:
        return pd.DataFrame()

    def ens(X):
        num = den = 0.0
        p_num = 0.0
        for n, m in feat_models.items():
            cols = getattr(getattr(m, "pre_", None), "cols_", None)
            if cols is None:
                continue
            Xi = X.reindex(columns=cols)
            p = m.predict(Xi)
            w = abs(weights.get(n, 1.0)) or 1e-6
            num += w * float(p["ret"].iloc[0])
            p_num += w * float(p["p_up"].iloc[0])
            den += w
        return (num / den, p_num / den) if den else (np.nan, np.nan)

    base_r, base_p = ens(x_now)
    rows = [{"scenario": "Base (current inputs)", "expected_return": base_r, "p_up": base_p, "delta_return": 0.0,
             "delta_p_up": 0.0, "features_changed": ""}]
    for name, spec in scenarios.items():
        xp = perturb(x_now, spec)
        changed = [k for k in list(spec.get("set", {})) + list(spec.get("add", {})) + list(spec.get("mult", {})) if k in x_now]
        if not changed:
            continue
        r, p = ens(xp)
        rows.append({"scenario": name, "expected_return": r, "p_up": p, "delta_return": r - base_r,
                     "delta_p_up": p - base_p, "features_changed": ", ".join(changed)})
    df = pd.DataFrame(rows)
    df.attrs["models_used"] = list(feat_models)
    df.attrs["note"] = "Feature-based models only; SCENARIOS show model sensitivity, not predictions."
    return df
