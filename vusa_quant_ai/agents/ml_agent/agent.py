"""Agent 6 - ML scientist: model predictions, consensus, calibration and reliability."""
from __future__ import annotations

import numpy as np

from ..base import Agent, Finding


class MLScientistAgent(Agent):
    name = "ML Scientist"
    role = "Model predictions, out-of-sample reliability and calibration"

    def run(self, state: dict):
        fc = state.get("forecast_primary", {})
        f, concerns, gaps = [], [], []
        if not fc or fc.get("p_up") is None or not np.isfinite(fc.get("p_up", np.nan)):
            return self._out([], ["No usable ML forecast"], ["forecast unavailable"], score=0.0)
        h = fc["horizon"]
        f.append(Finding(f"{h}D ensemble ({fc.get('ensemble_method')}) expected return {fc['expected_return']:+.2%}, "
                         f"calibrated P(up) {fc['p_up']:.0%}, {int(100 * fc.get('interval_level', 0.8))}% interval "
                         f"{fc['interval'][0]:+.1%} .. {fc['interval'][1]:+.1%}",
                         "bullish" if fc["p_up"] > 0.57 else "bearish" if fc["p_up"] < 0.43 else "neutral", 1.2))
        dis = fc.get("disagreement", {})
        if dis.get("sign_agreement") is not None:
            f.append(Finding(f"Model sign agreement {dis['sign_agreement']:.0%} across {dis.get('n_models')} models "
                             f"(spread of forecasts {dis.get('std_of_model_returns', np.nan):.2%})", "neutral", 0.4))
            if dis["sign_agreement"] < 0.65:
                concerns.append("Models disagree on direction - epistemic uncertainty is high.")
        ens = fc.get("ensemble_oos", {})
        naive = fc.get("model_scores", {}).get("naive_drift", {})
        if ens:
            da, ic = ens.get("directional_accuracy"), ens.get("ic_spearman")
            f.append(Finding(f"Ensemble out-of-sample: directional accuracy {da:.1%}, rank IC {ic:+.3f}, n={ens.get('n')}",
                             "neutral", 0.3))
            if naive and da is not None and naive.get("directional_accuracy") is not None and da <= naive["directional_accuracy"] + 0.005:
                concerns.append(f"Ensemble does NOT beat the naive drift benchmark out-of-sample "
                                f"({da:.1%} vs {naive['directional_accuracy']:.1%}).")
        cal = fc.get("calibration", {})
        if cal:
            e = cal.get("calibrated", {}).get("ece")
            if e is not None and e > 0.08:
                concerns.append(f"Probability calibration is weak out-of-sample (ECE {e:.2f}).")
        cov = fc.get("conformal", {})
        if cov.get("coverage") is not None and abs(cov["coverage"] - cov["nominal"]) > 0.08:
            concerns.append(f"Conformal interval coverage {cov['coverage']:.0%} vs nominal {cov['nominal']:.0%} out-of-sample.")
        for k in ("model_drift", "feature_drift"):
            d = state.get(k, {})
            if d.get("alert"):
                concerns.append(d.get("message", k))
        errs = fc.get("model_errors", {})
        if errs:
            gaps.append(f"{len(errs)} models skipped/failed: " + ", ".join(list(errs)[:6]))
        return self._out(f, concerns, gaps)
