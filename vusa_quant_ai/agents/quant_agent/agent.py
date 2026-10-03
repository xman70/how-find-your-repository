"""Agent 1 - Quant: statistical patterns in price (return distribution, autocorrelation,
seasonality-free drift, tail behaviour, structural change)."""
from __future__ import annotations

import numpy as np

from ..base import Agent, Finding


class QuantAgent(Agent):
    name = "Quant"
    role = "Price and statistical pattern analysis"

    def run(self, state: dict):
        f, gaps, concerns = [], [], []
        st = state.get("stats", {})
        if st:
            ac = st.get("autocorr_1d")
            if ac is not None and np.isfinite(ac):
                f.append(Finding(f"Lag-1 daily return autocorrelation {ac:+.3f} "
                                 f"({'momentum-like' if ac > 0.05 else 'mean-reverting' if ac < -0.05 else 'close to random walk'})",
                                 "neutral", 0.3))
            dr = st.get("drift_1y_ann")
            if dr is not None:
                f.append(Finding(f"Trailing 1y annualised drift {dr:+.1%} vs long-run {st.get('drift_long_ann', np.nan):+.1%}",
                                 "bullish" if dr > 0.05 else "bearish" if dr < -0.05 else "neutral", 0.8))
            sk = st.get("skew_60d")
            if sk is not None and sk < -1:
                concerns.append(f"Strong negative skew in last 60 sessions ({sk:.2f}) - crash-prone return distribution")
            hr = st.get("hit_rate_60d")
            if hr is not None:
                f.append(Finding(f"Up-day frequency last 60 sessions {hr:.0%}", "bullish" if hr > 0.56 else
                                 "bearish" if hr < 0.44 else "neutral", 0.5))
        cp = state.get("structural_change", {})
        if cp.get("structural_change_detected"):
            concerns.append("Change-point tests flag a recent structural break: historical relationships may not hold.")
            f.append(Finding("Structural change detected (CUSUM / Page-Hinkley / BOCPD / KS)", "risk", 0.8))
        an = state.get("analogues", {}).get("summary", {})
        if an:
            r20 = an.get("ret_20d", {})
            base = state.get("analogues", {}).get("base_rates", {}).get("base_20d_mean", 0.0)
            f.append(Finding(f"Nearest historical analogues: 20D weighted mean {r20.get('weighted_mean', np.nan):+.1%} "
                             f"(unconditional {base:+.1%}), {r20.get('p_positive', np.nan):.0%} positive",
                             "bullish" if r20.get("weighted_mean", 0) > base + 0.005 else
                             "bearish" if r20.get("weighted_mean", 0) < base - 0.005 else "neutral", 1.0))
        else:
            gaps.append("analogue search unavailable")
        return self._out(f, concerns, gaps)
