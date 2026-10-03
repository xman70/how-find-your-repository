"""Agent 4 - Technical analyst: trend, momentum, volatility, breadth, market structure."""
from __future__ import annotations

import numpy as np

from ..base import Agent, Finding, comp_score


class TechnicalAgent(Agent):
    name = "Technical"
    role = "Trend, momentum, volatility and market structure"

    def run(self, state: dict):
        f, concerns = [], []
        comps = state.get("decision", {}).get("components", {})
        for k in ("trend", "momentum", "volatility", "breadth"):
            c = comps.get(k, {})
            s = c.get("score")
            if s is None:
                continue
            d = "bullish" if s >= 60 else "bearish" if s <= 40 else "neutral"
            f.append(Finding(f"{k.capitalize()} score {s:.0f}/100: " + "; ".join(c.get("evidence", [])[:2]), d, 1.0))
        sr = state.get("support_resistance", {})
        price = state.get("price")
        if sr and price:
            if sr.get("support"):
                s0 = sr["support"][0]["level"]
                f.append(Finding(f"Nearest support {s0:.2f} ({s0 / price - 1:+.1%})", "neutral", 0.2))
            if sr.get("resistance"):
                r0 = sr["resistance"][0]["level"]
                f.append(Finding(f"Nearest resistance {r0:.2f} ({r0 / price - 1:+.1%})", "neutral", 0.2))
        rsi = state.get("features_now", {}).get("rsi14")
        if rsi is not None and np.isfinite(rsi) and rsi > 75:
            concerns.append(f"RSI {rsi:.0f}: overbought - short-term pullback risk")
        if rsi is not None and np.isfinite(rsi) and rsi < 28:
            concerns.append(f"RSI {rsi:.0f}: oversold - mean-reversion bounce possible but trend may be broken")
        t, m = comp_score(state, "trend"), comp_score(state, "momentum")
        if np.isfinite(t) and np.isfinite(m) and (t - 50) * (m - 50) < 0:
            concerns.append("Trend and momentum disagree (possible turning point).")
        return self._out(f, concerns)
