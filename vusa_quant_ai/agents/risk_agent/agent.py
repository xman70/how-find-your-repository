"""Agent 5 - Risk manager: downside, tails, stress scenarios, shocks."""
from __future__ import annotations

import numpy as np

from ..base import Agent, Finding


class RiskAgent(Agent):
    name = "Risk Manager"
    role = "Downside risk, stress scenarios and shock monitoring"

    def run(self, state: dict):
        rm, f, concerns = state.get("risk_metrics", {}), [], []
        risk = state.get("decision", {}).get("risk")
        if risk is not None:
            f.append(Finding(f"Composite risk score {risk:.0f}/100", "bearish" if risk >= 60 else "bullish" if risk <= 35 else "neutral", 1.2))
        if rm:
            f.append(Finding(f"1-day 95% VaR {rm.get('VaR95_1d_hist', np.nan):.2%} (historical), CVaR {rm.get('CVaR95_1d_hist', np.nan):.2%}; "
                             f"20-day 95% VaR {rm.get('VaR95_20d_hist', np.nan):.1%}", "neutral", 0.5))
            dd = rm.get("drawdown_current", 0)
            f.append(Finding(f"Current drawdown {dd:.1%}; worst 1y drawdown {rm.get('max_drawdown_1y', np.nan):.1%}",
                             "bearish" if dd < -0.1 else "neutral", 0.8))
        mc = state.get("monte_carlo_primary", {})
        if mc:
            f.append(Finding(f"Monte Carlo (block bootstrap) 20D: P10 {mc.get('P10', np.nan):+.1%}, median {mc.get('P50', np.nan):+.1%}, "
                             f"P90 {mc.get('P90', np.nan):+.1%}; P(max drawdown > 10%) {mc.get('p_maxdd_gt10', np.nan):.0%}", "neutral", 0.6))
        worst = sorted([s for s in state.get("stress", []) if s.get("estimated_vusa_impact") is not None and
                        np.isfinite(s["estimated_vusa_impact"]) and not s["scenario"].startswith("Replay")],
                       key=lambda s: s["estimated_vusa_impact"])[:2]
        for s in worst:
            f.append(Finding(f"SCENARIO '{s['scenario']}': estimated VUSA impact {s['estimated_vusa_impact']:+.1%} ({s['basis']})", "risk", 0.4))
        for fl in state.get("shocks", {}).get("flags", []):
            concerns.append(f"Shock: {fl}")
        p_dd = state.get("forecast_primary", {}).get("p_dd10")
        if p_dd is not None and np.isfinite(p_dd) and p_dd > 0.2:
            concerns.append(f"Calibrated P(max drawdown > 10% within horizon) = {p_dd:.0%}")
        for b in state.get("correlation_breakdowns", [])[:3]:
            concerns.append(f"Correlation breakdown vs {b['asset']}: {b['current_corr']:+.2f} vs typical {b['typical_corr']:+.2f}")
        return self._out(f, concerns)
