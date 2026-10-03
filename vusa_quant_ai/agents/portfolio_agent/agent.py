"""Agent 8 - Portfolio analyst: risk/reward and exposure (research only, never trades)."""
from __future__ import annotations

import numpy as np

from ..base import Agent, Finding, comp_score


class PortfolioAgent(Agent):
    name = "Portfolio Analyst"
    role = "Risk/reward, exposure and position-sizing research"

    def run(self, state: dict):
        f, concerns, gaps = [], [], []
        rr = comp_score(state, "risk_reward")
        if np.isfinite(rr):
            ev = state.get("decision", {}).get("components", {}).get("risk_reward", {}).get("evidence", [])
            f.append(Finding(f"Risk/reward score {rr:.0f}/100: " + "; ".join(ev[:2]),
                             "bullish" if rr >= 60 else "bearish" if rr <= 40 else "neutral", 1.0))
        d = state.get("decision", {})
        if d:
            f.append(Finding(f"Quadrant: {d.get('quadrant')}", "neutral", 0.2))
        pe = state.get("portfolio")
        if pe:
            f.append(Finding(f"Current exposure {pe['exposure_pct']:.0%} of portfolio; estimated 20D VaR95 "
                             f"{pe['est_20d_VaR95_value']:,.0f} ({pe['portfolio_impact_VaR_pct']:.1%} of portfolio)", "neutral", 0.5))
            if pe.get("within_tolerance") is False:
                concerns.append(f"Estimated 20D downside exceeds the stated {state.get('risk_tolerance')} tolerance "
                                f"({pe['loss_tolerance_pct']:.0%}).")
        else:
            gaps.append("Portfolio details not entered (Settings > Portfolio) - exposure analysis skipped")
        ps = state.get("position_sizing", {})
        if ps:
            best = max(ps.items(), key=lambda kv: kv[1].get("sharpe", -9) if kv[1].get("sharpe") is not None else -9)
            f.append(Finding(f"Historically, '{best[0]}' sizing had the best Sharpe ({best[1].get('sharpe', np.nan):.2f}, "
                             f"max DD {best[1].get('max_drawdown', np.nan):.1%}) - research only, no leverage", "neutral", 0.2))
        return self._out(f, concerns, gaps)
