"""Agent 2 - Macro: rates, inflation, growth, labour, credit, liquidity, recession risk."""
from __future__ import annotations

import numpy as np

from ..base import Agent, Finding


def _v(m: dict, k: str):
    v = m.get(k)
    return v if v is not None and np.isfinite(v) else None


class MacroAgent(Agent):
    name = "Macro"
    role = "Rates, inflation, GDP, labour, liquidity and recession indicators"

    def run(self, state: dict):
        m = state.get("macro_snapshot", {})
        f, concerns, gaps = [], [], []
        if not m:
            return self._out([], [], ["No macro data available (FRED unreachable or not configured)"], score=0.0)
        if (c := _v(m, "curve_10y3m")) is not None or (c := _v(m, "curve_10y_3m_mkt")) is not None:
            f.append(Finding(f"10Y-3M curve {c:+.2f}pp", "bearish" if c < 0 else "bullish" if c > 0.75 else "neutral", 0.6))
            if c < 0:
                concerns.append("Inverted yield curve: historically a recession lead indicator with long, variable lags.")
        if (hy := _v(m, "hy_oas")) is not None:
            chg = _v(m, "hy_oas_chg_20d") or 0.0
            f.append(Finding(f"High-yield OAS {hy:.2f}% ({chg:+.2f} over 20d)",
                             "bearish" if hy > 5.5 or chg > 0.5 else "bullish" if hy < 4 and chg <= 0 else "neutral", 1.0))
        rec = _v(m, "sahm_rt") if _v(m, "sahm_rt") is not None else _v(m, "unrate_vs_12m_low")
        if rec is not None:
            f.append(Finding(f"Sahm-style recession indicator {rec:.2f} (trigger 0.50)",
                             "bearish" if rec >= 0.5 else "neutral" if rec >= 0.3 else "bullish", 1.2))
            if rec >= 0.5:
                concerns.append("Labour-market recession trigger is active.")
        if (inf := _v(m, "cpi_yoy")) is not None:
            tr = _v(m, "inflation_trend_6m") or 0.0
            f.append(Finding(f"CPI inflation {inf:.1%} YoY, 6m trend {tr:+.2%}",
                             "bearish" if inf > 0.04 and tr > 0 else "bullish" if tr < 0 and inf < 0.035 else "neutral", 0.8))
        if (ry := _v(m, "real_yield_10y")) is not None:
            f.append(Finding(f"10Y real yield {ry:.2f}%", "bearish" if ry > 2.0 else "neutral", 0.5))
        if (ffc := _v(m, "fed_funds_chg_6m")) is not None:
            f.append(Finding(f"Fed funds change over 6m {ffc:+.2f}pp ({'easing' if ffc < -0.1 else 'tightening' if ffc > 0.1 else 'on hold'})",
                             "bullish" if ffc < -0.1 else "bearish" if ffc > 0.25 else "neutral", 0.6))
        if (nf := _v(m, "nfci")) is not None:
            f.append(Finding(f"Chicago Fed NFCI {nf:+.2f} ({'tight' if nf > 0 else 'loose'} financial conditions)",
                             "bearish" if nf > 0 else "bullish", 0.6))
        for k in ("PCEPILFE", "GDPC1", "PAYEMS"):
            if state.get("macro_status", {}).get(k, "").startswith("unavailable"):
                gaps.append(f"{k} unavailable")
        pit = state.get("pit_quality")
        if pit:
            gaps.append(pit)
        return self._out(f, concerns, gaps)
