"""Alert engine: compares today's decision state with the previous one and raises
structured alerts (stored in the database and shown in the GUI)."""
from __future__ import annotations

import hashlib

import numpy as np

from ..core.runtime import utcnow_iso

KINDS = ["SIGNAL CHANGE", "PROBABILITY CHANGE", "REGIME CHANGE", "VOLATILITY SHOCK", "MODEL DRIFT", "DATA FAILURE",
         "NEWS SHOCK", "MACRO EVENT", "PRICE BREAKOUT", "LARGE DRAWDOWN", "UNUSUAL MARKET CONDITION"]


def generate_alerts(state: dict, prev: dict | None, cfg) -> list[dict]:
    a = []

    def add(kind, sev, msg, payload=None):
        aid = hashlib.sha1(f"{state.get('as_of')}{kind}{msg}".encode()).hexdigest()[:16]
        a.append({"alert_id": aid, "created_at": utcnow_iso(), "kind": kind, "severity": sev, "message": msg,
                  "payload": payload or {}, "acknowledged": 0})

    d = state.get("decision", {})
    if d.get("changed"):
        add("SIGNAL CHANGE", "high", d.get("change_reason") or f"{d.get('previous_signal')} -> {d.get('signal')}")
    fc = state.get("forecast_primary", {})
    if prev and fc.get("p_up") is not None and prev.get("forecast_primary", {}).get("p_up") is not None:
        dp = fc["p_up"] - prev["forecast_primary"]["p_up"]
        if abs(dp) >= cfg.probability_change:
            add("PROBABILITY CHANGE", "medium", f"{fc.get('horizon')}D P(up) changed {dp:+.0%} to {fc['p_up']:.0%}")
    rg = state.get("regime", {})
    if rg.get("days_in_regime", 99) <= 1 and rg.get("previous_regime"):
        add("REGIME CHANGE", "high", f"Regime changed {rg['previous_regime']} -> {rg['regime']}")
    f = state.get("features_now", {})
    rvz = f.get("rv20_z_5y")
    if rvz is not None and np.isfinite(rvz) and rvz > cfg.volatility_z:
        add("VOLATILITY SHOCK", "high", f"20d realised volatility z-score {rvz:.1f} (5y)")
    if state.get("model_drift", {}).get("alert"):
        add("MODEL DRIFT", "high", state["model_drift"]["message"])
    if state.get("feature_drift", {}).get("alert"):
        add("UNUSUAL MARKET CONDITION", "medium", state["feature_drift"]["message"])
    if not state.get("data_current", True):
        add("DATA FAILURE", "high", "; ".join(state.get("warnings", []))[:400] or "Data not current")
    failed = [s for s in state.get("source_health", []) if s.get("status") == "failed"]
    if failed:
        add("DATA FAILURE", "low", f"{len(failed)} source(s) failed: " + ", ".join(f"{s['dataset']}/{s['source']}" for s in failed[:6]))
    if state.get("news", {}).get("news_shock"):
        add("NEWS SHOCK", "high", "Multiple high-magnitude bearish events from credible sources")
    for e in state.get("news", {}).get("events", []):
        if e["category"] in ("Monetary Policy", "Inflation", "Employment") and e["confidence"] >= 80:
            add("MACRO EVENT", "medium", f"{e['category']}: {e.get('title', '')[:120]}")
            break
    _price, hi = state.get("price"), f.get("dist_high_60")
    if hi is not None and np.isfinite(hi) and hi >= -0.001:
        add("PRICE BREAKOUT", "medium", f"Price at/above its {cfg.breakout_lookback}-session high")
    lo = f.get("dist_low_60")
    if lo is not None and np.isfinite(lo) and lo <= 0.001:
        add("PRICE BREAKOUT", "medium", f"Price at/below its {cfg.breakout_lookback}-session low (downside breakout)")
    dd = state.get("risk_metrics", {}).get("drawdown_current")
    if dd is not None and dd <= -cfg.drawdown_pct:
        add("LARGE DRAWDOWN", "high", f"VUSA is {dd:.1%} below its peak")
    for fl in state.get("shocks", {}).get("flags", []):
        add("UNUSUAL MARKET CONDITION", "medium", fl)
    return a
