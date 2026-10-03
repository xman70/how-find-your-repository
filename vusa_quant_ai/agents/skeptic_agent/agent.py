"""Agent 7 - Skeptic (mandatory adversarial review).

Whenever the proposed signal is BUY, it actively searches the whole state for evidence
supporting HOLD or SELL; when SELL, for HOLD or BUY; when HOLD, for both directions.
It also attacks the *process*: weak out-of-sample skill, poor calibration, model drift,
out-of-distribution inputs, stale data, proxy history, revision risk and small samples.
The goal is to surface confirmation bias, not to win an argument.
"""
from __future__ import annotations

import numpy as np

from ..base import Agent, AgentOutput, Finding


class SkepticAgent(Agent):
    name = "Skeptic"
    role = "Adversarial review: argue against the proposed signal"

    def run(self, state: dict) -> AgentOutput:
        d = state.get("decision", {})
        sig = d.get("signal", "HOLD")
        comps = d.get("components", {})
        against: list[Finding] = []
        want = {"BUY": ("bearish",), "SELL": ("bullish",), "HOLD": ("bullish", "bearish")}[sig]
        for name, c in comps.items():
            s = c.get("score")
            if s is None:
                continue
            dirn = "bullish" if s >= 58 else "bearish" if s <= 42 else None
            if dirn in want:
                against.append(Finding(f"{name} ({s:.0f}/100) argues {dirn}: " + "; ".join(c.get("evidence", [])[:2]), dirn, 1.0))
        for ag in state.get("agents", {}).values():
            for fnd in ag.get("findings", []):
                if fnd["direction"] in want or (sig == "BUY" and fnd["direction"] == "risk"):
                    against.append(Finding(f"[{ag['agent']}] {fnd['text']}", fnd["direction"], 0.5))
            for c in ag.get("concerns", []):
                against.append(Finding(f"[{ag['agent']} concern] {c}", "risk", 0.6))
        process = []
        fc = state.get("forecast_primary", {})
        ens = fc.get("ensemble_oos", {})
        if ens and (ens.get("directional_accuracy") or 0) < 0.55:
            process.append(f"Out-of-sample directional accuracy is only {ens.get('directional_accuracy', np.nan):.1%} - "
                           "barely better than a coin flip; the edge may be noise.")
        naive = fc.get("model_scores", {}).get("naive_drift", {})
        if ens and naive and (ens.get("directional_accuracy") or 0) <= (naive.get("directional_accuracy") or 0) + 0.01:
            process.append("The ensemble does not clearly beat the 'always expect the average drift' benchmark.")
        if fc.get("calibration", {}).get("calibrated", {}).get("ece", 0) > 0.07:
            process.append("Probabilities are not well calibrated out-of-sample.")
        if state.get("model_drift", {}).get("alert"):
            process.append("Model drift: recent live performance is worse than historical validation.")
        if state.get("feature_drift", {}).get("alert"):
            process.append("Today's inputs are outside the training distribution (OUT-OF-DISTRIBUTION).")
        if not state.get("data_current", True):
            process.append("Data is NOT current - the assessment may be stale.")
        if state.get("is_synthetic"):
            process.append("SYNTHETIC DEMO DATA - nothing here describes the real market.")
        if state.get("proxy_note"):
            process.append("Part of the training history is a proxy (S&P 500 in EUR) rather than VUSA itself.")
        audit = state.get("leakage_audit", {})
        for c in audit.get("checks", []):
            if c["status"] != "PASS":
                process.append(f"Leakage audit {c['status']}: {c['name']} - {c['detail'][:140]}")
        if fc.get("n_oos", 1e9) < 500:
            process.append(f"Only {fc.get('n_oos')} out-of-sample observations; statistics are noisy.")
        for c in d.get("contradictions", []):
            process.append(c)
        bull = [x.text for x in against if x.direction == "bullish"]
        bear = [x.text for x in against if x.direction in ("bearish", "risk")]
        # skeptic score: positive = case against the signal is strong
        n_comp = sum(1 for c in comps.values() if c.get("score") is not None) or 1
        n_against_comp = sum(1 for x in against if x.weight == 1.0)
        strength = min(1.0, 0.6 * n_against_comp / n_comp + 0.06 * len(process))
        out = AgentOutput(self.name, self.role, "MIXED", -strength if sig == "BUY" else strength if sig == "SELL" else 0.0,
                          against[:25], process, [])
        out.counter_case = {"proposed_signal": sig, "evidence_against": bull if sig == "SELL" else bear,
                            "evidence_for_opposite": bull if sig != "BUY" else [], "process_risks": process,
                            "strength_of_counter_case": round(strength, 2)}  # type: ignore[attr-defined]
        return out
