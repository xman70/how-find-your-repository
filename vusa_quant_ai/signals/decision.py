"""DECISION ENGINE.

Final signal = f(12 independent components, risk score, confirmations, hysteresis,
confidence penalties). The decision is never "ML says up -> BUY".

* Opportunity score (0-100): weighted average of the 12 components (weights shrink with
  missing inputs).
* Risk score (0-100): from the risk engine (vol, drawdown, tails, regime, simulated
  downside, shocks, forecast uncertainty).
* Thresholds + hysteresis + minimum duration -> stable BUY / HOLD / SELL.
* Confirmation: a STRONG signal needs agreement of ML, trend, risk/reward and regime.
  If only one subsystem is bullish the output is "HOLD / WEAK BULLISH".
* Confidence: forecast confidence x component agreement x (1 - penalties for shocks,
  drift, out-of-distribution inputs and stale data).
* "What would change my mind": the input levels at which the signal would flip.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config.settings import SignalSettings
from .components import Component

KEY_CONFIRMERS = ("ml_forecast", "trend", "risk_reward", "regime")
EMOJI = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}


@dataclass
class Decision:
    signal: str
    label: str
    strength: str
    opportunity: float
    risk: float
    final_score: float
    confidence: float
    agreement: float
    components: dict[str, dict]
    contributions: dict[str, float]
    quadrant: str
    previous_signal: str | None
    duration_days: int
    changed: bool
    change_reason: str
    confirmations: dict[str, bool]
    contradictions: list[str]
    what_would_change: list[str]
    penalties: dict[str, float]
    notes: list[str] = field(default_factory=list)

    @property
    def emoji(self) -> str:
        return EMOJI[self.signal]

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def opportunity_score(comps: dict[str, Component], weights: dict[str, float]) -> tuple[float, dict[str, float], dict]:
    num = den = 0.0
    eff_w = {}
    for name, c in comps.items():
        if c is None or not np.isfinite(c.score):
            continue
        w = weights.get(name, 0.5) * max(0.25, c.coverage)
        eff_w[name] = w
        num += w * c.score
        den += w
    opp = num / den if den else 50.0
    contrib = {n: eff_w[n] * (comps[n].score - 50) / den for n in eff_w}  # points vs neutral 50
    return float(opp), contrib, eff_w


def quadrant(opp: float, risk: float) -> str:
    return f"{'HIGH' if opp >= 55 else 'LOW'} OPPORTUNITY / {'HIGH' if risk >= 50 else 'LOW'} RISK"


def decide(comps: dict[str, Component], risk: float, cfg: SignalSettings, forecast_conf: float, history: pd.DataFrame | None,
           penalties: dict[str, float], context: dict) -> Decision:
    opp, contrib, eff_w = opportunity_score(comps, cfg.weights)
    # risk-adjusted final score: high risk drags the score down, low risk adds a little
    final = float(np.clip(opp - 0.25 * (risk - 50), 0, 100))

    prev = history.iloc[-1] if history is not None and len(history) else None
    prev_sig = prev["signal"] if prev is not None else None
    prev_dur = int(prev["duration_days"]) if prev is not None and pd.notna(prev.get("duration_days")) else 0

    # ------------------------------------------------- hysteresis state machine
    b, s, h = cfg.buy_threshold, cfg.sell_threshold, cfg.hysteresis
    young = prev_sig is not None and prev_dur < cfg.min_signal_days
    margin = h * (2 if young else 1)
    if prev_sig == "BUY":
        raw = "BUY" if final >= b - margin and risk <= cfg.max_risk_for_buy + h else ("SELL" if final <= s - margin else "HOLD")
    elif prev_sig == "SELL":
        raw = "SELL" if final <= s + margin else ("BUY" if final >= b + margin and risk <= cfg.max_risk_for_buy else "HOLD")
    elif prev_sig == "HOLD":
        raw = "BUY" if final >= b + (h if young else 0) and risk <= cfg.max_risk_for_buy else (
            "SELL" if final <= s - (h if young else 0) else "HOLD")
    else:
        raw = "BUY" if final >= b and risk <= cfg.max_risk_for_buy else ("SELL" if final <= s else "HOLD")

    # ------------------------------------------------- confirmation
    conf_map = {}
    for k in KEY_CONFIRMERS:
        c = comps.get(k)
        sc = c.score if c is not None and np.isfinite(c.score) else 50
        conf_map[k] = bool(sc >= 60) if raw != "SELL" else bool(sc <= 40)
    n_conf = sum(conf_map.values())
    bullish_systems = [k for k, c in comps.items() if c is not None and np.isfinite(c.score) and c.score >= 60]
    bearish_systems = [k for k, c in comps.items() if c is not None and np.isfinite(c.score) and c.score <= 40]
    signal, strength = raw, "MODERATE"
    notes = []
    if raw == "BUY":
        if n_conf >= cfg.strong_confirmations_required and final >= cfg.strong_signal_threshold:
            strength = "STRONG"
        elif len(bullish_systems) <= 1:
            signal, strength = "HOLD", "WEAK BULLISH"
            notes.append("Only one subsystem is bullish - downgraded to HOLD / WEAK BULLISH.")
    elif raw == "SELL":
        if n_conf >= cfg.strong_confirmations_required and final <= 100 - cfg.strong_signal_threshold:
            strength = "STRONG"
        elif len(bearish_systems) <= 1:
            signal, strength = "HOLD", "WEAK BEARISH"
            notes.append("Only one subsystem is bearish - downgraded to HOLD / WEAK BEARISH.")
    else:
        strength = "WEAK BULLISH" if final >= (b + 50) / 2 else ("WEAK BEARISH" if final <= (s + 50) / 2 else "NEUTRAL")
    if final >= b and risk > cfg.max_risk_for_buy and signal != "BUY":
        notes.append(f"Score {final:.0f} reaches the BUY band but the risk score {risk:.0f} exceeds the BUY limit "
                     f"{cfg.max_risk_for_buy:.0f} - BUY withheld.")

    # ------------------------------------------------- agreement & confidence
    scores = np.array([c.score for c in comps.values() if c is not None and np.isfinite(c.score)])
    dirs = np.sign(scores - 50)
    agreement = float(max((dirs > 0).mean(), (dirs < 0).mean())) if len(dirs) else 0.0
    dispersion = float(np.std(scores) / 25) if len(scores) else 1.0
    base_conf = 0.5 * forecast_conf + 0.3 * agreement + 0.2 * (1 - min(1.0, dispersion))
    pen_total = 1.0
    for v in penalties.values():
        pen_total *= (1 - v)
    confidence = float(np.clip(base_conf * pen_total, 0.05, 0.95))
    if signal in ("BUY", "SELL") and confidence < cfg.min_confidence_for_directional:
        notes.append(f"Raw signal {signal} suppressed: confidence {confidence:.0%} is below the "
                     f"{cfg.min_confidence_for_directional:.0%} minimum for a directional signal.")
        strength = f"LOW CONFIDENCE - raw {signal}"
        signal = "HOLD"

    # ------------------------------------------------- contradictions
    contradictions = []
    labels = {k: ("bullish" if c.score >= 60 else "bearish" if c.score <= 40 else "neutral")
              for k, c in comps.items() if c is not None and np.isfinite(c.score)}
    bulls = [k for k, v in labels.items() if v == "bullish"]
    bears = [k for k, v in labels.items() if v == "bearish"]
    if bulls and bears:
        contradictions.append("Signal disagreement detected: bullish = " + ", ".join(bulls) + "; bearish = " + ", ".join(bears))
    if comps.get("valuation") and np.isfinite(comps["valuation"].score) and comps["valuation"].score < 40 and signal == "BUY":
        contradictions.append("Valuation looks expensive while the overall signal is BUY.")
    if comps.get("ml_forecast") and comps.get("trend"):
        m, t = comps["ml_forecast"].score, comps["trend"].score
        if np.isfinite(m) and np.isfinite(t) and (m - 50) * (t - 50) < 0 and abs(m - 50) > 10 and abs(t - 50) > 10:
            contradictions.append(f"ML forecast ({m:.0f}) and trend ({t:.0f}) point in opposite directions.")

    # ------------------------------------------------- change tracking
    changed = prev_sig is not None and prev_sig != signal
    duration = 1 if (prev_sig is None or changed) else prev_dur + 1
    reason = ""
    if changed:
        top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:3]
        reason = (f"{prev_sig} -> {signal}: final score {final:.0f} crossed the "
                  f"{'BUY' if signal == 'BUY' else 'SELL' if signal == 'SELL' else 'HOLD'} band; main drivers: " +
                  ", ".join(f"{k} {v:+.1f}" for k, v in top) +
                  (f"; confirmations {n_conf}/{len(KEY_CONFIRMERS)}" if signal != "HOLD" else ""))

    wwc = what_would_change(signal, final, comps, eff_w, cfg, risk, context)
    return Decision(signal=signal, label=f"{EMOJI[signal]} {signal}" + (f" ({strength})" if strength else ""),
                    strength=strength, opportunity=round(opp, 1), risk=round(risk, 1), final_score=round(final, 1),
                    confidence=round(confidence, 3), agreement=round(agreement, 3),
                    components={k: c.to_dict() for k, c in comps.items() if c is not None},
                    contributions={k: round(v, 2) for k, v in contrib.items()}, quadrant=quadrant(opp, risk),
                    previous_signal=prev_sig, duration_days=duration, changed=changed, change_reason=reason,
                    confirmations=conf_map, contradictions=contradictions, what_would_change=wwc,
                    penalties={k: round(v, 3) for k, v in penalties.items()}, notes=notes)


def what_would_change(signal: str, final: float, comps: dict[str, Component], eff_w: dict, cfg: SignalSettings,
                      risk: float, ctx: dict) -> list[str]:
    out = []
    den = sum(eff_w.values()) or 1.0
    if signal == "BUY":
        target, verb = cfg.buy_threshold - cfg.hysteresis, "weaken to HOLD"
    elif signal == "SELL":
        target, verb = cfg.sell_threshold + cfg.hysteresis, "improve to HOLD"
    else:
        up_gap, dn_gap = cfg.buy_threshold - final, final - cfg.sell_threshold
        out.append(f"Final score {final:.0f}: needs +{up_gap:.0f} points for BUY or -{dn_gap:.0f} for SELL.")
        target, verb = (cfg.buy_threshold, "upgrade to BUY") if up_gap <= dn_gap else (cfg.sell_threshold, "downgrade to SELL")
    gap = target - final
    for name, w in sorted(eff_w.items(), key=lambda kv: -kv[1]):
        c = comps[name]
        needed = c.score + gap * den / w
        if 0 <= needed <= 100 and abs(needed - c.score) < 45:
            out.append(f"Signal would {verb} if {name} score moved from {c.score:.0f} to {needed:.0f} (others unchanged).")
    if signal == "BUY":
        out.append(f"Signal would {verb} if the risk score rose above {cfg.max_risk_for_buy:.0f} (now {risk:.0f}).")
    price, sma200 = ctx.get("price"), ctx.get("sma200")
    if price and sma200:
        out.append(f"Trend break level: price {'below' if price > sma200 else 'above'} SMA200 at {sma200:.2f} "
                   f"({sma200 / price - 1:+.1%} from {price:.2f}).")
    if ctx.get("vix_now") is not None:
        out.append(f"Volatility threshold: VIX above {max(25.0, ctx['vix_now'] * 1.5):.0f} would cut the volatility component sharply.")
    if ctx.get("p_up") is not None:
        p = ctx["p_up"]
        out.append(f"ML threshold: calibrated 20D P(up) {'falling below 50%' if p >= 0.5 else 'rising above 55%'} (now {p:.0%}).")
    if ctx.get("p_lt5") is not None and np.isfinite(ctx["p_lt5"]):
        out.append(f"Downside threshold: P(20D return < -5%) above 25% (now {ctx['p_lt5']:.0%}).")
    out.append("A regime change (e.g. to Bear / High volatility), model-drift alarm, or a sharp rise in model disagreement "
               "would lower confidence and may change the signal.")
    return out
