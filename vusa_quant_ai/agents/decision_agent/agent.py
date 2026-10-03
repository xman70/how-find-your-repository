"""Decision Aggregator - summarises the committee (not a vote).

Produces BULL CASE / BEAR CASE / NEUTRAL CASE with evidence from every agent, lists
disagreements explicitly, and states how the quantitative decision relates to the
committee's evidence.
"""
from __future__ import annotations

from ..base import AgentOutput


def aggregate(outputs: dict[str, AgentOutput], decision: dict) -> dict:
    bull, bear, neutral, concerns, gaps = [], [], [], [], []
    stances = {}
    for name, o in outputs.items():
        stances[name] = o.stance
        for f in o.findings:
            line = f"[{name}] {f.text}"
            if f.direction == "bullish":
                bull.append((f.weight, line))
            elif f.direction in ("bearish", "risk"):
                bear.append((f.weight, line))
            else:
                neutral.append((f.weight, line))
        concerns += [f"[{name}] {c}" for c in o.concerns]
        gaps += [f"[{name}] {g}" for g in o.data_gaps]
    order = lambda xs: [t for _, t in sorted(xs, key=lambda x: -x[0])]  # noqa: E731
    non_skeptic = {k: v for k, v in stances.items() if k != "Skeptic"}
    distinct = set(v for v in non_skeptic.values() if v != "NEUTRAL")
    disagreement = len(distinct) > 1
    summary = ("Committee stances: " + ", ".join(f"{k}={v}" for k, v in non_skeptic.items()) +
               (". DISAGREEMENT between specialists." if disagreement else ". No directional conflict between specialists."))
    skeptic = outputs.get("Skeptic")
    return {"bull_case": order(bull)[:10], "bear_case": order(bear)[:10], "neutral_case": order(neutral)[:8],
            "key_risks": concerns[:12], "data_gaps": gaps, "stances": stances, "disagreement": disagreement,
            "summary": summary, "skeptic": getattr(skeptic, "counter_case", {}) if skeptic else {},
            "decision_relation": f"Quantitative decision: {decision.get('label')} (final score {decision.get('final_score')}, "
                                 f"opportunity {decision.get('opportunity')}, risk {decision.get('risk')}). The committee "
                                 f"does not override it; it documents the evidence and the counter-case."}
