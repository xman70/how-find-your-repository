"""Virtual specialist agents (the DAILY AI MARKET COMMITTEE).

Agents are deterministic analysts operating on the structured state produced by the
quantitative engine. They never fetch or invent numbers: each finding cites a value that
exists in the state. The committee is not a vote - the Decision Aggregator summarises
evidence and disagreements; the Skeptic must argue against the proposed signal.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Finding:
    text: str
    direction: str  # bullish | bearish | neutral | risk
    weight: float = 1.0


@dataclass
class AgentOutput:
    agent: str
    role: str
    stance: str  # BULLISH | BEARISH | NEUTRAL | MIXED
    score: float  # -1 .. +1
    findings: list[Finding] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    data_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(float(self.score), 3) if np.isfinite(self.score) else None
        return d


def stance_from(score: float, thr: float = 0.15) -> str:
    if not np.isfinite(score):
        return "NEUTRAL"
    return "BULLISH" if score > thr else ("BEARISH" if score < -thr else "NEUTRAL")


def finding_score(findings: list[Finding]) -> float:
    num = sum(f.weight * (1 if f.direction == "bullish" else -1 if f.direction in ("bearish", "risk") else 0)
              for f in findings)
    den = sum(f.weight for f in findings) or 1.0
    return num / den


def comp_score(state: dict, name: str) -> float:
    c = state.get("decision", {}).get("components", {}).get(name, {})
    s = c.get("score")
    return float(s) if s is not None else np.nan


class Agent:
    name = "agent"
    role = ""

    def run(self, state: dict) -> AgentOutput:  # pragma: no cover - interface
        raise NotImplementedError

    def _out(self, findings: list[Finding], concerns=None, gaps=None, score: float | None = None) -> AgentOutput:
        sc = finding_score(findings) if score is None else score
        return AgentOutput(self.name, self.role, stance_from(sc), sc, findings, concerns or [], gaps or [])
